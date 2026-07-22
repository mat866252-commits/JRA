#!/usr/bin/env python3
"""Recorta tiras de manhwa, registra diagnóstico y genera un manifiesto."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from html import escape
import hashlib
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from vision import VisionError, create_vision


def _write_json(path: str, data: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def find_gutter_splits(gray, min_height: int, std_threshold: float, blank_ratio: float):
    row_std = gray.std(axis=1)
    white = (gray > 245).mean(axis=1)
    black = (gray < 10).mean(axis=1)
    blank = (row_std <= std_threshold) & ((white >= blank_ratio) | (black >= blank_ratio))
    ranges, start = [], None
    for index, value in enumerate(blank):
        if value and start is None:
            start = index
        elif not value and start is not None:
            if index - start >= min_height:
                ranges.append((start, index))
            start = None
    if start is not None and len(blank) - start >= min_height:
        ranges.append((start, len(blank)))
    return ranges


def is_valid_panel(panel, min_height: int, max_blank_ratio: float) -> bool:
    if panel.shape[0] < min_height:
        return False
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    blank = ((gray > 240) | (gray < 15)).mean()
    return blank <= max_blank_ratio


def panel_metrics(panel) -> dict:
    """Métricas simples y reproducibles para priorizar revisión humana."""
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    blank = float(((gray > 240) | (gray < 15)).mean())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    quality = max(0.0, min(1.0, (1 - blank) * min(1, sharpness / 120.0) * min(1, panel.shape[0] / 400)))
    return {"blank_ratio": round(blank, 4), "sharpness": round(sharpness, 2), "quality_score": round(quality, 3)}


def trim_margins(panel, threshold: int, padding: int):
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    content = np.where((gray < threshold).any(axis=1))[0]
    if not len(content): return panel
    top, bottom = max(0, content[0] - padding), min(panel.shape[0], content[-1] + padding + 1)
    return panel[top:bottom, :]


def crop_panels_from_strip(path, args, review_dir=None):
    image = cv2.imread(path)
    if image is None:
        return [], 0, {"status": "unreadable", "gutters": 0}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gutters = find_gutter_splits(gray, args.gutter_min_height, args.gutter_std_threshold, args.gutter_blank_ratio)
    if not gutters:
        if is_valid_panel(image, args.min_height, args.max_blank_ratio):
            return [image], 0, {"status": "single_panel", "gutters": 0, "candidates": 1}
        if review_dir:
            cv2.imwrite(os.path.join(review_dir, f"{os.path.splitext(os.path.basename(path))[0]}_00.png"), image)
        return [], 1, {"status": "single_panel_rejected", "gutters": 0, "candidates": 1}
    boundaries = [0, *[(start + end) // 2 for start, end in gutters], gray.shape[0]]
    panels, discarded = [], 0
    for index, (top, bottom) in enumerate(zip(boundaries, boundaries[1:])):
        if bottom - top < args.min_panel_height:
            discarded += 1
            continue
        panel = image[top:bottom, :]
        if is_valid_panel(panel, args.min_height, args.max_blank_ratio):
            panels.append(panel)
        else:
            discarded += 1
            if review_dir:
                cv2.imwrite(os.path.join(review_dir, f"{os.path.splitext(os.path.basename(path))[0]}_{index:02d}.png"), panel)
    if not panels:
        if args.on_uncertain == "full":
            return [image], 0, {"status": "fallback_full_image", "gutters": len(gutters), "candidates": len(boundaries) - 1}
        return [], discarded, {"status": "uncertain_split", "gutters": len(gutters), "candidates": len(boundaries) - 1}
    return panels, discarded, {"status": "split", "gutters": len(gutters), "candidates": len(boundaries) - 1}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-panel-height", type=int, default=120)
    parser.add_argument("--min-height", type=int, default=250)
    parser.add_argument("--max-blank-ratio", type=float, default=.85)
    parser.add_argument("--gutter-min-height", type=int, default=15)
    parser.add_argument("--gutter-std-threshold", type=float, default=6.0)
    parser.add_argument("--gutter-blank-ratio", type=float, default=.985)
    parser.add_argument("--no-filter", action="store_true")
    parser.add_argument("--review", action="store_true")
    parser.add_argument("--on-uncertain", choices=("error", "full"), default="error",
                        help="Ante una tira que no se pueda dividir: error (seguro) o usar imagen completa.")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--diagnostics", default=None)
    parser.add_argument("--vision-provider", choices=("disabled", "ollama", "huggingface"), default="disabled")
    parser.add_argument("--vision-model", default="gemma3")
    parser.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    parser.add_argument("--vision-timeout", type=int, default=90)
    parser.add_argument("--vision-workers", type=int, default=2, help="Descripciones visuales simultáneas.")
    parser.add_argument("--vision-max-side", type=int, default=1280, help="Lado máximo enviado al proveedor de visión.")
    parser.add_argument("--expected-scenes", type=int, default=None, help="Falla si el total final no coincide.")
    parser.add_argument("--trim-margins", action="store_true", help="Elimina márgenes blancos verticales.")
    parser.add_argument("--trim-threshold", type=int, default=245)
    parser.add_argument("--trim-padding", type=int, default=4)
    parser.add_argument("--webp", action="store_true", help="Guarda WebP en lugar de PNG para reducir disco.")
    parser.add_argument("--csv", default=None, help="Exporta el manifiesto a CSV para revisión.")
    parser.add_argument("--html-report", default=None, help="Informe HTML local con métricas de las viñetas.")
    args = parser.parse_args()
    if args.no_filter:
        args.min_height, args.max_blank_ratio = 0, 1.0
    if args.gutter_min_height < 1 or args.gutter_std_threshold < 0 or not 0 < args.gutter_blank_ratio <= 1:
        parser.error("Los umbrales de gutter no son válidos.")
    if args.vision_workers < 1:
        parser.error("--vision-workers debe ser mayor que 0.")
    try:
        vision = create_vision(args.vision_provider, args.vision_model, args.ollama_host, args.vision_timeout, 2, args.vision_max_side)
    except VisionError as exc:
        sys.exit(f"[ERROR] Visión no disponible: {exc}")
    source, destination = os.path.abspath(args.input), os.path.abspath(args.output)
    if not os.path.isdir(source):
        sys.exit(f"[ERROR] La carpeta de entrada no existe: {source}")
    files = sorted(item for item in os.listdir(source) if item.lower().endswith((".png", ".jpg", ".jpeg", ".webp")))
    if not files:
        sys.exit(f"[ERROR] No hay imágenes en {source}")
    os.makedirs(destination, exist_ok=True)
    if args.replace:
        for item in os.listdir(destination):
            if item.startswith("escena_") and item.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                os.unlink(os.path.join(destination, item))
    review_dir = os.path.join(destination, "descartadas") if args.review else None
    if review_dir:
        os.makedirs(review_dir, exist_ok=True)
    manifest, diagnostics, scene, discarded_total, uncertain, hashes = [], [], 1, 0, [], {}
    for filename in files:
        panels, discarded, diagnostic = crop_panels_from_strip(os.path.join(source, filename), args, review_dir)
        diagnostic.update({"source": filename, "saved": len(panels), "discarded": discarded})
        diagnostics.append(diagnostic)
        discarded_total += discarded
        if diagnostic["status"] != "split":
            print(f"[diagnóstico] {filename}: {diagnostic['status']} (gutters={diagnostic['gutters']})")
        if diagnostic["status"] in {"unreadable", "single_panel_rejected", "uncertain_split"}:
            uncertain.append(filename)
        for panel in panels:
            if args.trim_margins:
                panel = trim_margins(panel, args.trim_threshold, args.trim_padding)
            extension = "webp" if args.webp else "png"
            name = f"escena_{scene:04d}.{extension}"
            target = os.path.join(destination, name)
            params = [cv2.IMWRITE_WEBP_QUALITY, 92] if args.webp else []
            if not cv2.imwrite(target, panel, params):
                sys.exit(f"[ERROR] No se pudo guardar {target}")
            digest = hashlib.sha256(panel.tobytes()).hexdigest()
            entry = {"scene": scene, "file": name, "source": filename, "width": int(panel.shape[1]), "height": int(panel.shape[0]), "sha256": digest, **panel_metrics(panel)}
            if digest in hashes: entry["duplicate_of"] = hashes[digest]
            else: hashes[digest] = scene
            entry["low_confidence"] = (
                entry.get("blank_ratio", 0) > 0.70
                or entry.get("quality_score", 1) < 0.25
                or entry.get("sharpness", 999) < 0.5
            )
            manifest.append(entry)
            scene += 1
        print(f"{filename}: {len(panels)} paneles, {discarded} descartados")
    if vision and manifest:
        prompt = "Describe la viñeta en español en una frase: personajes, acción y lugar visibles."
        with ThreadPoolExecutor(max_workers=args.vision_workers) as executor:
            futures = {executor.submit(vision.describe, os.path.join(destination, entry["file"]), prompt): entry for entry in manifest}
            for future in as_completed(futures):
                entry = futures[future]
                try:
                    entry["vision_description"] = future.result()
                except VisionError as exc:
                    entry["vision_error"] = str(exc)
                except Exception as exc:
                    entry["vision_error"] = f"Error inesperado de visión: {exc}"
    manifest_path = args.manifest or os.path.join(destination, "panels.json")
    _write_json(manifest_path, {"panels": manifest, "discarded": discarded_total, "settings": vars(args)})
    diagnostics_path = args.diagnostics or os.path.join(destination, "crop_diagnostics.jsonl")
    with open(diagnostics_path, "w", encoding="utf-8") as handle:
        for item in diagnostics:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    csv_path = args.csv or os.path.join(destination, "panels.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        fields = ["scene", "file", "source", "width", "height", "quality_score", "blank_ratio", "sharpness", "duplicate_of", "vision_description"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(manifest)
    report_path = args.html_report or os.path.join(destination, "crop_report.html")
    rows = "\n".join(
        f"<tr><td>{entry['scene']:04d}</td><td><img src='{escape(entry['file'], quote=True)}'></td>"
        f"<td>{escape(entry['source'])}</td><td>{entry['quality_score']}</td>"
        f"<td>{escape(str(entry.get('duplicate_of', '')))}</td>"
        f"<td>{escape(entry.get('vision_description', entry.get('vision_error', '')))}</td></tr>"
        for entry in manifest
    )
    Path(report_path).write_text("<!doctype html><meta charset='utf-8'><style>body{font-family:sans-serif}img{max-width:180px;max-height:160px}td{padding:6px;border-bottom:1px solid #ddd;vertical-align:top}</style><h1>Diagnóstico de viñetas</h1><table><tr><th>Escena</th><th>Viñeta</th><th>Origen</th><th>Calidad</th><th>Duplicada</th><th>Descripción</th></tr>" + rows + "</table>", encoding="utf-8")
    print(f"Total: {len(manifest)} paneles. Manifiesto: {manifest_path}. Diagnóstico: {diagnostics_path}")
    if uncertain:
        sys.exit("[ERROR] Recortes inciertos: " + ", ".join(uncertain) + ". Revisa diagnostics o usa --on-uncertain full conscientemente.")
    if args.expected_scenes is not None and len(manifest) != args.expected_scenes:
        sys.exit(f"[ERROR] Se esperaban {args.expected_scenes} escenas, se obtuvieron {len(manifest)} paneles.")


def crop_all_pages(input_dir: Path, output_dir: Path) -> list[dict]:
    """Wrapper: recorta vinetas desde imagenes y devuelve el manifiesto en memoria."""
    import argparse

    args = argparse.Namespace()
    args.input = str(input_dir)
    args.output = str(output_dir)
    args.min_panel_height = 250
    args.min_height = 400
    args.max_blank_ratio = 0.50
    args.gutter_min_height = 35
    args.gutter_std_threshold = 12.0
    args.gutter_blank_ratio = 0.995
    args.no_filter = False
    args.review = False
    args.on_uncertain = "error"
    args.replace = True
    args.manifest = None
    args.diagnostics = None
    args.vision_provider = "disabled"
    args.vision_model = "gemma3"
    args.ollama_host = "http://127.0.0.1:11434"
    args.vision_timeout = 90
    args.vision_workers = 2
    args.vision_max_side = 1280
    args.expected_scenes = None
    args.trim_margins = False
    args.trim_threshold = 245
    args.trim_padding = 4
    args.webp = False
    args.csv = None
    args.html_report = None

    source, destination = os.path.abspath(args.input), os.path.abspath(args.output)
    if not os.path.isdir(source):
        raise FileNotFoundError(f"El directorio de entrada no existe: {source}")
    files = sorted(item for item in os.listdir(source) if item.lower().endswith((".png", ".jpg", ".jpeg", ".webp")))
    if not files:
        raise FileNotFoundError(f"No hay imagenes en {source}")
    os.makedirs(destination, exist_ok=True)

    review_dir = os.path.join(destination, "descartadas") if args.review else None
    if review_dir:
        os.makedirs(review_dir, exist_ok=True)

    manifest, scene, hashes = [], 1, {}
    for filename in files:
        panels, discarded, diagnostic = crop_panels_from_strip(os.path.join(source, filename), args, review_dir)
        if diagnostic["status"] in ("unreadable", "single_panel_rejected", "uncertain_split"):
            continue
        for panel in panels:
            if args.trim_margins:
                panel = trim_margins(panel, args.trim_threshold, args.trim_padding)
            extension = "webp" if args.webp else "png"
            name = f"escena_{scene:04d}.{extension}"
            target = os.path.join(destination, name)
            params = [cv2.IMWRITE_WEBP_QUALITY, 92] if args.webp else []
            if not cv2.imwrite(target, panel, params):
                raise RuntimeError(f"No se pudo guardar {target}")
            digest = hashlib.sha256(panel.tobytes()).hexdigest()
            entry = {"scene": scene, "file": name, "source": filename, "width": int(panel.shape[1]),
                     "height": int(panel.shape[0]), "sha256": digest, **panel_metrics(panel)}
            if digest in hashes:
                entry["duplicate_of"] = hashes[digest]
            else:
                hashes[digest] = scene
            manifest.append(entry)
            scene += 1

    manifest_path = os.path.join(destination, "panels.json")
    _write_json(manifest_path, {"panels": manifest, "discarded": 0})

    return manifest


from pipeline_steps_base import PipelineStep, PipelineContext as PipelineCtx


class CropPanelsStep(PipelineStep):
    def __init__(self):
        super().__init__("crop_panels")

    def validate_contract(self, context: PipelineCtx) -> bool:
        from pathlib import Path
        input_path = Path(context.input_dir)
        if not input_path.exists() or not input_path.is_dir():
            self.logger.error(f"Contrato violado: El directorio de entrada no existe: {context.input_dir}")
            return False
        files = [f for f in input_path.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".pdf")]
        if not files:
            self.logger.error(f"Contrato violado: No hay imagenes en el directorio de entrada: {context.input_dir}")
            return False
        return True

    def should_skip(self, context: PipelineCtx) -> bool:
        if super().should_skip(context):
            return True
        if getattr(context, 'force', False):
            return False
        return (Path(context.output_dir) / "panels.json").is_file()

    def execute(self, context: PipelineCtx) -> bool:
        if self.should_skip(context):
            self.logger.info(f"Saltando fase {self.name}")
            return True
        try:
            self.logger.info(f"Iniciando recorte de vinetas para: {context.project_name}")
            from pdf_panels import crop_panels_from_pdf
            import argparse
            import glob
            pdf_files = glob.glob(os.path.join(context.input_dir, "*.pdf"))
            if not pdf_files:
                self.logger.error(f"No se encontro ningun archivo PDF en: {context.input_dir}")
                return False
            pdf_path = pdf_files[0]
            self.logger.info(f"Procesando PDF: {pdf_path}")
            ns = argparse.Namespace()
            ns.output = context.output_dir
            ns.replace = True
            ns.dpi = 150
            ns.detection_mode = "heuristic"
            ns.min_panel_height = 300
            ns.min_height = 300
            ns.max_blank_ratio = 0.40
            ns.gutter_min_height = 35
            ns.gutter_std_threshold = 12
            ns.gutter_blank_ratio = 0.995
            ns.fusion_min_confidence = 0.85
            ns.fusion_strip = 20
            ns.on_uncertain = "full"
            ns.vision_provider = "disabled"
            ns.vision_model = ""
            ns.ollama_host = "http://127.0.0.1:11434"
            ns.vision_timeout = 120
            ns.vision_workers = 4
            ns.vision_max_side = 1280
            ns.no_discard = False
            ns.discard_score_threshold = 0.5
            ns.review = False
            ns.ocr = True
            ns.ocr_languages = ["es", "en"]
            ns.expected_scenes = None
            ns.trim_margins = True
            ns.trim_threshold = 245
            ns.trim_padding = 4
            ns.webp = False
            ns.no_filter = False
            ns.manifest = None
            ns.diagnostics = None
            ns.fusion_log = None
            ns.csv = None
            ns.html_report = None
            panels_list, _, _, discarded = crop_panels_from_pdf(pdf_path, ns)

            _write_json(
                os.path.join(context.output_dir, "panels.json"),
                {"panels": panels_list, "discarded": discarded}
            )
            self.logger.info(f"Recortadas {len(panels_list)} vinetas")
            return True
        except Exception as e:
            return self.on_error(context, e)


if __name__ == "__main__":
    main()
