#!/usr/bin/env python3
"""Quality Auditor V1: Validación narrativa pre-renderizado.

Lee segments.json y panels.json, usa Gemini para verificar que el texto
narrativo de cada segmento coincida con la descripción visual del panel
asignado. Si detecta incongruencias, busca un panel mejor en el pool
completo de paneles descartados.

Uso:
  python quality_auditor.py --panels-dir output/capitulo_1
  python quality_auditor.py --panels-dir output/capitulo_1 --fix
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger("quality_auditor")


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Archivo no encontrado: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_segment(
    segment_text: str,
    panel_desc: str,
    panel_category: str,
    panel_stars: int,
    provider,
) -> dict:
    """Valida que el texto narrativo coincida con la descripción del panel.

    Returns:
        dict con {match, confidence, reason, suggested_panel_desc}
    """
    prompt = (
        "Eres un AUDITOR DE CONTINUIDAD de videos de manhwa.\n"
        "Debes verificar si el TEXTO NARRATIVO coincide con la "
        "DESCRIPCION VISUAL del panel asignado.\n\n"
        "TEXTO NARRATIVO:\n"
        f"{segment_text}\n\n"
        "DESCRIPCION DEL PANEL:\n"
        f"{panel_desc}\n\n"
        f"CATEGORIA: {panel_category}\n"
        f"ESTRELLAS: {panel_stars}/5\n\n"
        "Responde SOLO JSON (sin markdown):\n"
        '{\n'
        '  "match": true/false,\n'
        '  "confidence": 0.0-1.0,\n'
        '  "reason": "explicacion breve"\n'
        "}\n\n"
        '"match" es TRUE si el texto describe correctamente lo que se ve en el panel.\n'
        '"match" es FALSE si hay CONTRADICCION (ej: texto dice "esta ganando" pero panel muestra "sangrando").'
    )

    try:
        response = provider.describe_text(prompt)
        result = json.loads(response)
        return {
            "match": result.get("match", True),
            "confidence": result.get("confidence", 0.5),
            "reason": result.get("reason", ""),
        }
    except Exception as exc:
        logger.warning(f"[AUDITOR] Error validando segmento: {exc}")
        return {"match": True, "confidence": 0.0, "reason": "error de validacion"}


def find_better_panel(
    segment_text: str,
    current_panel: dict,
    all_panels: list[dict],
    provider,
) -> dict | None:
    """Busca un panel mejor en el pool completo si el actual no coincide.

    Args:
        segment_text: Texto narrativo del segmento.
        current_panel: Panel actualmente asignado.
        all_panels: Lista completa de paneles (incluyendo descartados).
        provider: LLMProvider.

    Returns:
        Panel sugerido o None si no hay mejor opcion.
    """
    current_id = current_panel.get("scene")
    candidates = [p for p in all_panels if p.get("scene") != current_id]
    if not candidates:
        return None

    # Filtra candidatos potenciales por palabras clave compartidas
    text_lower = segment_text.lower()
    text_words = set(text_lower.split())

    scored = []
    for p in candidates:
        desc = (p.get("vision_description") or "").lower()
        ocr = (p.get("ocr_text") or "").lower()
        combined = desc + " " + ocr
        overlap = len(text_words & set(combined.split()))
        stars = p.get("stars", 1)
        score = overlap * 3 + stars * 2
        scored.append((score, p))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_candidates = [p for _, p in scored[:5]]

    if not top_candidates:
        return None

    prompt = (
        "Eres un EDITOR de video de manhwa. Necesitas reemplazar un panel "
        "que NO coincide con el texto narrado.\n\n"
        f"TEXTO NARRATIVO:\n{segment_text}\n\n"
        f"PANEL ACTUAL (no coincide):\n"
        f"  ID: {current_id}\n"
        f"  Desc: {current_panel.get('vision_description', '')[:200]}\n\n"
        "PANELES CANDIDATOS:\n"
    )
    for i, p in enumerate(top_candidates):
        prompt += (
            f"  [{i}] ID:{p['scene']:04d} | ★{p.get('stars',1)} | "
            f"{p.get('category','?')} | "
            f"{p.get('vision_description','')[:150]}\n"
        )
    prompt += (
        "\nCual panel candidate coincide MEJOR con el texto narrativo?\n"
        "Responde SOLO JSON:\n"
        '{"best_index": 0, "reason": "explicacion"}\n'
        'Si NINGUNO sirve, responde {"best_index": -1, "reason": "..."}'
    )

    try:
        response = provider.describe_text(prompt)
        result = json.loads(response)
        idx = result.get("best_index", -1)
        if 0 <= idx < len(top_candidates):
            return top_candidates[idx]
    except Exception as exc:
        logger.warning(f"[AUDITOR] Error buscando panel alternativo: {exc}")

    return None


def audit_segments(
    panels_dir: str,
    provider,
    auto_fix: bool = False,
) -> list[dict]:
    """Ejecuta auditoria de calidad narrativa sobre los segmentos.

    Args:
        panels_dir: Directorio con segments.json, panels.json.
        provider: LLMProvider.
        auto_fix: Si True, intenta reasignar paneles incongruentes.

    Returns:
        Lista de hallazgos (incongruencias detectadas).
    """
    panels_dir_path = Path(panels_dir)

    # Carga datos
    segments_data = load_json(panels_dir_path / "segments.json")
    panels_data = load_json(panels_dir_path / "panels.json")

    segments = segments_data.get("segments", [])
    all_panels = panels_data.get("panels", [])
    panel_map = {p["scene"]: p for p in all_panels}

    if not segments:
        raise ValueError("segments.json no contiene segmentos")
    if not all_panels:
        raise ValueError("panels.json no contiene paneles")

    total = len(segments)
    logger.info(f"[AUDITOR] Auditanado {total} segmentos contra {len(all_panels)} paneles...")

    findings = []
    fix_count = 0
    for i, seg in enumerate(segments):
        scene_id = seg.get("panel_scene", 0)
        text = seg.get("text", "")
        panel = panel_map.get(scene_id)

        if not panel or not text:
            continue

        panel_desc = panel.get("vision_description") or panel.get("vision_error") or ""
        panel_cat = panel.get("category", "")
        panel_stars = panel.get("stars", 1)

        result = validate_segment(text, panel_desc[:300], panel_cat, panel_stars, provider)

        finding = {
            "segment_index": i,
            "scene_id": scene_id,
            "narrative_text": text[:100],
            "panel_desc": panel_desc[:100],
            "match": result["match"],
            "confidence": result["confidence"],
            "reason": result["reason"],
        }

        if not result["match"] and result["confidence"] >= 0.6:
            logger.warning(
                f"[AUDITOR] Segmento {i} (escena {scene_id:04d}): "
                f"INCONSISTENCIA ({result['confidence']:.0%}) - {result['reason']}"
            )
            findings.append(finding)

            if auto_fix:
                better = find_better_panel(text, panel, all_panels, provider)
                if better:
                    old_id = scene_id
                    new_id = better["scene"]
                    seg["panel_scene"] = new_id
                    seg["panel_file"] = better.get("file", f"escena_{new_id:04d}.png")
                    fix_count += 1
                    logger.info(f"[AUDITOR] Panel corregido: escena {old_id:04d} -> {new_id:04d}")
                    finding["fixed"] = True
                    finding["new_scene_id"] = new_id
                    finding["new_panel_desc"] = (better.get("vision_description") or "")[:100]
                else:
                    finding["fixed"] = False
        else:
            logger.info(
                f"[AUDITOR] Segmento {i} (escena {scene_id:04d}): OK "
                f"({result['confidence']:.0%})"
            )

    # Guarda reporte
    report = {
        "total_segments": total,
        "issues_found": len(findings),
        "auto_fixes": fix_count,
        "issues": findings,
        "all_ok": len(findings) == 0,
    }
    report_path = panels_dir_path / "quality_report_narrativo.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if findings:
        logger.info(f"[AUDITOR] Reporte: {len(findings)} incongruencias, {fix_count} correcciones")
    else:
        logger.info(f"[AUDITOR] Reporte: 0 incongruencias. Todo correcto.")

    # Si hubo correcciones, guarda segments actualizado
    if auto_fix and fix_count > 0:
        segments_data["segments"] = segments
        segments_path = panels_dir_path / "segments.json"
        segments_path.write_text(json.dumps(segments_data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"[AUDITOR] segments.json actualizado con {fix_count} correcciones")

    return findings


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Quality Auditor: validacion narrativa pre-renderizado"
    )
    parser.add_argument("--panels-dir", required=True, help="Directorio con segments.json y panels.json")
    parser.add_argument("--fix", action="store_true", help="Auto-corregir paneles incongruentes")
    parser.add_argument("--text-provider", default="gemini")
    parser.add_argument("--text-model", default="gemini-flash-lite-latest")
    parser.add_argument("--text-timeout", type=int, default=60)

    args = parser.parse_args()

    from llm.factory import LLMProviderFactory
    provider = LLMProviderFactory.create(
        args.text_provider,
        model=args.text_model,
        timeout=args.text_timeout,
    )

    try:
        findings = audit_segments(
            panels_dir=args.panels_dir,
            provider=provider,
            auto_fix=args.fix,
        )
        if findings:
            print(f"\n[AUDITOR] {len(findings)} incongruencias detectadas.")
            print(f"[AUDITOR] Usa --fix para auto-corregir.")
            for f in findings[:5]:
                print(f"  - Escena {f['scene_id']:04d}: {f['reason'][:100]}")
            if len(findings) > 5:
                print(f"  ... y {len(findings)-5} mas")
        else:
            print("[AUDITOR] Todo correcto. 0 incongruencias.")
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        sys.exit(f"[ERROR] {exc}")


if __name__ == "__main__":
    main()
