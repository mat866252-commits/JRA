#!/usr/bin/env python3
"""Batch Pipeline V1: Procesa múltiples capítulos en paralelo y construye un arco.

Escanea un directorio con N PDFs, lanza un pipeline por cada uno (en paralelo),
y al final concatená todos los videos en un arco único con transiciones.

Uso:
    python batch_pipeline.py --input-dir input_capitulos/ --output-dir output/ \\
        --project "Solo Leveling" --target-duration 3600 --workers 4
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("batch_pipeline")

STATE_FILE = "batch_state.json"


def discover_pdfs(input_dir: Path, sort_key: str = "numeric") -> list[Path]:
    """Encuentra todos los PDFs en input_dir ordenados.

    Args:
        input_dir: Directorio a escanear.
        sort_key: 'numeric' (default) extrae número, 'alpha' orden alfabético.

    Returns:
        Lista de rutas de PDFs ordenadas.
    """
    pdfs = sorted(input_dir.glob("*.pdf"))
    if not pdfs:
        pdfs = sorted(input_dir.glob("*.PDF"))

    if sort_key == "numeric":
        import re
        def num_key(p: Path) -> int:
            match = re.search(r"(\d+)", p.stem)
            return int(match.group(1)) if match else 0
        pdfs.sort(key=num_key)

    return pdfs


def run_single_pipeline(pdf_path: str, output_base: str, args) -> dict:
    """Ejecuta run_pipeline.py para un solo PDF y devuelve el resultado."""
    pdf = Path(pdf_path)
    chapter_num = _detect_number(pdf)
    project_name = f"{args.project_prefix}_{chapter_num:03d}"
    output_dir = Path(output_base) / project_name
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline_script = str(Path(__file__).resolve().parent / "run_pipeline.py")
    cmd = [
        sys.executable, pipeline_script,
        "--project", project_name,
        "--input-dir", str(pdf.parent),
        "--output-dir", str(output_dir),
        "--chapter", str(chapter_num),
        "--text-provider", args.text_provider,
        "--text-model", args.text_model,
        "--text-timeout", str(args.text_timeout),
    ]

    if args.pdf:
        cmd.append("--pdf")
    if args.resolution:
        cmd.extend(["--resolution", args.resolution])
    if args.fps:
        cmd.extend(["--fps", str(args.fps)])
    if args.crf:
        cmd.extend(["--crf", str(args.crf)])
    if args.encoder:
        cmd.extend(["--encoder", args.encoder])
    if args.music_base_dir:
        cmd.extend(["--music-base-dir", args.music_base_dir])
    if args.quality_audit:
        cmd.append("--quality-audit")
    if args.quality_audit_fix:
        cmd.append("--quality-audit-fix")
    if args.thumbnail:
        cmd.append("--thumbnail")
    if args.target_duration:
        cmd.extend(["--target-duration", str(args.target_duration)])
    if args.crossfade:
        cmd.extend(["--crossfade", str(args.crossfade)])
    if args.max_workers:
        cmd.extend(["--max-workers", str(args.max_workers)])
    if args.preset:
        cmd.extend(["--preset", args.preset])

    logger.info(f"[BATCH] Pipeline para capítulo {chapter_num}: {pdf.name}")
    logger.info(f"[BATCH] Comando: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=args.pipeline_timeout)
        video_final = output_dir / "video_final.mp4"
        success = result.returncode == 0 and video_final.is_file()

        if success:
            size_mb = video_final.stat().st_size / 1024 / 1024
            logger.info(f"[BATCH] Capítulo {chapter_num} OK: {video_final} ({size_mb:.0f} MB)")
        else:
            logger.error(f"[BATCH] Capítulo {chapter_num} FALLÓ: {result.stderr[-1000:]}")

        return {
            "chapter": chapter_num,
            "source": str(pdf),
            "output_dir": str(output_dir),
            "video": str(video_final) if video_final.is_file() else None,
            "success": success,
            "timestamp": datetime.now().isoformat(),
        }
    except subprocess.TimeoutExpired:
        logger.error(f"[BATCH] Capítulo {chapter_num} TIMEOUT ({args.pipeline_timeout}s)")
        return {"chapter": chapter_num, "source": str(pdf), "output_dir": str(output_dir),
                "video": None, "success": False, "error": "timeout"}
    except Exception as exc:
        logger.error(f"[BATCH] Capítulo {chapter_num} ERROR: {exc}")
        return {"chapter": chapter_num, "source": str(pdf), "output_dir": str(output_dir),
                "video": None, "success": False, "error": str(exc)}


def _detect_number(item: Path) -> int:
    import re
    match = re.search(r"(\d+)", item.stem)
    return int(match.group(1)) if match else 1


def run_batch(args) -> None:
    """Ejecuta el batch completo: descubre PDFs -> paralelo -> arco."""
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdfs = discover_pdfs(input_dir, args.sort_key)
    if not pdfs:
        logger.error(f"No se encontraron PDFs en {input_dir}")
        sys.exit(1)

    logger.info(f"[BATCH] {len(pdfs)} PDFs encontrados en {input_dir}")
    for p in pdfs:
        logger.info(f"  - {p.name}")

    # Filtrar ya procesados si --resume
    state_path = output_dir / STATE_FILE
    state = {"processed": [], "results": []}
    if args.resume and state_path.is_file():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            processed = set(state.get("processed", []))
            pdfs = [p for p in pdfs if p.name not in processed]
            logger.info(f"[BATCH] Resume: {len(processed)} ya procesados, {len(pdfs)} pendientes")
        except Exception:
            pass

    if not pdfs:
        logger.info("[BATCH] Todos los PDFs ya fueron procesados")
        return

    results: list[dict] = []
    target_duration = args.target_duration or (len(pdfs) * 600)

    if args.max_workers > 1 and len(pdfs) > 1:
        logger.info(f"[BATCH] Procesando {len(pdfs)} capítulos en paralelo "
                    f"({args.max_workers} workers)")
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(run_single_pipeline, str(p), str(output_dir), args): p
                for p in pdfs
            }
            for future in as_completed(futures):
                pdf_name = futures[future].name
                try:
                    result = future.result()
                    results.append(result)
                    state["processed"].append(pdf_name)
                    state["results"].append(result)
                    state_path.write_text(
                        json.dumps(state, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                except Exception as exc:
                    logger.error(f"[BATCH] Error en {pdf_name}: {exc}")
    else:
        logger.info(f"[BATCH] Procesando {len(pdfs)} capítulos secuencialmente")
        for pdf in pdfs:
            result = run_single_pipeline(str(pdf), str(output_dir), args)
            results.append(result)
            state["processed"].append(pdf.name)
            state["results"].append(result)
            state_path.write_text(
                json.dumps(state, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    # Reporte
    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]
    logger.info(f"[BATCH] Completados: {len(successful)}/{len(results)} "
                f"({len(failed)} fallos)")

    if failed:
        logger.warning(f"[BATCH] Capítulos fallidos: {[r['chapter'] for r in failed]}")

    # Construir arco final si hay suficientes videos
    if args.skip_arc:
        logger.info("[BATCH] --skip-arc activado, omitiendo construcción del arco")
        return

    videos = [r["video"] for r in successful if r.get("video")]
    if len(videos) < 2:
        logger.warning(f"[BATCH] {len(videos)} videos disponibles, insuficientes para arco")
        if videos:
            import shutil
            arc_path = str(output_dir / "arc_final.mp4")
            shutil.copy2(videos[0], arc_path)
            logger.info(f"[BATCH] Solo 1 video, copiado a: {arc_path}")
        return

    logger.info(f"[BATCH] Construyendo arco con {len(videos)} videos...")

    from arc_builder import build_arc_from_videos

    source_names = []
    for r in successful:
        if r.get("video"):
            source_names.append(f"Capítulo {r['chapter']}")

    arc_path = str(output_dir / "arc_final.mp4")
    try:
        build_arc_from_videos(
            video_paths=videos,
            output_path=arc_path,
            transition_duration=args.transition_duration,
            resolution=args.resolution,
            fps=args.fps,
            crf=args.crf,
            preset=args.preset,
            source_names=source_names,
        )
        logger.info(f"[BATCH] Arco final: {arc_path}")
    except Exception as exc:
        logger.error(f"[BATCH] Error construyendo arco: {exc}")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Batch Pipeline: procesa N capítulos en paralelo y construye arco",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input-dir", required=True,
                        help="Directorio con los PDFs de los capítulos")
    parser.add_argument("--output-dir", required=True,
                        help="Directorio de salida para pipelines y arco final")
    parser.add_argument("--project-prefix", default="capitulo",
                        help="Prefijo para nombres de proyecto (default: capitulo)")
    parser.add_argument("--sort-key", default="numeric",
                        choices=["numeric", "alpha"],
                        help="Orden de procesamiento (default: numeric)")
    parser.add_argument("--target-duration", type=float, default=3600.0,
                        help="Duración objetivo en segundos (default 3600 = 1h)")
    parser.add_argument("--workers", dest="max_workers", type=int, default=4,
                        help="Máximo de pipelines paralelos (default 4)")
    parser.add_argument("--pipeline-timeout", type=int, default=3600,
                        help="Timeout por pipeline en segundos (default 3600)")
    parser.add_argument("--skip-arc", action="store_true",
                        help="No construir el arco final")
    parser.add_argument("--resume", action="store_true",
                        help="Reanudar desde el estado guardado")
    parser.add_argument("--transition-duration", type=float, default=1.5)
    parser.add_argument("--pdf", action="store_true")
    parser.add_argument("--resolution", default="1920x1080")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--crf", type=int, default=23)
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--encoder", default="libx264")
    parser.add_argument("--text-provider", default="gemini")
    parser.add_argument("--text-model", default="gemini-flash-lite-latest")
    parser.add_argument("--text-timeout", type=int, default=120)
    parser.add_argument("--music-base-dir", default=None)
    parser.add_argument("--quality-audit", action="store_true")
    parser.add_argument("--quality-audit-fix", action="store_true")
    parser.add_argument("--thumbnail", action="store_true")
    parser.add_argument("--crossfade", type=int, default=12)

    args = parser.parse_args()
    run_batch(args)


if __name__ == "__main__":
    main()
