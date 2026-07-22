#!/usr/bin/env python3
"""Orchestrator V1: Fabrica automatica de videos.

Escanea un directorio cada N segundos en busca de nuevos PDFs/carpetas,
dispara el pipeline automaticamente, y opcionalmente sube a YouTube.

Uso:
  python orchestrator.py --watch input_capitulos/ --output output/
  python orchestrator.py --watch input_capitulos/ --once  # un solo ciclo
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("orchestrator")

STATE_FILE = "orchestrator_state.json"


def load_state(state_path: str) -> dict:
    path = Path(state_path)
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            # Ensure pending exists for batch mode
            if "pending" not in data:
                data["pending"] = []
            return data
        except (OSError, json.JSONDecodeError):
            pass
    return {"processed": [], "last_scan": None, "history": [], "pending": [], "batch_count": 0}


def save_state(state_path: str, state: dict) -> None:
    Path(state_path).write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def scan_for_new_chapters(watch_dir: Path, state: dict, extensions: set[str]) -> list[Path]:
    """Escanea el directorio vigilado en busca de nuevos PDFs/carpetas.

    Returns:
        Lista de rutas de capitulos nuevos (no procesados antes).
    """
    if not watch_dir.is_dir():
        logger.warning(f"Directorio vigilado no existe: {watch_dir}")
        return []

    processed = set(state.get("processed", []))

    # Busca archivos PDF
    new_items: list[Path] = []
    for ext in extensions:
        for f in sorted(watch_dir.glob(f"*{ext}")):
            key = f.name
            if key not in processed:
                new_items.append(f)

    # Busca subdirectorios (carpetas de imagenes)
    for d in sorted(watch_dir.iterdir()):
        if d.is_dir() and not d.name.startswith("_"):
            key = d.name
            if key not in processed:
                new_items.append(d)

    return new_items


def detect_chapter_number(item: Path) -> int:
    """Extrae numero de capitulo del nombre del archivo/carpeta."""
    import re
    match = re.search(r"(\d+)", item.stem)
    if match:
        return int(match.group(1))
    return 1


def run_pipeline(item: Path, args) -> bool:
    """Ejecuta run_pipeline.py para un capitulo.

    Args:
        item: Ruta al PDF o carpeta del capitulo.
        args: Argumentos del orquestador.

    Returns:
        True si el pipeline se completo exitosamente.
    """
    chapter_num = detect_chapter_number(item)
    project_name = f"{args.project_prefix}_{chapter_num:03d}"
    output_dir = Path(args.output_dir) / project_name
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline_script = str(Path(__file__).resolve().parent / "run_pipeline.py")
    cmd = [
        sys.executable, pipeline_script,
        "--project", project_name,
        "--input-dir", str(item) if item.is_dir() else str(item.parent),
        "--output-dir", str(output_dir),
        "--chapter", str(chapter_num),
        "--text-provider", args.text_provider,
        "--text-model", args.text_model,
        "--text-timeout", str(args.text_timeout),
    ]

    if args.providers_config:
        cmd.extend(["--providers-config", args.providers_config])
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

    logger.info(f"[ORCHESTRATOR] Ejecutando pipeline para: {item.name}")
    logger.info(f"[ORCHESTRATOR] Comando: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=args.pipeline_timeout,
        )
        if result.returncode == 0:
            logger.info(f"[ORCHESTRATOR] Pipeline completado: {item.name}")
            # Busca video final
            video_final = output_dir / "video_final.mp4"
            thumbnail = output_dir / "thumbnail_final.png"
            report = {
                "chapter": chapter_num,
                "source": str(item),
                "output_dir": str(output_dir),
                "video": str(video_final) if video_final.is_file() else None,
                "thumbnail": str(thumbnail) if thumbnail.is_file() else None,
                "timestamp": datetime.now().isoformat(),
                "status": "success",
            }
            if video_final.is_file():
                size_mb = video_final.stat().st_size / 1024 / 1024
                logger.info(f"[ORCHESTRATOR] Video: {video_final} ({size_mb:.0f} MB)")
            return True
        else:
            logger.error(f"[ORCHESTRATOR] Pipeline FALLIDO: {item.name}")
            logger.error(f"[ORCHESTRATOR] stderr: {result.stderr[-1000:]}")
            return False
    except subprocess.TimeoutExpired:
        logger.error(f"[ORCHESTRATOR] Pipeline TIMEOUT: {item.name}")
        return False
    except Exception as exc:
        logger.error(f"[ORCHESTRATOR] Error: {exc}")
        return False


def upload_video(chapter_result: dict, args) -> bool:
    """Sube el video a YouTube usando upload_youtube.py.

    Args:
        chapter_result: Resultado del pipeline (dict con video, thumbnail, etc).
        args: Argumentos del orquestador.

    Returns:
        True si la subida fue exitosa.
    """
    video_path = chapter_result.get("video")
    if not video_path or not Path(video_path).is_file():
        logger.warning(f"[ORCHESTRATOR] No hay video para subir: {chapter_result.get('source')}")
        return False

    if not args.youtube_secrets or not Path(args.youtube_secrets).is_file():
        logger.warning("[ORCHESTRATOR] No hay credenciales YouTube (--youtube-secrets)")
        return False

    chapter_num = chapter_result.get("chapter", 1)
    title = args.youtube_title_format.format(
        chapter=chapter_num,
        project=args.project_prefix,
        date=datetime.now().strftime("%Y-%m-%d"),
    )
    description = args.youtube_description.format(
        chapter=chapter_num,
        project=args.project_prefix,
    )

    cmd = [
        sys.executable, "upload_youtube.py",
        "--video", video_path,
        "--title", title,
        "--description", description,
        "--client-secrets", args.youtube_secrets,
        "--token", str(Path(args.output_dir) / "youtube_token.json"),
        "--privacy", args.youtube_privacy,
    ]

    thumbnail_path = chapter_result.get("thumbnail")
    if thumbnail_path and Path(thumbnail_path).is_file():
        cmd.extend(["--thumbnail", thumbnail_path])

    if args.youtube_playlist:
        cmd.extend(["--playlist", args.youtube_playlist])

    logger.info(f"[ORCHESTRATOR] Subiendo a YouTube: {title}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=args.upload_timeout)
        if result.returncode == 0:
            logger.info(f"[ORCHESTRATOR] YouTube OK: {result.stdout.strip()}")
            return True
        else:
            logger.error(f"[ORCHESTRATOR] YouTube FALLIDO: {result.stderr[-500:]}")
            return False
    except Exception as exc:
        logger.error(f"[ORCHESTRATOR] Error subiendo: {exc}")
        return False


def orchestrate(args) -> None:
    """Loop principal del orquestador."""
    watch_dir = Path(args.watch_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state_path = str(output_dir / STATE_FILE)
    state = load_state(state_path)

    extensions = {".pdf"} | set(args.extensions.split(",")) if args.extensions else {".pdf"}
    extensions = {e.strip().lower() if e.startswith(".") else f".{e.strip().lower()}" for e in extensions}

    batch_mode = getattr(args, "batch_mode", False)
    batch_workers = getattr(args, "batch_workers", 4)
    batch_preset = getattr(args, "batch_preset", "medium")

    logger.info(f"[ORCHESTRATOR] Vigilando: {watch_dir}")
    logger.info(f"[ORCHESTRATOR] Extensiones: {extensions}")
    logger.info(f"[ORCHESTRATOR] Output: {output_dir}")
    logger.info(f"[ORCHESTRATOR] Intervalo: {args.interval}s")
    if batch_mode:
        logger.info(f"[ORCHESTRATOR] Modo BATCH: acumulando hasta {args.target_duration}s de contenido")

    while True:
        try:
            new_items = scan_for_new_chapters(watch_dir, state, extensions)

            if new_items:
                if batch_mode:
                    # En modo batch, acumula items sin procesar
                    for item in new_items:
                        logger.info(f"[ORCHESTRATOR] Nuevo capitulo en cola: {item.name}")
                        state["processed"].append(item.name)
                        state["pending"].append({
                            "name": item.name,
                            "path": str(item),
                            "chapter": detect_chapter_number(item),
                            "timestamp": datetime.now().isoformat(),
                        })
                    save_state(state_path, state)

                    # Verifica si ya hay suficientes capítulos acumulados
                    chapters_per_hour = max(1, int(args.target_duration / 600))
                    if len(state.get("pending", [])) >= chapters_per_hour:
                        logger.info(f"[ORCHESTRATOR] {len(state['pending'])} capítulos acumulados. "
                                    f"Lanzando batch pipeline...")
                        _run_batch_pipeline(state, output_dir, args,
                                            batch_workers, batch_preset)
                        state["pending"] = []
                        state["batch_count"] = state.get("batch_count", 0) + 1
                        save_state(state_path, state)
                else:
                    for item in new_items:
                        logger.info(f"[ORCHESTRATOR] Nuevo capitulo detectado: {item.name}")
                        success = run_pipeline(item, args)

                        chapter_num = detect_chapter_number(item)
                        entry = {
                            "name": item.name,
                            "path": str(item),
                            "chapter": chapter_num,
                            "timestamp": datetime.now().isoformat(),
                            "status": "success" if success else "failed",
                        }
                        state["processed"].append(item.name)
                        state["history"].append(entry)

                        # Subida opcional a YouTube
                        if success and args.youtube_secrets:
                            video_path = output_dir / f"{args.project_prefix}_{chapter_num:03d}" / "video_final.mp4"
                            thumb_path = output_dir / f"{args.project_prefix}_{chapter_num:03d}" / "thumbnail_final.png"
                            chapter_result = {
                                "chapter": chapter_num,
                                "source": str(item),
                                "video": str(video_path) if video_path.is_file() else None,
                                "thumbnail": str(thumb_path) if thumb_path.is_file() else None,
                            }
                            if chapter_result["video"]:
                                upload_video(chapter_result, args)

                        save_state(state_path, state)

                logger.info(f"[ORCHESTRATOR] Ciclo completado. Procesados: {len(new_items)} items.")
            else:
                pending_count = len(state.get("pending", []))
                msg = f"{len(state['processed'])} items procesados hasta ahora."
                if batch_mode and pending_count:
                    msg += f" {pending_count} pendientes en cola batch."
                logger.info(f"[ORCHESTRATOR] Sin novedades. {msg}")

            if args.once:
                break

            time.sleep(args.interval)

        except KeyboardInterrupt:
            logger.info("[ORCHESTRATOR] Detenido por el usuario.")
            save_state(state_path, state)
            break
        except Exception as exc:
            logger.error(f"[ORCHESTRATOR] Error en ciclo: {exc}", exc_info=True)
            if args.once:
                break
            time.sleep(60)


def _run_batch_pipeline(state: dict, output_dir: Path, args, workers: int, preset: str) -> None:
    """Ejecuta batch_pipeline.py con los capítulos acumulados."""
    pending = state.get("pending", [])
    if not pending:
        logger.warning("[ORCHESTRATOR] No hay capítulos pendientes para el batch")
        return

    logger.info(f"[ORCHESTRATOR] Ejecutando batch con {len(pending)} capítulos")

    batch_input = output_dir / "_batch_input"
    batch_input.mkdir(parents=True, exist_ok=True)

    pdfs_found = []
    for item in pending:
        item_path = Path(item["path"])
        if item_path.is_dir():
            pdfs = list(item_path.glob("*.pdf"))
            pdfs_found.extend(pdfs)
        elif item_path.suffix.lower() == ".pdf":
            pdfs_found.append(item_path)
        else:
            logger.warning(f"[ORCHESTRATOR] Item no reconocido: {item_path}")

    cmd = [
        sys.executable, "batch_pipeline.py",
        "--input-dir", str(batch_input),
        "--output-dir", str(output_dir / "_batch_output"),
        "--project-prefix", args.project_prefix,
        "--target-duration", str(getattr(args, "target_duration", 3600)),
        "--workers", str(workers),
        "--pipeline-timeout", str(getattr(args, "pipeline_timeout", 3600)),
        "--resolution", args.resolution,
        "--fps", str(args.fps),
        "--crf", str(args.crf),
        "--preset", preset,
        "--encoder", args.encoder,
        "--text-provider", args.text_provider,
        "--text-model", args.text_model,
        "--text-timeout", str(args.text_timeout),
    ]
    if args.music_base_dir:
        cmd.extend(["--music-base-dir", args.music_base_dir])

    # Copia los PDFs al directorio batch temporal
    for pdf in pdfs_found:
        import shutil
        shutil.copy2(str(pdf), str(batch_input / pdf.name))

    logger.info(f"[ORCHESTRATOR] Comando batch: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                 timeout=getattr(args, "pipeline_timeout", 7200) * 2)
        if result.returncode:
            logger.error(f"[ORCHESTRATOR] Batch falló: {result.stderr[-1500:]}")
        else:
            logger.info(f"[ORCHESTRATOR] Batch completado: {result.stdout[-500:]}")
    except Exception as exc:
        logger.error(f"[ORCHESTRATOR] Error en batch: {exc}")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Orchestrator: fabrica automatica de videos de manhwa",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--watch-dir", required=True,
                        help="Directorio a vigilar (ej: input_capitulos/)")
    parser.add_argument("--output-dir", required=True,
                        help="Directorio de salida para los pipelines")
    parser.add_argument("--project-prefix", default="capitulo",
                        help="Prefijo para nombres de proyecto (default: capitulo)")
    parser.add_argument("--interval", type=int, default=3600,
                        help="Intervalo de escaneo en segundos (default: 3600 = 1 hora)")
    parser.add_argument("--once", action="store_true",
                        help="Ejecuta un solo ciclo y termina")
    parser.add_argument("--extensions", default=".pdf",
                        help="Extensiones a vigilar separadas por coma (default: .pdf)")

    # Pipeline options
    parser.add_argument("--pdf", action="store_true",
                        help="Los items son archivos PDF individuales")
    parser.add_argument("--resolution", default="1920x1080")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--crf", type=int, default=23)
    parser.add_argument("--encoder", default="libx264")
    parser.add_argument("--text-provider", default="gemini")
    parser.add_argument("--text-model", default="gemini-flash-lite-latest")
    parser.add_argument("--text-timeout", type=int, default=120)
    parser.add_argument("--providers-config", default=None,
                        help="Ruta al archivo providers.yaml (cloud/local)")
    parser.add_argument("--music-base-dir", default=None,
                        help="Carpeta con musica por intensidad")
    parser.add_argument("--quality-audit", action="store_true")
    parser.add_argument("--quality-audit-fix", action="store_true")
    parser.add_argument("--thumbnail", action="store_true")
    parser.add_argument("--target-duration", type=float, default=600.0)
    parser.add_argument("--crossfade", type=int, default=12)
    parser.add_argument("--pipeline-timeout", type=int, default=1800,
                        help="Timeout del pipeline en segundos (default: 1800)")
    parser.add_argument("--batch-mode", action="store_true",
                        help="Acumula capítulos hasta alcanzar target-duration total y dispara batch_pipeline.py")
    parser.add_argument("--batch-workers", type=int, default=4,
                        help="Workers para batch pipeline (default 4)")
    parser.add_argument("--batch-preset", default="medium",
                        help="Preset FFmpeg para batch")

    # YouTube upload options
    parser.add_argument("--youtube-secrets", default=None,
                        help="Ruta a client_secrets.json para subida automatica")
    parser.add_argument("--youtube-privacy", default="private",
                        choices=["private", "unlisted", "public"])
    parser.add_argument("--youtube-title-format", default="Resumen Manhwa - Capitulo {chapter} | {project}",
                        help="Formato del titulo (placeholders: {chapter}, {project}, {date})")
    parser.add_argument("--youtube-description", default="Resumen del capitulo {chapter} de {project}. Video generado automaticamente.",
                        help="Descripcion del video")
    parser.add_argument("--youtube-playlist", default=None,
                        help="ID de playlist de YouTube para anadir el video")
    parser.add_argument("--upload-timeout", type=int, default=3600,
                        help="Timeout de subida en segundos (default: 3600)")

    args = parser.parse_args()
    orchestrate(args)


if __name__ == "__main__":
    main()
