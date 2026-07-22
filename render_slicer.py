#!/usr/bin/env python3
"""Render Slicer V1: Parte segments.json en N chunks y los renderiza en paralelo.

Divide un video largo en segmentos de duración uniforme, lanza N procesos FFmpeg
simultáneos y concatena el resultado con ``-c copy`` (sin recodificación).

Uso:
    python render_slicer.py --panels panels/ --segments segments.json \\
        --audio full_audio.mp3 --output final.mp4 --chunks 6 --workers 4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def _seg_duration(s: dict) -> float:
    return s.get("duration_sec", s.get("end", 10) - s.get("start", 0))


def split_segments(segments_path: str, n_chunks: int) -> list[list[dict]]:
    """Divide la lista de segmentos en N chunks de duración aproximadamente igual.

    Usa ``end - start`` como peso para que cada chunk tenga ~misma duración
    total de video, no necesariamente el mismo número de segmentos.
    """
    segments = json.loads(Path(segments_path).read_text(encoding="utf-8"))
    segs = segments.get("segments", [])
    if not segs:
        return []

    total_dur = sum(_seg_duration(s) for s in segs)
    target_per_chunk = total_dur / n_chunks
    chunks: list[list[dict]] = [[] for _ in range(n_chunks)]
    chunk_idx = 0
    acc = 0.0

    for seg in segs:
        dur = _seg_duration(seg)
        if chunk_idx < n_chunks - 1 and acc + dur > target_per_chunk * 1.15:
            chunk_idx += 1
            acc = 0.0
        chunks[chunk_idx].append(seg)
        acc += dur

    return chunks


def write_chunk_files(chunks: list[list[dict]], temp_dir: str,
                      original_segments_path: str) -> list[str]:
    """Escribe un segments_chunk_N.json por cada chunk en temp_dir."""
    original = json.loads(Path(original_segments_path).read_text(encoding="utf-8"))
    paths = []
    for i, chunk in enumerate(chunks):
        chunk_data = dict(original)
        chunk_data["segments"] = chunk
        chunk_data["total_duration"] = sum(_seg_duration(s) for s in chunk)
        path = Path(temp_dir) / f"segments_chunk_{i:04d}.json"
        path.write_text(json.dumps(chunk_data, ensure_ascii=False, indent=2), encoding="utf-8")
        paths.append(str(path))
    return paths


def render_chunk(panels_dir: str, segments_chunk: str, audio_path: str,
                 output_path: str, resolution: str, fps: int, crf: int,
                 preset: str, encoder: str,                  music_base_dir: str | None,
                 crossfade: int,
                 chunk_index: int, total_chunks: int) -> str:
    """Renderiza UN chunk llamando a assemble_video.py como subproceso."""
    # NOTA: acoplado al CLI de assemble_video.py; si ese cambia, actualizar aqui
    cmd = [
        sys.executable, "-m", "assemble_video",
        "--panels", panels_dir,
        "--segments", segments_chunk,
        "--audio", audio_path,
        "--output", output_path,
        "--resolution", resolution,
        "--fps", str(fps),
        "--crf", str(crf),
        "--preset", preset,
        "--encoder", encoder,
        "--crossfade", str(crossfade),
    ]
    if music_base_dir:
        cmd += ["--music-base-dir", music_base_dir]

    print(f"[SLICER] Chunk {chunk_index + 1}/{total_chunks} -> {output_path}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if result.returncode:
        raise RuntimeError(
            f"Chunk {chunk_index + 1} falló: {result.stderr[-2000:]}"
        )
    return output_path


def concat_chunks(chunk_videos: list[str], output_path: str) -> str:
    """Concatena todos los chunks con ffmpeg -c copy.

    Usa concat demuxer (sin recodificación) para unir en segundos.
    """
    if len(chunk_videos) == 1:
        import shutil
        shutil.copy2(chunk_videos[0], output_path)
        return output_path

    concat_dir = Path(output_path).parent
    concat_file = concat_dir / "_chunks_concat.txt"
    lines = [f"file '{Path(p).as_posix()}'\n" for p in sorted(chunk_videos)]
    concat_file.write_text("".join(lines), encoding="utf-8")

    temp_output = concat_dir / "_chunks_joined.mp4"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", str(concat_file),
        "-c", "copy", "-movflags", "+faststart",
        str(temp_output),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode:
        raise RuntimeError(f"Concat de chunks falló: {result.stderr[-1500:]}")

    if Path(output_path).exists():
        Path(output_path).unlink()
    temp_output.rename(output_path)

    try:
        concat_file.unlink()
    except OSError:
        pass

    print(f"[SLICER] {len(chunk_videos)} chunks concatenados en: {output_path}")
    return output_path


def render_parallel(panels_dir: str, segments_path: str, audio_path: str,
                    output_path: str, n_chunks: int = 6, workers: int = 4,
                    resolution: str = "1920x1080", fps: int = 30,
                    crf: int = 23, preset: str = "medium",
                    encoder: str = "libx264",
                    music_base_dir: str | None = None,
                    crossfade: int = 12) -> str:
    """Pipeline completo: split -> render paralelo -> concat.

    Args:
        panels_dir: Directorio con las imágenes de paneles.
        segments_path: Ruta a segments.json.
        audio_path: Ruta al audio maestro continuo (full_audio.mp3).
        output_path: Ruta del video final.
        n_chunks: Número de chunks en que dividir.
        workers: Máximo de procesos paralelos.
        resolution, fps, crf, preset, encoder: Parámetros de FFmpeg.
        music_base_dir: Carpeta con música por categoría.
        burn_subtitles: Incrustar subtítulos.
        subtitle_style: Estilo de subtítulos.
        crossfade: Frames de crossfade.

    Returns:
        Ruta del video final.
    """
    segments = json.loads(Path(segments_path).read_text(encoding="utf-8"))
    segs = segments.get("segments", [])
    if not segs:
        raise ValueError("segments.json está vacío")

    total_dur = sum(_seg_duration(s) for s in segs)
    print(f"[SLICER] {len(segs)} segmentos, {total_dur:.0f}s total, "
          f"{n_chunks} chunks, {workers} workers")

    with tempfile.TemporaryDirectory(prefix="manhwa_slicer_") as tmp:
        chunks = split_segments(segments_path, n_chunks)
        chunk_files = write_chunk_files(chunks, tmp, segments_path)

        chunk_outputs = []
        for i, cf in enumerate(chunk_files):
            out = Path(tmp) / f"chunk_{i:04d}.mp4"
            chunk_outputs.append(str(out))

        futures = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for i in range(len(chunk_files)):
                future = executor.submit(
                    render_chunk, panels_dir, chunk_files[i], audio_path,
                    chunk_outputs[i], resolution, fps, crf, preset, encoder,
                    music_base_dir,
                    crossfade, i, len(chunk_files),
                )
                futures[future] = i

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    future.result()
                    print(f"[SLICER] Chunk {idx + 1}/{len(chunk_files)} completado")
                except Exception as exc:
                    raise RuntimeError(f"Chunk {idx + 1} error: {exc}") from exc

        exit_code = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", chunk_outputs[0]],
            capture_output=True, text=True, timeout=30,
        )
        if exit_code.returncode:
            raise RuntimeError("ffprobe falló al validar chunks")

        concat_chunks(chunk_outputs, output_path)

    final_dur = _get_duration(output_path)
    size_mb = Path(output_path).stat().st_size / 1024 / 1024
    print(f"[SLICER] Video final: {output_path} "
          f"({size_mb:.0f} MB, {final_dur:.0f}s / {final_dur / 60:.1f} min)")
    return output_path


def _get_duration(path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode:
        return 0.0
    return float(result.stdout.strip())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render Slicer: divide, renderiza en paralelo y concatena",
    )
    parser.add_argument("--panels", required=True)
    parser.add_argument("--segments", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunks", type=int, default=6,
                        help="Número de chunks (default 6)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Máximo workers paralelos (default 4)")
    parser.add_argument("--resolution", default="1920x1080")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--crf", type=int, default=23)
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--encoder", default="libx264")
    parser.add_argument("--music-base-dir", default=None)
    parser.add_argument("--crossfade", type=int, default=12)
    args = parser.parse_args()

    for p in [args.panels, args.segments, args.audio]:
        if not Path(p).exists():
            sys.exit(f"ERROR: {p} no encontrado")

    try:
        render_parallel(
            panels_dir=args.panels,
            segments_path=args.segments,
            audio_path=args.audio,
            output_path=args.output,
            n_chunks=args.chunks,
            workers=args.workers,
            resolution=args.resolution,
            fps=args.fps,
            crf=args.crf,
            preset=args.preset,
            encoder=args.encoder,
            music_base_dir=args.music_base_dir,
            crossfade=args.crossfade,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        sys.exit(f"[ERROR] {exc}")


if __name__ == "__main__":
    main()
