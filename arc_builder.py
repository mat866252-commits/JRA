#!/usr/bin/env python3
"""Arc Builder V1: Agrupador de capítulos para videos largos (15-20 min).

Toma los resultados de N capítulos y los concatena en un solo video con
transiciones de pantalla negra + silencio entre capítulos.

YouTube premia los videos de 15-20 min sobre los de 5-8 min.
Este script multiplica el tiempo de retención al ofrecer arcos completos.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def run(command: list[str], timeout: int = 300) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, timeout=timeout)


def validate_file_exists(path: str, desc: str = "Archivo") -> None:
    if not Path(path).is_file():
        raise FileNotFoundError(f"{desc} no encontrado: {path}")


def check_dependencies() -> None:
    missing = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
    if missing:
        raise RuntimeError(f"Falta en PATH: {', '.join(missing)}")



def generate_blank_transition(
    output_path: str, duration_s: float = 0.5,
    width: int = 1920, height: int = 1080, fps: int = 30,
    color: str = "black",
) -> str:
    """Genera clip de transición: pantalla negra + silencio."""
    command = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", f"color=c={color}:s={width}x{height}:d={duration_s}:r={fps}",
        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest", output_path,
    ]
    result = run(command, 60)
    if result.returncode:
        raise RuntimeError(f"Transición: {result.stderr[-500:]}")
    return output_path


def get_video_duration(video_path: str) -> float:
    result = run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", video_path,
    ], 30)
    if result.returncode:
        raise ValueError(f"No se pudo leer duración: {video_path}")
    return float(result.stdout.strip())


def build_arc(
    chapter_dirs: list[Path],
    output_path: str,
    transition_duration: float = 0.5,
    resolution: str = "1920x1080",
    fps: int = 30,
    crf: int = 23,
    preset: str = "medium",
    target_duration: float | None = None,
) -> str:
    """Construye un video de arco desde múltiples capítulos.

    Args:
        chapter_dirs: Lista de directorios de salida de cada capítulo.
        output_path: Ruta del video final.
        transition_duration: Duración de la transición entre capítulos.
        resolution: Resolución del video.
        fps: FPS del video.
        crf: Calidad de video.
        preset: Preset de FFmpeg.

    Returns:
        Ruta del video final.
    """
    check_dependencies()
    w, h = (int(x) for x in resolution.split("x"))

    # Busca video_final.mp4 en cada directorio de capítulo
    chapter_videos: list[Path] = []
    chapter_info: list[dict] = []
    for ch_dir in chapter_dirs:
        candidates = [
            ch_dir / "video_final.mp4",
            ch_dir / "output.mp4",
            ch_dir / "final.mp4",
        ]
        video_path = None
        for c in candidates:
            if c.is_file():
                video_path = c
                break
        if video_path is None:
            mp4_files = sorted(ch_dir.glob("*.mp4"))
            if mp4_files:
                video_path = mp4_files[-1]
        if video_path and video_path.is_file():
            dur = get_video_duration(str(video_path))
            chapter_videos.append(video_path)
            chapter_info.append({
                "dir": str(ch_dir),
                "file": str(video_path),
                "duration": dur,
            })
        else:
            print(f"[AVISO] No se encontró video en: {ch_dir}")

    if len(chapter_videos) < 2:
        if len(chapter_videos) == 1:
            shutil.copy2(str(chapter_videos[0]), output_path)
            print(f"[ARCO] Solo 1 capítulo, copiando directamente: {output_path}")
            return output_path
        raise RuntimeError(f"Se necesitan al menos 2 capítulos (encontrados: {len(chapter_videos)})")

    total_dur = sum(ci["duration"] for ci in chapter_info)
    trans_total = transition_duration * (len(chapter_videos) - 1)
    estimated = total_dur + trans_total
    print(f"[ARCO] {len(chapter_videos)} capítulos, {estimated:.0f}s (~{estimated/60:.1f} min)")

    with tempfile.TemporaryDirectory(prefix="manhwa_arc_") as tmp:
        concat_parts: list[Path] = []

        for i, video_path in enumerate(chapter_videos):
            concat_parts.append(video_path)

            if i < len(chapter_videos) - 1:
                transition_clip = Path(tmp) / f"transition_{i:04d}.mp4"
                generate_blank_transition(
                    str(transition_clip), transition_duration,
                    width=w, height=h, fps=fps,
                )
                concat_parts.append(transition_clip)

        # Concatena usando concat demuxer (más rápido, sin re-codificar)
        concat_file = Path(tmp) / "concat.txt"
        lines = []
        for part in concat_parts:
            lines.append(f"file '{part.as_posix()}'\n")
        concat_file.write_text("".join(lines), encoding="utf-8")

        temp_output = Path(tmp) / "arc_joined.mp4"
        command = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", str(concat_file),
            "-c", "copy", "-movflags", "+faststart",
            str(temp_output),
        ]
        result = run(command, 600)
        if result.returncode:
            print(f"[AVISO] Concat directo falló, usando re-codificación: {result.stderr[-300:]}")
            # Fallback: re-codificar todo con filter_complex concat
            inputs = []
            for part in concat_parts:
                inputs.extend(["-i", str(part)])
            filter_parts = []
            for i in range(len(concat_parts)):
                filter_parts.append(f"[{i}:v][{i}:a]")
            filter_str = "".join(filter_parts) + f"concat=n={len(concat_parts)}:v=1:a=1[v][a]"
            command = [
                "ffmpeg", "-y", "-loglevel", "error",
                *inputs,
                "-filter_complex", filter_str,
                "-map", "[v]", "-map", "[a]",
                "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
                "-c:a", "aac", "-b:a", "192k",
                "-movflags", "+faststart",
                str(temp_output),
            ]
            result = run(command, 900)
            if result.returncode:
                raise RuntimeError(f"Arc: {result.stderr[-1000:]}")

        # Aplica normalización de audio final
        final_output = output_path
        command = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(temp_output),
            "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            final_output,
        ]
        result = run(command, 600)
        if result.returncode:
            print(f"[AVISO] Normalización falló, usando sin normalizar")
            shutil.copy2(str(temp_output), final_output)

    # Reporte
    final_dur = get_video_duration(final_output)
    print(f"[ARCO] Video final: {final_output}")
    print(f"[ARCO] Duración: {final_dur:.0f}s ({final_dur/60:.1f} min)")
    print(f"[ARCO] Tamaño: {Path(final_output).stat().st_size / 1024 / 1024:.0f} MB")

    # Guarda metadatos
    meta_path = Path(final_output).with_suffix(".arc_meta.json")
    meta_path.write_text(
        json.dumps({
            "type": "arc",
            "chapters": chapter_info,
            "total_duration": round(final_dur, 2),
            "estimated_duration": round(estimated, 2),
            "transition_duration": transition_duration,
            "resolution": resolution,
            "fps": fps,
            "target_duration": target_duration,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[ARCO] Metadatos: {meta_path}")

    return final_output


def build_arc_from_videos(
    video_paths: list[str],
    output_path: str,
    transition_duration: float = 0.5,
    resolution: str = "1920x1080",
    fps: int = 30,
    crf: int = 23,
    preset: str = "medium",
    source_names: list[str] | None = None,
) -> str:
    """Construye un arco desde una lista directa de archivos de video.

    Similar a build_arc pero acepta rutas de video ya renderizadas
    en lugar de directorios de capítulos.

    Args:
        video_paths: Lista de rutas a archivos de video.
        output_path: Ruta del video final del arco.
        transition_duration: Duración de la transición entre videos.
        resolution: Resolución del video.
        fps: FPS del video.
        crf: Calidad de video.
        preset: Preset de FFmpeg.
        source_names: Nombres descriptivos para cada video (opcional).

    Returns:
        Ruta del video final.
    """
    check_dependencies()
    w, h = (int(x) for x in resolution.split("x"))

    chapter_videos: list[Path] = []
    chapter_info: list[dict] = []
    for i, vp in enumerate(video_paths):
        p = Path(vp)
        if not p.is_file():
            print(f"[AVISO] Video no encontrado: {vp}")
            continue
        dur = get_video_duration(str(p))
        chapter_videos.append(p)
        chapter_info.append({
            "file": str(p),
            "duration": dur,
            "name": source_names[i] if source_names and i < len(source_names) else p.stem,
        })

    if len(chapter_videos) < 2:
        if len(chapter_videos) == 1:
            shutil.copy2(str(chapter_videos[0]), output_path)
            print(f"[ARCO] Solo 1 video, copiando directamente: {output_path}")
            return output_path
        raise RuntimeError(f"Se necesitan al menos 2 videos (encontrados: {len(chapter_videos)})")

    total_dur = sum(ci["duration"] for ci in chapter_info)
    trans_total = transition_duration * (len(chapter_videos) - 1)
    estimated = total_dur + trans_total
    print(f"[ARCO] {len(chapter_videos)} videos, {estimated:.0f}s (~{estimated/60:.1f} min)")

    with tempfile.TemporaryDirectory(prefix="manhwa_arc_") as tmp:
        concat_parts: list[Path] = []

        for i, video_path in enumerate(chapter_videos):
            concat_parts.append(video_path)
            if i < len(chapter_videos) - 1:
                transition_clip = Path(tmp) / f"transition_{i:04d}.mp4"
                generate_blank_transition(
                    str(transition_clip), transition_duration,
                    width=w, height=h, fps=fps,
                )
                concat_parts.append(transition_clip)

        concat_file = Path(tmp) / "concat.txt"
        lines = [f"file '{part.as_posix()}'\n" for part in concat_parts]
        concat_file.write_text("".join(lines), encoding="utf-8")

        temp_output = Path(tmp) / "arc_joined.mp4"
        command = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", str(concat_file),
            "-c", "copy", "-movflags", "+faststart",
            str(temp_output),
        ]
        result = run(command, 600)
        if result.returncode:
            print(f"[AVISO] Concat directo falló, usando re-codificación: {result.stderr[-300:]}")
            inputs = []
            for part in concat_parts:
                inputs.extend(["-i", str(part)])
            filter_parts = [f"[{i}:v][{i}:a]" for i in range(len(concat_parts))]
            filter_str = "".join(filter_parts) + f"concat=n={len(concat_parts)}:v=1:a=1[v][a]"
            command = [
                "ffmpeg", "-y", "-loglevel", "error",
                *inputs,
                "-filter_complex", filter_str,
                "-map", "[v]", "-map", "[a]",
                "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
                "-c:a", "aac", "-b:a", "192k",
                "-movflags", "+faststart",
                str(temp_output),
            ]
            result = run(command, 900)
            if result.returncode:
                raise RuntimeError(f"Arc: {result.stderr[-1000:]}")

        final_output = output_path
        command = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(temp_output),
            "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            final_output,
        ]
        result = run(command, 600)
        if result.returncode:
            print(f"[AVISO] Normalización falló, usando sin normalizar")
            shutil.copy2(str(temp_output), final_output)

    final_dur = get_video_duration(final_output)
    print(f"[ARCO] Video final: {final_output}")
    print(f"[ARCO] Duración: {final_dur:.0f}s ({final_dur/60:.1f} min)")
    print(f"[ARCO] Tamaño: {Path(final_output).stat().st_size / 1024 / 1024:.0f} MB")

    meta_path = Path(final_output).with_suffix(".arc_meta.json")
    meta_path.write_text(
        json.dumps({
            "type": "arc_from_videos",
            "videos": chapter_info,
            "total_duration": round(final_dur, 2),
            "estimated_duration": round(estimated, 2),
            "transition_duration": transition_duration,
            "resolution": resolution,
            "fps": fps,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return final_output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Arc Builder: agrupa N capítulos en un video largo (15-20 min)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--chapters", nargs="+", default=None,
                        help="Directorios de salida de cada capítulo (ordenados)")
    parser.add_argument("--videos", nargs="+", default=None,
                        help="Rutas directas a archivos de video (alternativa a --chapters)")
    parser.add_argument("--output", required=True,
                        help="Ruta del video final del arco")
    parser.add_argument("--transition-duration", type=float, default=1.5,
                        help="Duración en segundos de la transición entre capítulos (default 1.5)")
    parser.add_argument("--resolution", default="1920x1080",
                        help="Resolución del video (default 1920x1080)")
    parser.add_argument("--fps", type=int, default=30, choices=[24, 25, 30, 60])
    parser.add_argument("--crf", type=int, default=23)
    parser.add_argument("--preset", default="medium",
                        help="Preset de FFmpeg (ultrafast, fast, medium, slow)")
    parser.add_argument("--target-duration", type=float, default=None,
                        help="Duración objetivo en segundos (opcional, solo metadato)")

    args = parser.parse_args()

    if args.videos:
        try:
            build_arc_from_videos(
                video_paths=args.videos,
                output_path=args.output,
                transition_duration=args.transition_duration,
                resolution=args.resolution,
                fps=args.fps,
                crf=args.crf,
                preset=args.preset,
            )
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            sys.exit(f"[ERROR] {exc}")
        return

    if not args.chapters:
        sys.exit("ERROR: Debes especificar --chapters o --videos")

    chapter_dirs = [Path(d) for d in args.chapters]
    for d in chapter_dirs:
        if not d.is_dir():
            sys.exit(f"ERROR: Directorio no encontrado: {d}")

    try:
        build_arc(
            chapter_dirs=chapter_dirs,
            output_path=args.output,
            transition_duration=args.transition_duration,
            resolution=args.resolution,
            fps=args.fps,
            crf=args.crf,
            preset=args.preset,
            target_duration=args.target_duration,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        sys.exit(f"[ERROR] {exc}")


if __name__ == "__main__":
    main()
