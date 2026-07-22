#!/usr/bin/env python3
"""Monta un vídeo de resumen desde viñetas, guion y audios numerados.

La regla principal es no ocultar errores de alineación: con ``--script`` los
números de escena deben existir exactamente una vez en las tres fuentes.

v3: Añade transiciones suaves entre escenas (fundido, disolvencia, corte limpio)
    según el tipo narrativo y un informe de calidad pre-exportación.
    Soporta grupos de escenas (scene_groups.json) para mostrar múltiples viñetas
    en un mismo clip con micro-transiciones internas.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from config import VideoConfig, parse_resolution
from scene_files import indexed_files

logger = logging.getLogger("assemble_video")


def validate_file_exists(file_path: str, description: str = "Archivo"):
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"ERROR CRITICO: {description} no encontrado en: {path}\nVerifica que las fases anteriores se ejecutaron correctamente.")
    if not path.is_file():
        raise IsADirectoryError(f"ERROR CRITICO: Se esperaba un archivo, pero se encontro un directorio en: {path}")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a"}
MODES = ("zoom", "pan", "static", "panzoom")
TRANSITIONS = ("cut", "fade", "dissolve", "fade_black")
AUDIO_NORMALIZATION_FILTER = "loudnorm=I=-16:TP=-1.5:LRA=11"


def run(command: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, timeout=timeout)


def check_dependencies() -> None:
    missing = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
    if missing:
        raise RuntimeError(f"Falta en PATH: {', '.join(missing)}")


def load_panel_metadata(panels_dir: str, manifest_path: str | None) -> dict[str, dict]:
    path = Path(manifest_path) if manifest_path else Path(panels_dir) / "panels.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {item["file"]: item for item in data.get("panels", []) if item.get("file")}
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[AVISO] No se pudo leer el manifiesto de paneles {path}: {exc}", file=sys.stderr)
        return {}


def load_scene_groups(path: str) -> dict[int, list[int]] | None:
    """Carga grupos narrativos desde scene_groups.json.
    
    Acepta un directorio (busca scene_groups.json dentro) o la ruta completa al archivo JSON.
    """
    p = Path(path)
    if p.is_dir():
        p = p / "scene_groups.json"
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        mapping: dict[int, list[int]] = {}
        for group in data.get("groups", []):
            for scene_num in group.get("scenes", []):
                mapping[scene_num] = group["scenes"]
        return mapping
    except (OSError, json.JSONDecodeError):
        return None


def collect_scenes(panels_dir: str, audio_dir: str, script_path: str | None, manifest_path: str | None,
                   groups: dict[int, list[int]] | None = None) -> list[dict]:
    panels = indexed_files(panels_dir, IMAGE_EXTENSIONS, "viñetas")
    audios = indexed_files(audio_dir, AUDIO_EXTENSIONS, "audios")
    script = dict(parse_script(script_path)) if script_path else {}
    metadata = load_panel_metadata(panels_dir, manifest_path)
    expected = set(script) if script_path else set(panels) & set(audios)
    groups = groups or {}
    # Un número de escena identifica hoy un GRUPO narrativo (scene_groups.json), que puede
    # incluir varias viñetas fusionadas bajo el mismo bloque de audio. Un panel que fue
    # absorbido en el grupo de otra escena representativa NO es un "huérfano sin guion":
    # se cubre a través del grupo de su representante. Solo cuenta como error real un panel
    # que no aparece ni como escena propia ni dentro de ningún grupo esperado.
    covered_panels: set[int] = set()
    for number in expected:
        covered_panels |= set(groups.get(number, [number]))
    missing_panels = sorted({p for number in expected for p in groups.get(number, [number]) if p not in panels})
    missing_audios = sorted(expected - set(audios))
    extras_panels = sorted(set(panels) - covered_panels) if script_path else sorted(set(panels) - expected)
    extras_audios = sorted(set(audios) - expected)
    if missing_panels or missing_audios or (script_path and (extras_panels or extras_audios)):
        parts = []
        for label, values in (("viñetas ausentes", missing_panels), ("audios ausentes", missing_audios),
                              ("viñetas sin guion ni grupo", extras_panels), ("audios sin guion", extras_audios)):
            if values:
                parts.append(f"{label}: {', '.join(f'{value:04d}' for value in values)}")
        raise ValueError(" | ".join(parts))
    if not script_path:
        print("[AVISO] Sin --script: se monta por número y el matcher queda desactivado.")
    scenes = []
    for number in sorted(expected):
        members = groups.get(number, [number])
        member_paths = [str(panels[member]) for member in members if member in panels]
        if not member_paths:
            member_paths = [str(panels[number])]
        scenes.append({
            "number": number, "image_path": member_paths[0], "image_paths": member_paths,
            "audio_path": str(audios[number]), "text": script.get(number),
            "vision_description": metadata.get(panels[number].name, {}).get("vision_description"),
        })
    return scenes


def resolve_assignments(scenes: list[dict], matcher, window: int, min_similarity: float,
                        min_margin: float, continuity_gap: int) -> tuple[list[dict], list[dict]]:
    """Reasigna solo ante evidencia visual fuerte dentro de una ventana local."""
    decisions: list[dict] = []
    if matcher is None:
        return scenes, decisions
    paths = [scene["image_path"] for scene in scenes]
    resolved = []
    last_index: int | None = None
    for index, scene in enumerate(scenes):
        start, end = max(0, index - window), min(len(scenes), index + window + 1)
        candidate_indexes = list(range(start, end))
        scores = matcher.similarities(scene["text"], [paths[item] for item in candidate_indexes])
        score_by_index = dict(zip(candidate_indexes, scores))
        default_score = score_by_index[index]
        best_index = max(candidate_indexes, key=score_by_index.get)
        best_score = score_by_index[best_index]
        margin = best_score - default_score
        continuity_ok = last_index is None or abs(best_index - last_index) <= continuity_gap
        accepted = (best_index != index and best_score >= min_similarity and margin >= min_margin and continuity_ok)
        selected_index = best_index if accepted else index
        selected = dict(scene)
        selected["image_path"] = paths[selected_index]
        if accepted:
            # El matcher encontró una viñeta mejor para esta narración que las del grupo
            # original: el clip se renderiza con esa única viñeta corregida, no con el
            # grupo de partida (si no, la corrección del matcher quedaría invisible en
            # el vídeo porque build_clip prioriza 'image_paths' sobre 'image_path').
            selected["image_paths"] = [paths[selected_index]]
        selected["vision_description"] = scenes[selected_index].get("vision_description")
        selected["match"] = {"default_score": round(default_score, 5), "best_score": round(best_score, 5),
                             "margin": round(margin, 5), "accepted": accepted, "best_panel": scenes[best_index]["number"],
                             "continuity_ok": continuity_ok}
        resolved.append(selected)
        last_index = selected_index
        if accepted:
            decisions.append({
                "scene": scene["number"], "default_panel": scenes[index]["number"],
                "selected_panel": scenes[selected_index]["number"], "default_score": round(default_score, 5),
                "selected_score": round(best_score, 5), "margin": round(margin, 5),
                "window": [scenes[start]["number"], scenes[end - 1]["number"]],
                "vision_description": selected.get("vision_description"),
            })
    return resolved, decisions


def duration(path: str) -> float:
    result = run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", path], 30)
    if result.returncode:
        raise ValueError(result.stderr.strip() or f"No se pudo leer {path}")
    value = float(result.stdout.strip())
    if value <= 0.2:
        raise ValueError(f"Audio vacío o demasiado corto: {path}")
    return value


def scene_motion(aspect_ratio: float, frames: int, config: VideoConfig) -> str:
    """Smart Webtoon Scroll / Gentle Zoom según la proporción del panel.

    Si aspect_ratio > 1.2 (panel vertical):
        Escala al ancho, luego zoompan con z=1 desplazándose verticalmente
        de arriba (y=0) a abajo (y=ih-oh) simulando el scroll de lectura.
        Centrado horizontalmente.
    Si aspect_ratio <= 1.2 (panel horizontal/cuadrado):
        Escala grande, zoom suave 1.0→1.05 centrado.
    """
    last = max(1, frames - 1)

    if aspect_ratio > 1.2:
        img_h = config.width * aspect_ratio
        max_offset = max(0, img_h - config.height)
        # Limita scroll para evitar OOM en paneles extremadamente largos
        max_offset = min(max_offset, config.height * 3)
        step = max_offset / last if last else 0
        return (
            f"scale={config.width}:-2,"
            f"zoompan=z='1':"
            f"x='(iw-ow)/2':"
            f"y='min({step}*on,{max_offset})':"
            f"d={frames}:s={config.width}x{config.height}:fps={config.fps}"
        )
    else:
        end_z = 1.05
        return (
            f"scale={int(config.width * 1.3)}:-2,"
            f"zoompan=z='1+({end_z}-1)*pow(on/{last},2)':"
            f"x='iw/2-iw/zoom/2':y='ih/2-ih/zoom/2':"
            f"d={frames}:s={config.width}x{config.height}:fps={config.fps}"
        )


def _get_aspect_ratio(image_path: str) -> float:
    """Obtiene la proporción (alto/ancho) de una imagen sin cargarla completamente."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height",
         "-of", "default=noprint_wrappers=1:nokey=1", image_path],
        capture_output=True, text=True, timeout=15,
    )
    if result.returncode:
        return 1.0
    lines = result.stdout.strip().split()
    if len(lines) >= 2:
        try:
            w = float(lines[0])
            h = float(lines[1])
            return h / w if w > 0 else 1.0
        except ValueError:
            pass
    return 1.0


def _render_hook_clip(
    panel_path: str, output_path: str, duration_s: float,
    config: VideoConfig, preset: str, crf: int, encoder: str,
) -> None:
    """Renderiza clip de apertura para hook con fondo espejado + Smart Scroll.
    Incluye silent audio track para compatibilidad con concat demuxer.
    """
    validate_file_exists(panel_path, "Hook panel")
    frames = max(1, round(duration_s * config.fps))
    aspect = _get_aspect_ratio(panel_path)
    motion = scene_motion(aspect, frames, config)
    fade_start = max(0.0, duration_s - 0.4)
    filter_complex = (
        f"[0:v]split[b][f];"
        f"[b]scale={config.width}:{config.height},"
        f"boxblur=30:10,eq=brightness=-0.1[bg];"
        f"[f]{motion},fade=t=out:st={fade_start:.3f}:d=0.4[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2,format=yuv420p[v]"
    )
    command = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-loop", "1", "-i", panel_path,
        "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-filter_complex", filter_complex, "-map", "[v]", "-map", "1:a",
        "-ar", "44100", "-ac", "2",
        "-c:v", encoder, "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k", "-t", f"{duration_s:.3f}", "-shortest", output_path,
    ]
    result = run(command, 600)
    if result.returncode:
        raise RuntimeError(f"Hook: {result.stderr[-1500:]}")


CATEGORY_MUSIC_MAP = {
    "accion": "action",
    "accion_grupal": "action",
    "conversacion": "sad",
    "primer_plano": "tension",
    "fondo": "tension",
    "fondo_vacio": "tension",
    "fondo/texto": "tension",
    "texto_puro": "tension",
    "paisaje": "tension",
}


def _select_music_for_category(category: str, music_base: str) -> str | None:
    """Elige una pista de música aleatoria según la categoría narrativa."""
    subdir_name = CATEGORY_MUSIC_MAP.get(category, "tension")
    music_dir = Path(music_base) / subdir_name
    if not music_dir.is_dir():
        return None
    songs = list(music_dir.glob("*.mp3")) + list(music_dir.glob("*.wav")) + list(music_dir.glob("*.m4a"))
    if not songs:
        return None
    return str(random.choice(songs))


def _build_composite_music(
    segments: list[dict],
    panel_meta: dict[int, dict],
    music_base: str,
    output_path: str,
    sample_rate: int = 44100,
) -> str | None:
    """Construye una pista de música compuesta seleccionando pista por categoría de cada segmento.

    Para cada segmento, elige música según su categoría, la recorta a la duración del segmento
    y concatena todo en un solo archivo de audio.
    """
    import subprocess as _sp

    music_pieces = []
    for i, seg in enumerate(segments):
        scene_id = seg.get("panel_scene", 0)
        pmeta = panel_meta.get(scene_id, {})
        category = pmeta.get("category", "")
        seg_duration = seg.get("duration_sec", seg.get("end", 10) - seg.get("start", 0))
        if seg_duration <= 0.3:
            continue

        music_file = _select_music_for_category(category, music_base)
        if not music_file:
            continue

        piece = Path(output_path).parent / f"_music_piece_{i:04d}.mp3"
        # Extrae la duración exacta del segmento de la pista de música
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", music_file,
            "-af", f"atrim=duration={seg_duration:.3f},volume=0.15",
            "-c:a", "libmp3lame", "-q:a", "2",
            "-ar", str(sample_rate),
            str(piece),
        ]
        result = _sp.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0 and piece.exists() and piece.stat().st_size > 256:
            music_pieces.append(str(piece))

    if not music_pieces:
        return None

    if len(music_pieces) == 1:
        return music_pieces[0]

    # Concatena todas las piezas
    concat_file = Path(output_path).with_suffix(".music_concat.txt")
    concat_file.write_text(
        "".join(f"file '{Path(p).as_posix()}'\n" for p in music_pieces), encoding="utf-8"
    )
    composite = output_path
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", str(concat_file),
        "-c", "copy",
        str(composite),
    ]
    result = _sp.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode:
        return None

    # Limpia piezas temporales
    for p in music_pieces:
        try:
            Path(p).unlink()
        except OSError:
            pass

    return str(composite)


def mix_background_music(input_video: str, music_path: str, output_video: str,
                          config: VideoConfig, music_volume_db: float = -22.0,
                          duck_ratio: float = 8.0) -> None:
    """V4: Mezcla música de fondo con sidechain ducking profesional.

    La música baja automáticamente cuando el narrador habla y sube en silencios.
    Filtro exacto: volume=0.15, sidechaincompress=threshold=0.01:ratio=20:attack=15:release=200
    """
    validate_file_exists(music_path, "Pista de música")
    filter_complex = (
        "[1:a]volume=0.10,"
        "sidechaincompress=threshold=0.01:ratio=10:attack=10:release=300[bg];"
        "[0:a][bg]amix=inputs=2:duration=first:dropout_transition=0:weights=1 1[aout]"
    )
    command = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", input_video, "-i", music_path,
        "-filter_complex", filter_complex,
        "-map", "0:v", "-map", "[aout]",
        "-c:v", "copy", "-c:a", config.audio_codec, "-b:a", "192k",
        "-shortest", output_video,
    ]
    result = run(command, 600)
    if result.returncode:
        raise RuntimeError(result.stderr[-1500:])


def build_clip(scene: dict, destination: str,
               preset: str, crf: int, encoder: str, config: VideoConfig) -> float:
    """Renderiza el clip de una escena con fondo espejado y Smart Webtoon Scroll.

    La duracion del clip es EXACTAMENTE la duracion del audio.
    Soporta múltiples image_paths para grupos de paneles.
    """
    image_paths = scene.get("image_paths") or [scene["image_path"]]
    for path in image_paths:
        validate_file_exists(path, "Imagen de la escena")
    validate_file_exists(scene["audio_path"], "Audio de la escena")
    seconds = duration(scene["audio_path"])

    panel_count = len(image_paths)
    sub_seconds = seconds / panel_count
    inputs: list[str] = []
    video_labels: list[str] = []
    filter_parts: list[str] = []
    for index, image_path in enumerate(image_paths):
        inputs.extend(["-loop", "1", "-i", image_path])
        sub_frames = max(1, round(sub_seconds * config.fps))
        aspect = _get_aspect_ratio(image_path)
        motion = scene_motion(aspect, sub_frames, config)
        label = f"v{index}"
        bi = f"bg{index}"
        fi = f"fg{index}"
        filter_parts.append(
            f"[{index}:v]split[{bi}_src][{fi}_src];"
            f"[{bi}_src]scale={config.width}:{config.height},"
            f"boxblur=30:10,eq=brightness=-0.1[{bi}];"
            f"[{fi}_src]{motion}[{fi}];"
            f"[{bi}][{fi}]overlay=(W-w)/2:(H-h)/2,"
            f"trim=duration={sub_seconds:.3f},setpts=PTS-STARTPTS,fps={config.fps}[{label}]"
        )
        video_labels.append(f"[{label}]")

    audio_input_index = panel_count
    if panel_count > 1:
        filter_parts.append(f"{''.join(video_labels)}concat=n={panel_count}:v=1:a=0[vconcat]")
        base_v = "[vconcat]"
    else:
        base_v = video_labels[0]

    filter_complex = ";".join(filter_parts)

    command = ["ffmpeg", "-y", "-loglevel", "error", *inputs, "-i", scene["audio_path"],
               "-filter_complex", filter_complex, "-map", base_v, "-map", f"{audio_input_index}:a",
               "-ar", "44100", "-ac", "2",
               "-c:v", encoder, "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
               "-c:a", "aac", "-b:a", "160k",
               "-shortest", str(destination)]
    result = run(command, 240)
    if result.returncode:
        raise RuntimeError(result.stderr[-1500:])
    return seconds








def apply_audio_normalization(input_video: str, output_video: str, config: VideoConfig) -> None:
    """Aplica loudnorm UNA SOLA VEZ al video completo.
    Edge TTS ya produce audio limpio, solo necesitamos normalizar
    el volumen a -16 LUFS (estandar YouTube)."""
    command = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", input_video,
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-c:v", "copy",
        "-c:a", config.audio_codec,
        "-b:a", "192k",
        output_video
    ]
    result = run(command, 600)
    if result.returncode:
        raise RuntimeError(result.stderr[-1500:])


def normalize_final_audio(input_video_path: str, output_video_path: str, config: VideoConfig) -> None:
    """Aplica loudnorm UNA SOLA VEZ al video final completo."""
    print(f"Aplicando normalizacion de audio final a: {output_video_path}")
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", input_video_path,
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-c:v", "copy",
        "-c:a", config.audio_codec,
        "-b:a", "192k",
        output_video_path
    ]
    try:
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True, timeout=600)
    except subprocess.CalledProcessError as e:
        print(f"Advertencia: Fallo la normalizacion final. El video se guardara sin normalizar. Error: {e.stderr}")
        import shutil
        shutil.copy(input_video_path, output_video_path)


def assemble_from_segments(
    panels_dir: str,
    audio_path: str,
    segments_path: str,
    output_path: str,
    resolution: str = "1920x1080",
    fps: int = 30,
    crf: int = 23,
    preset: str = "medium",
    encoder: str = "libx264",
    music: str | None = None,
    music_base_dir: str | None = None,
    music_volume: float = -22.0,
    duck_ratio: float = 8.0,
    crossfade_frames: int = 12,
) -> Path:
    """Ensambla video final a partir de segmentos + audio continuo.

    Flujo nuevo (segment-based):
      1. Lee segments.json con los tiempos y paneles asignados
      2. Lee full_audio.mp3 (un solo archivo continuo)
      3. Para cada segmento, crea un clip con Ken Burns
      4. Aplica crossfade entre clips
      5. Concatena todos los clips SIN pistas de audio
      6. Mezcla con el audio maestro continuo
      7. Añade música con ducking
      8. Normaliza loudness final
    """
    from config import VideoConfig, parse_resolution

    w, h = parse_resolution(resolution)
    config = VideoConfig(width=w, height=h, fps=fps)
    check_dependencies()

    segments_data = json.loads(Path(segments_path).read_text(encoding="utf-8"))
    segments = segments_data.get("segments", [])
    if not segments:
        raise ValueError("No hay segmentos en segments.json")

    full_audio = Path(audio_path)
    if not full_audio.is_file():
        raise FileNotFoundError(f"Audio no encontrado: {audio_path}")

    total_duration = segments_data.get("total_duration_sec") or segments_data.get("total_duration", 0)
    if total_duration <= 0 and segments:
        total_duration = sum(s.get("duration_sec", 4.5) for s in segments)
    if total_duration <= 0:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(full_audio)],
            capture_output=True, text=True, timeout=30,
        )
        total_duration = float(result.stdout.strip()) if not result.returncode else 60.0

    print(f"[MONTAJE] {len(segments)} segmentos, audio: {total_duration:.1f}s")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Carga metadatos de paneles para categorías
    panel_manifest = Path(panels_dir) / "panels.json"
    panel_meta = {}
    if panel_manifest.is_file():
        try:
            for p in json.loads(panel_manifest.read_text(encoding="utf-8")).get("panels", []):
                panel_meta[p["scene"]] = p
        except Exception as exc:
            logger.warning("Error cargando panel_meta: %s", exc)

    # Valida que todos los panel_scene de los segmentos existan en panel_meta
    missing_ids = [s.get("panel_scene", 0) for s in segments if s.get("panel_scene", 0) not in panel_meta and panel_meta]
    if missing_ids:
        logger.warning(f"[MONTAJE] {len(missing_ids)} scene_ids sin metadata en panels.json: {missing_ids[:15]}")
        panel_meta.clear()

    with tempfile.TemporaryDirectory(prefix="manhwa_segments_") as temp:
        # V4: Hook de apertura (visual + música, SIN narrador)
        hook_data = segments_data.get("hook")
        hook_clip_path = None
        if hook_data and hook_data.get("panel_id") and hook_data.get("text"):
            hook_panel_id = hook_data["panel_id"]
            hook_file = hook_data.get("panel_file", f"escena_{hook_panel_id:04d}.png")
            hook_panel_path = Path(panels_dir) / hook_file
            if hook_panel_path.is_file():
                hook_duration = min(hook_data.get("duration", 10), 10.0)
                hook_clip_out = Path(temp) / "_hook.mp4"
                _render_hook_clip(str(hook_panel_path), str(hook_clip_out), hook_duration,
                                   config, preset, crf, encoder)
                hook_clip_path = hook_clip_out
                print(f"[HOOK] Clip de apertura renderizado ({hook_duration:.1f}s)")

        clip_paths = []
        if hook_clip_path:
            clip_paths.append(hook_clip_path)
        for i, seg in enumerate(segments):
            panel_file = seg.get("panel_file", f"escena_{seg.get('panel_scene', 0):04d}.png")
            panel_path = Path(panels_dir) / panel_file
            if not panel_path.is_file():
                panel_path = Path(panels_dir) / f"escena_{seg.get('panel_scene', 0):04d}.png"
                if not panel_path.is_file():
                    print(f"[AVISO] Panel no encontrado: {panel_file}, saltando segmento {i}")
                    continue

            # V4: Duración por palabras por segundo (2.5 wps estándar)
            text = seg.get("text", "")
            word_count = len(text.split()) if text else 1
            seg_duration = (word_count / 2.5) + 0.5
            # Escala para que calce con el audio total
            seg_duration = seg_duration * (total_duration / max(sum(
                (len(s.get("text", "").split()) / 2.5) + 0.5 for s in segments if s.get("text")
            ), total_duration))

            if seg_duration <= 0.3:
                seg_duration = 0.3

            # Obtiene categoría y estrellas para Ken Burns dirigido
            scene_id = seg.get("panel_scene", 0)
            pmeta = panel_meta.get(scene_id, {})
            category = pmeta.get("category", "")
            stars = pmeta.get("stars", 3)

            clip_out = Path(temp) / f"seg_{i:04d}.mp4"
            _render_segment_clip(str(panel_path), str(clip_out), seg_duration, config,
                                 preset, crf, encoder)
            clip_paths.append(clip_out)

        if not clip_paths:
            raise RuntimeError("No se pudo renderizar ningún segmento")

        clips_joined = Path(temp) / "clips_joined.mp4"
        _concat_clips(clip_paths, str(clips_joined), crossfade_frames, config.fps)

        temp_final = Path(temp) / "with_audio.mp4"
        _merge_audio_and_video(str(clips_joined), str(full_audio), str(temp_final),
                                config, encoder, preset, crf)

        current = temp_final
        # V4: Música por intensidad (composite por categoría) o single track legacy
        use_composite = music_base_dir and Path(music_base_dir).is_dir()
        if use_composite:
            comp_music = Path(temp) / "composite_music.mp3"
            music_track = _build_composite_music(
                segments, panel_meta, music_base_dir, str(comp_music)
            )
            if music_track:
                mixed = Path(temp) / "with_music.mp4"
                try:
                    mix_background_music(str(current), music_track, str(mixed), config,
                                          music_volume, duck_ratio)
                    current = mixed
                    print(f"[MONTAJE] Música compuesta por categoría + ducking profesional")
                except RuntimeError as exc:
                    print(f"[AVISO] No se pudo mezclar música compuesta: {exc}")
        elif music:
            mixed = Path(temp) / "with_music.mp4"
            try:
                mix_background_music(str(current), music, str(mixed), config,
                                      music_volume, duck_ratio)
                current = mixed
                print(f"[MONTAJE] Música mezclada con ducking ({music_volume} dB)")
            except RuntimeError as exc:
                print(f"[AVISO] No se pudo mezclar música: {exc}")

        normalize_final_audio(str(current), str(output), config)

        pass

    final_dur = duration(str(output))
    print(f"[MONTAJE] Video listo: {output} ({output.stat().st_size / 1024 / 1024:.1f} MB, {final_dur:.1f}s)")
    return output


def _render_segment_clip(
    panel_path: str, output_path: str, duration_s: float,
    config: VideoConfig, preset: str, crf: int, encoder: str,
) -> None:
    """Renderiza clip con fondo espejado desenfocado + Smart Webtoon Scroll.

    Capa de fondo: panel escalado a 1920x1080 (sin proporción) + boxblur + oscurecido.
    Capa frontal: panel con scroll vertical (si es alto) o zoom suave (si es ancho).
    """
    validate_file_exists(panel_path, "Panel")
    frames = max(1, round(duration_s * config.fps))
    aspect = _get_aspect_ratio(panel_path)
    motion = scene_motion(aspect, frames, config)

    filter_complex = (
        f"[0:v]split[b][f];"
        f"[b]scale={config.width}:{config.height},"
        f"boxblur=30:10,eq=brightness=-0.1[bg];"
        f"[f]{motion}[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2,format=yuv420p[v]"
    )

    command = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-loop", "1", "-i", panel_path,
        "-filter_complex", filter_complex, "-map", "[v]",
        "-c:v", encoder, "-preset", preset, "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-an", "-t", f"{duration_s:.3f}", "-shortest", output_path,
    ]
    result = run(command, 120)
    if result.returncode:
        raise RuntimeError(f"FFmpeg: {result.stderr[-1000:]}")


def _get_clip_duration(file_path: str) -> float:
    """Obtiene la duración de un clip de video usando ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", file_path],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode:
        raise RuntimeError(f"ffprobe falló en {file_path}: {result.stderr}")
    return float(result.stdout.strip())


def _concat_clips(clip_paths: list[Path], output_path: str,
                   crossfade_frames: int = 12, fps: int = 30) -> None:
    """Concatena clips con xfade fade (0.4s) entre ellos.

    Usa el filtro xfade=transition=fade:duration=0.4s.
    Si falla, hace concat simple con -c copy.
    """
    if len(clip_paths) == 1:
        shutil.copy2(str(clip_paths[0]), output_path)
        return

    inputs = []
    for clip_path in clip_paths:
        inputs.extend(["-i", str(clip_path)])

    if crossfade_frames > 0 and len(clip_paths) >= 2:
        duration_s = max(0.1, crossfade_frames / max(fps, 1))
        clip_durs = [_get_clip_duration(str(p)) for p in clip_paths]

        filter_parts = []
        cumulative = clip_durs[0]
        prev_label = "0:v"

        for i in range(1, len(clip_paths)):
            new_label = f"v{i-1}"
            offset = max(0, cumulative - duration_s)
            filter_parts.append(
                f"[{prev_label}][{i}:v]xfade=transition=fade:duration={duration_s:.3f}"
                f":offset={offset:.3f}[{new_label}]"
            )
            cumulative += clip_durs[i] - duration_s
            prev_label = new_label

        final_label = f"v{len(clip_paths)-2}"
        filter_complex = ";".join(filter_parts)
        command = [
            "ffmpeg", "-y", "-loglevel", "error",
            *inputs,
            "-filter_complex", filter_complex,
            "-map", f"[{final_label}]",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p",
            output_path,
        ]
    else:
        concat_file = Path(output_path).with_suffix(".concat.txt")
        concat_file.write_text(
            "".join(f"file '{p.as_posix()}'\n" for p in clip_paths), encoding="utf-8"
        )
        command = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", str(concat_file),
            "-c", "copy", output_path,
        ]

    result = run(command, 300)

    if result.returncode:
        print("[AVISO] xfade falló, usando concat simple")
        concat_file = Path(output_path).with_suffix(".concat.txt")
        concat_file.write_text(
            "".join(f"file '{p.as_posix()}'\n" for p in clip_paths), encoding="utf-8"
        )
        command = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", str(concat_file),
            "-c", "copy", output_path,
        ]
        result = run(command, 300)
        if result.returncode:
            raise RuntimeError(f"Concat: {result.stderr[-1000:]}")


def _merge_audio_and_video(
    video_path: str, audio_path: str, output_path: str,
    config: VideoConfig, encoder: str, preset: str, crf: int,
) -> None:
    """Mezcla video (sin audio) con el audio maestro continuo."""
    command = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", video_path, "-i", audio_path,
        "-c:v", encoder, "-preset", preset, "-crf", str(crf),
        "-c:a", "aac", "-b:a", "192k",
        "-map", "0:v", "-map", "1:a",
        "-shortest", output_path,
    ]
    result = run(command, 600)
    if result.returncode:
        raise RuntimeError(f"Merge audio: {result.stderr[-1000:]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Montaje segmentado de vídeo manhwa")
    parser.add_argument("--panels", required=True, help="Carpeta con las imágenes de paneles")
    parser.add_argument("--audio", required=True, help="Audio maestro continuo (full_audio.mp3)")
    parser.add_argument("--segments", required=True, help="segments.json con la segmentación")
    parser.add_argument("--output", required=True, help="Ruta del video final")
    parser.add_argument("--resolution", default="1920x1080")
    parser.add_argument("--fps", type=int, default=30, choices=(24, 25, 30, 60))
    parser.add_argument("--crf", type=int, default=23)
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--encoder", default="libx264")
    parser.add_argument("--music", default=None, help="Pista de música de fondo (legacy)")
    parser.add_argument("--music-base-dir", default=None, help="Carpeta con subcarpetas tension/ action/ sad/ para música por categoría")
    parser.add_argument("--music-volume", type=float, default=-22.0)
    parser.add_argument("--crossfade", type=int, default=12, help="Frames de crossfade (0 = desactivado)")
    args = parser.parse_args()

    try:
        assemble_from_segments(
            panels_dir=args.panels,
            audio_path=args.audio,
            segments_path=args.segments,
            output_path=args.output,
            resolution=args.resolution,
            fps=args.fps,
            crf=args.crf,
            preset=args.preset,
            encoder=args.encoder,
            music=args.music,
            music_base_dir=args.music_base_dir,
            music_volume=args.music_volume,
            crossfade_frames=args.crossfade,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        sys.exit(f"[ERROR] {exc}")


def assemble_final_video(matched_scenes: list[dict], output_path: Path,
                         resolution: str = "1920x1080", fps: int = 30,
                         preset: str = "medium", crf: int = 23, encoder: str = "libx264",
                         seed: int = 42) -> Path:
    """Wrapper: ensambla video final desde escenas emparejadas, sin subprocess."""
    from config import VideoConfig, parse_resolution

    width, height = parse_resolution(resolution)
    config = VideoConfig(width=width, height=height, fps=fps)

    scenes = matched_scenes
    temp_dir = Path(tempfile.mkdtemp(prefix="manhwa_clips_"))
    clips = []

    for index, scene in enumerate(scenes, start=1):
        clip = temp_dir / f"clip_{index:04d}.mp4"
        try:
            dmult = scene.get("duration_mult", 1.0)
            build_clip(scene, str(clip),
                       preset, crf, encoder, config)
            clips.append(clip)
            scene["duration"] = clip_duration(scene, dmult)
        except Exception as exc:
            # shutil.rmtree en vez de borrar clip a clip + rmdir(): rmdir()
            # exige un directorio vacio, y aqui siempre queda algo suelto
            # (el clip parcialmente escrito que fallo, que nunca entra en
            # `clips` porque solo se appendea tras un build_clip exitoso).
            # Con rmdir() eso lanzaba OSError "Directory not empty" y
            # enmascaraba este RuntimeError, que es el que de verdad importa.
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise RuntimeError(f"Error generando clip {index}: {exc}") from exc

    concat_file = temp_dir / "concat.txt"
    concat_file.write_text("".join(f"file '{c.as_posix()}'\n" for c in clips), encoding="utf-8")

    temp_output = output_path.with_suffix(".temp.mp4")
    result = run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", str(concat_file), "-c", "copy", "-movflags", "+faststart", str(temp_output)],
        1200,
    )
    if result.returncode:
        raise RuntimeError(f"Concatenacion: {result.stderr[-1500:]}")

    try:
        normalize_final_audio(str(temp_output), str(output_path), config)
    except Exception as exc:
        print(f"[AVISO] Normalizacion de audio fallo, se usa el video sin normalizar: {exc}")
        if temp_output.exists():
            temp_output.replace(output_path)
    finally:
        if temp_output.exists():
            temp_output.unlink()
        # shutil.rmtree en vez de borrar clip a clip + rmdir(): concat.txt
        # sigue en temp_dir en este punto y rmdir() exige un directorio
        # vacio, asi que la limpieza fallaba con OSError en practicamente
        # cada ejecucion exitosa.
        shutil.rmtree(temp_dir, ignore_errors=True)

    return output_path


def clip_duration(scene: dict, dmult: float) -> float:
    """Calcula duracion estimada de un clip."""
    try:
        seconds = duration(scene["audio_path"])
        return seconds * dmult
    except Exception:
        return 5.0


from pipeline_steps_base import PipelineStep, PipelineContext as PipelineCtx
from config import VideoConfig, parse_resolution


class AssembleVideoStep(PipelineStep):

    def __init__(self, resolution: str = "1920x1080", fps: int = 30, crf: int = 23,
                 preset: str = "medium", encoder: str = "libx264",
                 music: str | None = None, music_base_dir: str | None = None,
                 music_volume: float = -22.0,
                 crossfade: int = 12,
                 max_workers: int = 1, target_duration: float = 600.0):
        super().__init__("assemble_video")
        self.resolution = resolution
        self.fps = fps
        self.crf = crf
        self.preset = preset
        self.encoder = encoder
        self.music = music
        self.music_base_dir = music_base_dir
        self.music_volume = music_volume
        self.crossfade = crossfade
        self.max_workers = max_workers
        self.target_duration = target_duration

    def validate_contract(self, context: PipelineCtx) -> bool:
        from pathlib import Path
        audio_path = Path(context.state.get("audio_path", ""))
        if not audio_path.is_file():
            audio_dir = Path(context.output_dir) / "audio"
            audio_path = audio_dir / "full_audio.mp3"
            if not audio_path.is_file():
                self.logger.error("Contrato violado: No se encuentra el audio maestro")
                return False
        segments_path = Path(context.output_dir) / "segments.json"
        if not segments_path.is_file():
            self.logger.error("Contrato violado: No se encuentra segments.json")
            return False
        panels_dir = Path(context.output_dir)
        if not panels_dir.is_dir():
            self.logger.error(f"Contrato violado: No existe el directorio de paneles: {context.output_dir}")
            return False
        return True

    def should_skip(self, context: PipelineCtx) -> bool:
        if super().should_skip(context):
            return True
        if getattr(context, 'force', False):
            return False
        return (Path(context.output_dir) / "video_final.mp4").is_file()

    def execute(self, context: PipelineCtx) -> bool:
        if self.should_skip(context):
            self.logger.info(f"Saltando fase {self.name}")
            return True
        try:
            import subprocess, sys
            output_video = Path(context.output_dir) / "video_final.mp4"
            audio_path = context.state.get("audio_path", "")
            if not audio_path:
                audio_dir = Path(context.output_dir) / "audio"
                audio_path = str(audio_dir / "full_audio.mp3")
            audio_path = str(audio_path)
            segments_path = str(Path(context.output_dir) / "segments.json")

            # Render paralelo con slicer si aplica
            use_slicer = (
                self.max_workers > 1
                and self.target_duration > 300
                and Path(segments_path).is_file()
                and Path(audio_path).is_file()
            )
            if use_slicer:
                n_chunks = min(self.max_workers * 2, max(2, int(self.target_duration / 300)))
                n_chunks = max(2, min(n_chunks, 12))
                self.logger.info(
                    f"[PARALELO] Usando Render Slicer: {n_chunks} chunks, "
                    f"{self.max_workers} workers"
                )
                from render_slicer import render_parallel
                render_parallel(
                    panels_dir=context.output_dir,
                    segments_path=segments_path,
                    audio_path=audio_path,
                    output_path=str(output_video),
                    n_chunks=n_chunks,
                    workers=self.max_workers,
                    resolution=self.resolution,
                    fps=self.fps,
                    crf=self.crf,
                    preset=self.preset,
                    encoder=self.encoder,
                    music_base_dir=self.music_base_dir,
                    crossfade=self.crossfade,
                )
            else:
                self.logger.info(f"Ensamblando video segmentado para: {context.project_name}")
                cmd = [
                    sys.executable, "-m", "assemble_video",
                    "--panels", str(context.output_dir),
                    "--audio", audio_path,
                    "--segments", segments_path,
                    "--output", str(output_video),
                    "--resolution", self.resolution,
                    "--fps", str(self.fps),
                    "--crf", str(self.crf),
                    "--preset", self.preset,
                    "--encoder", self.encoder,
                    "--crossfade", str(self.crossfade),
                ]
                if self.music_base_dir:
                    cmd += ["--music-base-dir", self.music_base_dir]
                elif self.music:
                    cmd += ["--music", self.music, "--music-volume", str(self.music_volume)]
                timeout = max(900, int(self.target_duration * 3))
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
                if result.returncode:
                    self.logger.error(f"Error ensamblando video: {result.stderr[-1500:]}")
                    return False
            context.state["video_path"] = str(output_video)
            self.logger.info(f"Video ensamblado en: {output_video}")
            return True
        except Exception as e:
            return self.on_error(context, e)


if __name__ == "__main__":
    main()
