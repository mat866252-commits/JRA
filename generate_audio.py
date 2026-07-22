#!/usr/bin/env python3
"""Genera, valida y reutiliza un audio por escena con Edge TTS."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import re
import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path

logger = logging.getLogger("generate_audio")

SCENE_HEADER = re.compile(r"^\s*ESCENA\s+0*(\d+)\s*:\s*(.*)$", re.IGNORECASE)
COMMENTS = ("#", "//")

# Ajuste de ritmo/tono por categoría narrativa del grupo.
# Los valores son deltas sobre el --rate/--pitch base.
# Aumentados para que la diferencia sea NOTORIA entre escenas.
CATEGORY_PROSODY_DELTA = {
    "accion": (35, 15),
    "accion_grupal": (30, 12),
    "conversacion": (5, 2),
    "primer_plano": (-20, -10),
    "texto_puro": (-10, -3),
    "fondo_vacio": (-25, -8),
    "fondo/texto": (-25, -8),
}
IMPACT_PROSODY_DELTA = (-20, -10)  # Revelaciones/clímax: más lento y grave.


def _parse_percent(value: str) -> float:
    return float(str(value).strip().replace("%", "").replace("+", ""))


def _parse_hz(value: str) -> float:
    return float(str(value).strip().replace("Hz", "").replace("+", ""))


def combine_prosody(base_rate: str, base_pitch: str, category: str, stars: int) -> tuple[str, str]:
    """Combina el rate/pitch base del usuario con el delta de la categoría narrativa del grupo."""
    rate_delta, pitch_delta = list(CATEGORY_PROSODY_DELTA.get(category, (0, 0)))
    if stars >= 4:
        rate_delta += IMPACT_PROSODY_DELTA[0]
        pitch_delta += IMPACT_PROSODY_DELTA[1]
    rate = _parse_percent(base_rate) + rate_delta
    pitch = _parse_hz(base_pitch) + pitch_delta
    rate_str = f"{'+' if rate >= 0 else ''}{rate:.0f}%"
    pitch_str = f"{'+' if pitch >= 0 else ''}{pitch:.0f}Hz"
    return rate_str, pitch_str


def load_groups_meta(groups_meta_path: str | None) -> dict[int, dict]:
    """Carga scene -> {category, stars} desde archivo de metadatos (generado en el paso de guion).
    Si no se aporta o no existe, se sigue funcionando sin ajuste de prosodia por categoría."""
    if not groups_meta_path:
        return {}
    path = Path(groups_meta_path)
    if not path.is_file():
        print(f"[AVISO] --groups-meta no encontrado ({path}); se generará audio con prosodia neutra.")
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {g["scene"]: {"category": g.get("category", ""), "stars": g.get("stars", 1)} for g in data.get("groups", [])}


async def normalize_pacing(input_path: Path, output_path: Path,
                            threshold_db: int = -40) -> None:
    """Recorta silencios al inicio/final con FFmpeg y copia el resultado.
    
    Edge TTS suele dejar 100-300ms de silencio al inicio y final.
    Este filtro los recorta para que entre escenas no haya pausas audibles.
    """
    if not shutil.which("ffprobe"):
        shutil.copy2(input_path, output_path)
        return

    command = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(input_path),
        "-af", f"silenceremove=start_periods=1:start_duration=0.5:start_threshold=-{abs(threshold_db)}dB:"
               f"detection=peak,aformat=dblp,areverse,"
               f"silenceremove=start_periods=1:start_duration=0.5:start_threshold=-{abs(threshold_db)}dB:"
               f"detection=peak,aformat=dblp,areverse",
        "-c:a", "libmp3lame", "-q:a", "2", str(output_path),
    ]
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()
    if process.returncode:
        shutil.copy2(input_path, output_path)


def normalise_text(text: str) -> str:
    """Homogeneiza Unicode y espacios sin alterar el contenido narrativo."""
    text = unicodedata.normalize("NFC", text).replace("\u00a0", " ")
    return re.sub(r"\s+", " ", text).strip()


def get_best_spanish_voice() -> str:
    """Detecta la mejor voz española disponible en Edge TTS.
    V4: Prioridad absoluta a voces MASCULINAS con autoridad (Gerardo, Álvaro, Jorge)."""
    try:
        import asyncio
        import edge_tts
        voices = asyncio.run(edge_tts.list_voices())
        spanish = [v for v in voices if v['Locale'].startswith('es-')]
        priority = [
            "es-MX-GerardoNeural",
            "es-ES-AlvaroNeural",
            "es-MX-JorgeNeural",
            "es-ES-ElviraNeural",
            "es-US-SabinaNeural",
            "es-ES-AbrilNeural",
        ]
        for preferred in priority:
            if any(v['ShortName'] == preferred for v in spanish):
                return preferred
        if spanish:
            return spanish[0]['ShortName']
    except Exception:
        pass
    return "es-MX-GerardoNeural"


def parse_script(path: str) -> list[tuple[int, str]]:
    """Lee escenas consecutivas, admite comentarios y normaliza espacios."""
    scenes: list[tuple[int, str]] = []
    current_number: int | None = None
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_number, current_lines
        if current_number is not None:
            text = normalise_text(" ".join(line.strip() for line in current_lines if line.strip()))
            if not text:
                raise ValueError(f"La escena {current_number:04d} no tiene texto.")
            scenes.append((current_number, text))
        current_number, current_lines = None, []

    script_path = Path(path)
    if not script_path.is_file():
        raise ValueError(f"El archivo de guion no existe: {path}")
    
    content = script_path.read_text(encoding="utf-8-sig")
    if not content.strip():
        raise ValueError("El guion está vacío.")
    
    has_scenes = False
    for raw in content.splitlines():
        line = raw.strip()
        if line.startswith(COMMENTS):
            continue
        match = SCENE_HEADER.match(raw)
        if match:
            has_scenes = True
            flush()
            current_number, current_lines = int(match.group(1)), [match.group(2)]
        elif current_number is not None:
            current_lines.append(raw)
    flush()
    if not has_scenes:
        raise ValueError("No se encontró ninguna cabecera 'ESCENA 0001: texto'. Verifica el formato del guion.")
    if not scenes:
        raise ValueError("No se encontró ninguna escena válida en el guion.")
    numbers = [number for number, _ in scenes]
    if len(numbers) != len(set(numbers)):
        raise ValueError("Hay números de escena repetidos en el guion.")
    if numbers != sorted(numbers):
        raise ValueError("Las escenas deben estar ordenadas de menor a mayor.")
    # Nota: los números de escena identifican grupos narrativos (scene_groups.json), no viñetas
    # individuales consecutivas. Puede haber huecos legítimos (paneles absorbidos en otro grupo),
    # así que ya NO se exige un rango contiguo aquí.
    return scenes


def probe_duration(path: Path) -> float:
    """Verifica que un archivo de audio sea reproducible y tenga duración útil."""
    if not path.is_file() or path.stat().st_size < 256:
        return 0.0
    if not shutil.which("ffprobe"):
        return 1.0  # Edge TTS ya verificó tamaño; no bloquea si FFmpeg no está instalado aún.
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                             "-of", "default=nw=1:nk=1", str(path)], capture_output=True, text=True)
    try:
        return float(result.stdout.strip()) if result.returncode == 0 else 0.0
    except ValueError:
        return 0.0


async def add_silence(input_path: Path, output_path: Path, start: float, end: float) -> None:
    """Aplica silencios sin bloquear el event loop que genera otros audios.
    
    Usa adelay para silencio inicial y apad para silencio final.
    Si adelay no está disponible, genera silencio con anullsrc y concat.
    """
    if start <= 0 and end <= 0:
        shutil.copy2(input_path, output_path)
        return

    filter_parts = []
    if start > 0:
        filter_parts.append(f"adelay={round(start * 1000)}")
    if end > 0:
        filter_parts.append(f"apad=pad_dur={end}")
    af_filter = ",".join(filter_parts)

    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(input_path),
        "-af", af_filter,
        "-c:a", "libmp3lame", "-q:a", "2", str(output_path),
    ]
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()
    if process.returncode:
        stderr_text = stderr.decode(errors="replace")
        if "No such filter" in stderr_text and any(filt in af_filter for filt in ("adelay", "apad")):
            silence = output_path.with_suffix(".silence.wav")

            try:
                probe = await asyncio.create_subprocess_exec(
                    "ffprobe", "-v", "error", "-show_entries",
                    "stream=sample_rate", "-of", "default=nw=1:nk=1",
                    str(input_path),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                probe_out, _ = await probe.communicate()
                ar = probe_out.decode().strip() or "44100"
            except Exception:
                ar = "44100"

            sil_duration = start if start > 0 else end
            gen = await asyncio.create_subprocess_exec(
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "lavfi", "-i", f"anullsrc=r={ar}:cl=stereo",
                "-t", str(sil_duration), str(silence),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            _, gen_stderr = await gen.communicate()
            if gen.returncode:
                raise RuntimeError(f"FFmpeg no pudo generar silencio: {(gen_stderr or b'').decode(errors='replace')[-300:]}")

            if start > 0 and end > 0:
                silence_end = output_path.with_suffix(".silence_end.wav")
                gen2 = await asyncio.create_subprocess_exec(
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-f", "lavfi", "-i", f"anullsrc=r={ar}:cl=stereo",
                    "-t", str(end), str(silence_end),
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.PIPE,
                )
                _, gen2_stderr = await gen2.communicate()
                if gen2.returncode:
                    raise RuntimeError(f"FFmpeg no pudo generar silencio final: {(gen2_stderr or b'').decode(errors='replace')[-300:]}")
                concat_filter = f"[0:a][1:a][2:a]concat=n=3:v=0:a=1"
                fallback = [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-i", str(silence), "-i", str(input_path), "-i", str(silence_end),
                    "-filter_complex", concat_filter,
                    "-c:a", "libmp3lame", "-q:a", "2", str(output_path),
                ]
            elif start > 0:
                concat_filter = f"[0:a][1:a]concat=n=2:v=0:a=1"
                inputs = [str(silence), str(input_path)]
                fallback = [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-i", inputs[0], "-i", inputs[1],
                    "-filter_complex", concat_filter,
                    "-c:a", "libmp3lame", "-q:a", "2", str(output_path),
                ]
            else:
                silence_end = output_path.with_suffix(".silence_end.wav")
                gen2 = await asyncio.create_subprocess_exec(
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-f", "lavfi", "-i", f"anullsrc=r={ar}:cl=stereo",
                    "-t", str(end), str(silence_end),
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.PIPE,
                )
                _, gen2_stderr = await gen2.communicate()
                if gen2.returncode:
                    raise RuntimeError(f"FFmpeg no pudo generar silencio final: {(gen2_stderr or b'').decode(errors='replace')[-300:]}")
                concat_filter = f"[0:a][1:a]concat=n=2:v=0:a=1"
                fallback = [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-i", str(input_path), "-i", str(silence_end),
                    "-filter_complex", concat_filter,
                    "-c:a", "libmp3lame", "-q:a", "2", str(output_path),
                ]
            process = await asyncio.create_subprocess_exec(
                *fallback,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            _, fb_stderr = await process.communicate()
            if silence.exists():
                silence.unlink()
            silence_end = output_path.with_suffix(".silence_end.wav")
            if silence_end.exists():
                silence_end.unlink()
            if process.returncode:
                raise RuntimeError(f"FFmpeg no pudo añadir silencios: {(fb_stderr or b'').decode(errors='replace')[-500:]}")
        else:
            raise RuntimeError(stderr.decode(errors="replace")[-500:] or "FFmpeg no pudo añadir silencios.")


async def generate_one(number: int, text: str, voice: str, output: Path, rate: str, semaphore: asyncio.Semaphore,
                       retries: int, retry_delay: float, silence_start: float, silence_end: float,
                       pitch: str = "+0Hz") -> tuple[int, float]:
    import edge_tts
    async with semaphore:
        temporary = output.with_suffix(output.suffix + ".part")
        error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                await edge_tts.Communicate(text, voice, rate=rate, pitch=pitch).save(str(temporary))
                if probe_duration(temporary) <= 0:
                    raise RuntimeError("El servicio devolvió un audio vacío o ilegible.")
                temporary.replace(output)
                # Normaliza el pacing: recorta silencios al inicio/final para que
                # entre escenas NO haya pausas audibles.
                normalized = output.with_suffix(".normalized.mp3")
                await normalize_pacing(output, normalized)
                try:
                    normalized.replace(output)
                except OSError:
                    shutil.copy2(str(normalized), str(output))
                    try:
                        normalized.unlink()
                    except OSError:
                        logger.warning("No se pudo limpiar .normalized.mp3: %s", normalized)
                # El silencio manual (--silence-start/--silence-end) sigue disponible como override
                # explícito por si se quiere un hueco deliberado más largo en algún punto puntual.
                if silence_start or silence_end:
                    padded = output.with_suffix(".padded.mp3")
                    await add_silence(output, padded, silence_start, silence_end)
                    try:
                        padded.replace(output)
                    except OSError:
                        shutil.copy2(str(padded), str(output))
                        padded.unlink()
                return number, probe_duration(output)
            except Exception as exc:
                error = exc
                if temporary.exists():
                    temporary.unlink()
                if attempt < retries:
                    await asyncio.sleep(retry_delay * (2 ** attempt))
        raise RuntimeError(f"Escena {number:04d}: {error}")


async def generate_all(scenes, args) -> None:
    output_dir = Path(args.output); output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "audio_manifest.json"
    try: previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError): previous = {}
    old = previous.get("scenes", {}); current, tasks = {}, []
    semaphore = asyncio.Semaphore(args.concurrency)
    selected = {args.scene} if args.scene else None
    groups_meta = load_groups_meta(getattr(args, "groups_meta", None))
    for number, text in scenes:
        if selected and number not in selected: continue
        output = output_dir / f"escena_{number:04d}.mp3"
        meta = groups_meta.get(number, {})
        rate, pitch = combine_prosody(args.rate, "+0Hz", meta.get("category", ""), meta.get("stars", 1))
        fingerprint = hashlib.sha256(f"{args.voice}\0{rate}\0{pitch}\0{args.silence_start}\0{args.silence_end}\0{text}".encode()).hexdigest()
        unchanged = old.get(f"{number:04d}", {}).get("fingerprint") == fingerprint
        current[f"{number:04d}"] = {"file": output.name, "fingerprint": fingerprint, "text": text}
        if output.exists() and not args.overwrite and unchanged:
            existing_dur = probe_duration(output)
            if existing_dur > 0:
                current[f"{number:04d}"]["duration"] = existing_dur; continue
        if output.exists() and not args.overwrite and not unchanged:
            print(f"[AVISO] escena {number:04d}: cambió el texto/voz; se regenera.")
        tasks.append(generate_one(number, text, args.voice, output, rate, semaphore, args.retries,
                                  args.retry_delay, args.silence_start, args.silence_end, pitch=pitch))
    errors = []
    for done, task in enumerate(asyncio.as_completed(tasks), 1):
        try:
            number, seconds = await task; current[f"{number:04d}"]["duration"] = round(seconds, 3)
            print(f"[{done}/{len(tasks)}] escena {number:04d} generada")
        except Exception as exc: errors.append(str(exc))
    merged = old if args.scene else {}
    merged.update(current)
    manifest_path.write_text(json.dumps({"voice": args.voice, "rate": args.rate, "scenes": merged}, ensure_ascii=False, indent=2), encoding="utf-8")
    if errors: raise RuntimeError("Fallaron audios:\n- " + "\n- ".join(errors))


async def list_voices() -> None:
    import edge_tts
    for voice in await edge_tts.list_voices(): print(f"{voice['ShortName']}: {voice.get('FriendlyName', '')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--script"); parser.add_argument("--output", default="audios")
    parser.add_argument("--voice", default=get_best_spanish_voice()); parser.add_argument("--rate", default="+0%")
    parser.add_argument("--continuous", action="store_true",
                         help="Modo continuo: genera un SOLO audio para todo el guion (sin ESCENA NNNN:).")
    parser.add_argument("--concurrency", type=int, default=3); parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--retries", type=int, default=6); parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--scene", type=int, help="Genera o regenera una sola escena.")
    parser.add_argument("--silence-start", type=float, default=0.0); parser.add_argument("--silence-end", type=float, default=0.0)
    parser.add_argument("--groups-meta", type=str, default=None,
                         help="Ruta a grupos_meta.json con {scene: {category, stars}} para ajustar rate/pitch por categoría.")
    parser.add_argument("--list-voices", action="store_true")
    args = parser.parse_args()
    if args.list_voices:
        try: asyncio.run(list_voices())
        except ImportError: sys.exit("[ERROR] Instala edge-tts.")
        return
    if not args.script: parser.error("--script es obligatorio salvo con --list-voices.")
    if not Path(args.script).is_file():
        parser.error(f"El archivo de guion no existe: {args.script}")
    if args.concurrency < 1 or args.retries < 0 or args.retry_delay <= 0 or min(args.silence_start, args.silence_end) < 0:
        parser.error("Valores de audio no válidos.")
    try:
        import edge_tts  # noqa: F401

        if args.continuous:
            output_path = Path(args.output)
            if output_path.is_dir():
                output_path = output_path / "full_audio.mp3"
            duration = generate_continuous_audio(
                script_path=args.script,
                output_path=str(output_path),
                voice=args.voice,
                rate=args.rate,
            )
            print(f"[OK] Audio continuo: {output_path} ({duration:.1f}s)")
            return

        scenes = parse_script(args.script)
        if args.scene and args.scene not in dict(scenes): raise ValueError(f"No existe la escena {args.scene:04d}.")
        asyncio.run(generate_all(scenes, args))
    except (ImportError, ValueError, RuntimeError) as exc: sys.exit(f"[ERROR] {exc}")


def generate_all_audio(matched_scenes: list[dict], output_dir: Path,
                       voice: str = "", rate: str = "+0%",
                       concurrency: int = 3, retries: int = 6, retry_delay: float = 2.0,
                       silence_start: float = 0.0, silence_end: float = 0.0,
                       groups_meta_path: str | None = None) -> list[dict]:
    """Wrapper: genera audio para todas las escenas y devuelve metadatos.

    matched_scenes: cada dict debe tener 'number' y 'text'; puede además traer 'category'
    y 'stars' directamente (evita depender de groups_meta_path si ya se tienen a mano).
    """
    scenes = [(s["number"], s["text"]) for s in matched_scenes if s.get("text")]
    inline_meta = {s["number"]: {"category": s.get("category", ""), "stars": s.get("stars", 1)}
                   for s in matched_scenes if s.get("category") is not None}
    groups_meta = inline_meta or load_groups_meta(groups_meta_path)
    if not voice:
        voice = get_best_spanish_voice()
    output_dir.mkdir(parents=True, exist_ok=True)

    import asyncio

    async def _run():
        manifest_path = output_dir / "audio_manifest.json"
        current = {}
        semaphore = asyncio.Semaphore(concurrency)
        tasks = []
        for number, text in scenes:
            output = output_dir / f"escena_{number:04d}.mp3"
            meta = groups_meta.get(number, {})
            scene_rate, scene_pitch = combine_prosody(rate, "+0Hz", meta.get("category", ""), meta.get("stars", 1))
            tasks.append(generate_one(number, text, voice, output, scene_rate, semaphore,
                                      retries, retry_delay, silence_start, silence_end, pitch=scene_pitch))
        errors = []
        audio_meta = []
        for done, task in enumerate(asyncio.as_completed(tasks), 1):
            try:
                number, seconds = await task
                audio_meta.append({"scene": number, "file": f"escena_{number:04d}.mp3", "duration": round(seconds, 3)})
                current[f"{number:04d}"] = {"file": f"escena_{number:04d}.mp3", "duration": round(seconds, 3)}
            except Exception as exc:
                errors.append(str(exc))
        manifest_path.write_text(
            json.dumps({"voice": voice, "rate": rate, "scenes": current}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if errors:
            raise RuntimeError("Fallaron audios:\n- " + "\n- ".join(errors))
        return audio_meta

    return asyncio.run(_run())


async def generate_full_audio(
    script_text: str,
    output_path: Path,
    voice: str,
    rate: str = "+0%",
    pitch: str = "+0Hz",
) -> float:
    """Genera un ÚNICO archivo de audio continuo para todo el guion.

    A diferencia del método anterior (un audio por escena), esto produce
    un solo archivo de audio SIN cortes ni pausas entre escenas.

    Args:
        script_text: Texto completo del guion (sin cabeceras ESCENA NNNN:).
        output_path: Ruta del archivo mp3 de salida.
        voice: Voz Edge TTS.
        rate: Velocidad (ej: "+0%").
        pitch: Tono (ej: "+0Hz").

    Returns:
        Duración en segundos del audio generado.
    """
    import edge_tts

    if not script_text.strip():
        raise ValueError("El guion está vacío")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp = output_path.with_suffix(".part.mp3")

    try:
        communicate = edge_tts.Communicate(script_text, voice, rate=rate, pitch=pitch)
        await communicate.save(str(temp))

        if not temp.exists() or temp.stat().st_size < 256:
            raise RuntimeError("Edge TTS devolvió un archivo vacío")

        # V4: Post-procesamiento FFmpeg — compand profesional + loudnorm
        # Filtro exacto: compand=0.3|0.3:1|1:-90/-60|-60/-40|-40/-30|-20/-20:6:0:-90:0.2, loudnorm
        # Esto elimina el sonido metálico y suena a micrófono profesional
        processed = output_path.with_suffix(".eq.mp3")
        pro_filter = (
            "compand=0.3|0.3:1|1:-90/-60|-60/-40|-40/-30|-20/-20:6:0:-90:0.2,"
            "loudnorm=I=-16:TP=-1.5:LRA=11"
        )

        process = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(temp),
            "-af", pro_filter,
            "-c:a", "libmp3lame", "-q:a", "2",
            str(processed),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate()

        if process.returncode == 0 and processed.exists() and processed.stat().st_size > 256:
            processed.replace(output_path)
        else:
            # Fallback: copia raw si FFmpeg falla
            temp.replace(output_path)
            if process.returncode:
                print(f"[AVISO] Post-procesamiento FFmpeg falló: {stderr.decode(errors='replace')[-300:]}")

        duration = probe_duration(output_path)
        return duration
    except Exception:
        if temp.exists():
            temp.unlink()
        raise



def generate_continuous_audio(
    script_path: str,
    output_path: str,
    voice: str | None = None,
    rate: str = "+0%",
    pause_ms: int = 600,
) -> float:
    """Wrapper síncrono para generate_full_audio.

    Inserta pausas de ``pause_ms`` milisegundos entre párrafos
    usando elipsis, que Edge TTS interpreta como silencio natural.
    """
    from pathlib import Path

    raw_script = Path(script_path).read_text(encoding="utf-8-sig")

    # Inserta pausas entre párrafos (secciones separadas por doble salto de línea)
    import re as _re
    paragraphs = _re.split(r"\n\s*\n+", raw_script.strip())
    if len(paragraphs) > 1:
        script = "\n...\n".join(paragraphs)
    else:
        script = raw_script

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if not voice:
        voice = get_best_spanish_voice()

    async def _run():
        return await generate_full_audio(script, out, voice, rate)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        duration = asyncio.run(_run())
        print(f"[AUDIO] Audio continuo generado: {duration:.1f}s ({duration/60:.1f} min)")
        print(f"[AUDIO] Voz: {voice} | Archivo: {out}")
        return duration

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor() as pool:
        future = pool.submit(lambda: asyncio.run(_run()))
        duration = future.result()

    print(f"[AUDIO] Audio continuo generado: {duration:.1f}s ({duration/60:.1f} min)")
    print(f"[AUDIO] Voz: {voice} | Archivo: {out}")
    return duration


from pipeline_steps_base import PipelineStep, PipelineContext as PipelineCtx


class GenerateAudioStep(PipelineStep):

    def __init__(self, parallel: bool = True, max_workers: int = 4):
        super().__init__("generate_audio")
        self.parallel = parallel
        self.max_workers = max_workers

    def validate_contract(self, context: PipelineCtx) -> bool:
        from pathlib import Path
        script = Path(context.script_path)
        if not script.exists():
            self.logger.error(f"Contrato violado: No se encuentra el guion en {context.script_path}")
            return False
        return True

    def should_skip(self, context: PipelineCtx) -> bool:
        if super().should_skip(context):
            return True
        if getattr(context, 'force', False):
            return False
        audio_path = Path(context.output_dir) / "audio" / "full_audio.mp3"
        return audio_path.is_file()

    def execute(self, context: PipelineCtx) -> bool:
        if self.should_skip(context):
            self.logger.info(f"Saltando fase {self.name}")
            return True
        try:
            self.logger.info(f"Generando audio continuo para: {context.project_name}")
            from pathlib import Path
            audio_dir = Path(context.output_dir) / "audio"
            audio_dir.mkdir(parents=True, exist_ok=True)
            audio_path = audio_dir / "full_audio.mp3"

            voice = get_best_spanish_voice()

            if context.state.get("use_continuous_audio", True):
                import subprocess, sys
                cmd = [
                    sys.executable, "-m", "generate_audio",
                    "--script", context.script_path,
                    "--output", str(audio_dir),
                    "--voice", voice,
                    "--continuous",
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode:
                    self.logger.error(f"Error generando audio continuo: {result.stderr}")
                    return False
            else:
                groups_meta_path = Path(context.output_dir) / "groups_meta.json"
                cmd = [
                    sys.executable, "-m", "generate_audio",
                    "--script", context.script_path,
                    "--output", str(audio_dir),
                    "--voice", voice,
                ]
                if groups_meta_path.is_file():
                    cmd.extend(["--groups-meta", str(groups_meta_path)])
                if self.parallel:
                    cmd.extend(["--concurrency", str(self.max_workers)])
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode:
                    self.logger.error(f"Error generando audio: {result.stderr}")
                    return False

            context.state["audio_dir"] = str(audio_dir)
            context.state["audio_path"] = str(audio_path)
            self.logger.info(f"Audio generado en: {audio_path}")
            return True
        except Exception as e:
            return self.on_error(context, e)


if __name__ == "__main__":
    main()
