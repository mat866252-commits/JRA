"""Vision Analyzer V1: Extraccion visual objetiva con Gemini Flash Lite.

Procesa todas las imagenes de un capitulo en batches dinamicos.
Genera vision_detail.json con SOLO hechos visuales (CERO interpretacion).
Cachea por SHA256 + prompt_version + model.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path


logger = logging.getLogger("vision_analyzer")

PROMPT_VERSION = "2026-07-21"
DEFAULT_MAX_INPUT_TOKENS = 2000
DEFAULT_MIN_BATCH = 2
DEFAULT_MAX_BATCH = 8
TOKENS_PER_IMAGE = 258
TOKENS_PER_CHAR = 0.25

CACHE_DIR = Path("data/vision_cache")

PROMPT_DESCRIBE = """Eres un analizador visual objetivo de manhwa. Describe SOLO lo que ves, sin interpretar.

Para cada imagen, responde con JSON EXACTO:

{
  "panels": [
    {
      "index": 0,
      "visual": {
        "characters": [
          {
            "name": null,
            "gender": {"value": "male|female|unknown", "confidence": 0.0-1.0},
            "emotion": {"value": "anger|sadness|joy|surprise|fear|disgust|neutral|determination", "confidence": 0.0-1.0},
            "position": {"value": "center|left|right|background|foreground", "confidence": 0.0-1.0},
            "gaze": {"value": "forward|left|right|up|down|towards_another|unknown", "confidence": 0.0-1.0}
          }
        ],
        "environment": {"value": "descripcion del escenario", "confidence": 0.0-1.0},
        "action": {"value": "descripcion de la accion objetiva", "confidence": 0.0-1.0},
        "dialogue": {"value": null, "confidence": null},
        "visible_text": {"value": null, "confidence": null},
        "camera": {"value": "close-up|medium|wide|extreme-wide|over-shoulder|point-of-view", "confidence": 0.0-1.0},
        "lighting": {"value": "brillante|oscuro|contraste|neblina|foco|natural|artificial", "confidence": 0.0-1.0},
        "changes_from_previous": {"value": "descripcion objetiva de cambios respecto al panel anterior o null si es el primero", "confidence": 0.0-1.0}
      }
    }
  ]
}

REGLAS:
- Usa NULL si no hay personajes, dialogo, o texto visible
- El campo "confidence" debe reflejar que tan seguro estas de lo que ves
- Los personajes deben estar en orden de izquierda a derecha
- Solo describe hechos observables: colores, posiciones, acciones fisicas, objetos visibles
- NO inferences, NO emociones de fondo, NO interpretacion narrativa
- Si el dialogo es legible en globos, transcribelo exactamente"""


def _image_hash(image_path: str) -> str:
    try:
        with open(image_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except FileNotFoundError:
        logger.warning("Imagen no encontrada para hash: %s", image_path)
        return hashlib.sha256(image_path.encode()).hexdigest()


def _estimate_tokens(panel: dict) -> int:
    base = TOKENS_PER_IMAGE
    ocr = panel.get("ocr_text", "") or ""
    text_overlay = int(len(ocr) * TOKENS_PER_CHAR)
    return base + text_overlay


def _build_batches(panels: list[dict], max_tokens: int = DEFAULT_MAX_INPUT_TOKENS) -> list[list[dict]]:
    batches: list[list[dict]] = []
    current_batch: list[dict] = []
    current_tokens = 0

    for panel in panels:
        tokens = _estimate_tokens(panel)
        if current_batch and current_tokens + tokens > max_tokens:
            if len(current_batch) >= DEFAULT_MIN_BATCH:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0
            else:
                pass
        current_batch.append(panel)
        current_tokens += tokens

        if len(current_batch) >= DEFAULT_MAX_BATCH:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0

    if current_batch:
        batches.append(current_batch)

    return batches


def _get_cache_key(image_path: str) -> str:
    h = _image_hash(image_path)
    raw = f"{h}|{PROMPT_VERSION}|gemini-flash-lite-latest"
    return hashlib.sha256(raw.encode()).hexdigest()


def _load_cache(cache_key: str) -> dict | None:
    cache_path = CACHE_DIR / f"{cache_key}.json"
    if cache_path.is_file():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
    return None


def _save_cache(cache_key: str, data: dict):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_DIR / f"{cache_key}.tmp"
    dst = CACHE_DIR / f"{cache_key}.json"
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(dst)


def _parse_vision_response(response_text: str, batch_panels: list[dict]) -> list[dict]:
    try:
        parsed = json.loads(response_text)
        results = parsed.get("panels", [])
    except json.JSONDecodeError:
        logger.warning("[VISION] Respuesta JSON invalida de Gemini, usando fallback")
        results = []

    panel_results = []
    for i, panel in enumerate(batch_panels):
        panel_id = panel.get("scene", i)
        if i < len(results):
            result = results[i]
            visual = result.get("visual", {})
        else:
            visual = {
                "characters": [],
                "environment": {"value": "", "confidence": 0.0},
                "action": {"value": "", "confidence": 0.0},
                "dialogue": None,
                "visible_text": None,
                "camera": {"value": "unknown", "confidence": 0.0},
                "lighting": {"value": "unknown", "confidence": 0.0},
                "changes_from_previous": {"value": None, "confidence": 0.0},
            }
        panel_results.append({
            "panel_id": panel_id,
            "image_hash": _image_hash(str(panel.get("_image_path", ""))),
            "file": panel.get("file", f"escena_{panel_id:04d}.png"),
            "visual": visual,
        })

    return panel_results


def analyze_panels(
    panels_dir: str,
    provider,
    output_dir: str | None = None,
    max_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
    force: bool = False,
) -> dict:
    """Procesa todas las imagenes del capitulo con Gemini Vision.

    Args:
        panels_dir: Directorio con panels.json y las imagenes.
        provider: LLMProvider con capacidad de vision (Gemini).
        output_dir: Directorio de salida para vision_detail.json.
        max_tokens: Tokens maximos por batch.
        force: Si True, re-procesa aunque el cache exista.

    Returns:
        dict con vision_detail.json completo.
    """
    manifest_path = Path(panels_dir) / "panels.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"No se encontro {manifest_path}")

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    all_panels = data.get("panels", [])
    if not all_panels:
        raise RuntimeError("panels.json esta vacio")

    panels_dir_path = Path(panels_dir)
    for p in all_panels:
        fname = p.get("file", f"escena_{p.get('scene', 0):04d}.png")
        p["_image_path"] = str(panels_dir_path / fname)

    all_panels.sort(key=lambda p: p.get("scene", 0))

    batches = _build_batches(all_panels, max_tokens=max_tokens)
    logger.info(f"[VISION] {len(all_panels)} paneles en {len(batches)} batches")

    all_results = []
    cache_hits = 0
    cache_misses = 0

    for batch_idx, batch in enumerate(batches):
        batch_results = []
        uncached_indices = []
        uncached_panels = []

        for i, panel in enumerate(batch):
            cache_key = _get_cache_key(panel["_image_path"])
            cached = _load_cache(cache_key) if not force else None
            if cached:
                batch_results.append((i, cached))
                cache_hits += 1
            else:
                uncached_indices.append(i)
                uncached_panels.append(panel)
                cache_misses += 1

        if uncached_panels:
            image_paths = [p["_image_path"] for p in uncached_panels]
            context = ""
            if batch_idx > 0 and all_results:
                last = all_results[-1]
                prev_vis = last.get("visual", {})
                prev_action = prev_vis.get("action", {}).get("value", "")
                prev_env = prev_vis.get("environment", {}).get("value", "")
                context = f"[CONTEXTO - Panel anterior]: Escena: {prev_env}. Accion: {prev_action}.\n"

            batch_descs = "\n".join(
                f"Panel {j}: archivo={os.path.basename(p['_image_path'])}, "
                f"texto_visible={p.get('ocr_text', '')[:100] or 'ninguno'}"
                for j, p in enumerate(uncached_panels)
            )

            prompt = (
                f"{PROMPT_DESCRIBE}\n\n"
                f"{context}"
                f"IMAGENES A ANALIZAR ({len(uncached_panels)} paneles):\n"
                f"{batch_descs}\n\n"
                "Analiza las imagenes en el ORDEN indicado y responde el JSON."
            )

            logger.info(f"[VISION] Batch {batch_idx + 1}/{len(batches)}: "
                        f"{len(uncached_panels)} paneles (cache miss)")
            try:
                response = provider.describe_multi(image_paths, prompt)
                parsed_results = _parse_vision_response(
                    response, uncached_panels,
                )
                for j, panel in enumerate(uncached_panels):
                    if j < len(parsed_results):
                        result = parsed_results[j]
                        cache_key = _get_cache_key(panel["_image_path"])
                        _save_cache(cache_key, result)
                        batch_results.append((uncached_indices[j], result))
            except Exception as exc:
                logger.error(f"[VISION] Error en batch {batch_idx + 1}: {exc}")
                for j, panel in enumerate(uncached_panels):
                    batch_results.append((uncached_indices[j], {
                        "panel_id": panel.get("scene", uncached_indices[j]),
                        "image_hash": _image_hash(panel["_image_path"]),
                        "file": panel.get("file", f"escena_{panel.get('scene', 0):04d}.png"),
                        "visual": {
                            "characters": [],
                            "environment": {"value": "", "confidence": 0.0},
                            "action": {"value": "", "confidence": 0.0},
                            "dialogue": None,
                            "visible_text": None,
                            "camera": {"value": "unknown", "confidence": 0.0},
                            "lighting": {"value": "unknown", "confidence": 0.0},
                            "changes_from_previous": {"value": None, "confidence": 0.0},
                        },
                        "error": str(exc)[:200],
                    }))

        batch_results.sort(key=lambda x: x[0])
        for _, result in batch_results:
            all_results.append(result)

        if batch_idx < len(batches) - 1:
            time.sleep(0.5)

    for r in all_results:
        vis = r.get("visual", {})
        confs = []
        for key in ("environment", "action", "camera", "lighting"):
            field = vis.get(key, {})
            if isinstance(field, dict) and field.get("confidence"):
                confs.append(field["confidence"])
        for char in vis.get("characters", []):
            for k in ("gender", "emotion", "position", "gaze"):
                field = char.get(k, {})
                if isinstance(field, dict) and field.get("confidence"):
                    confs.append(field["confidence"])
        r["vision_quality"] = round(sum(confs) / len(confs), 2) if confs else 0.0

    output = {
        "schema_version": 2,
        "generated_by": {
            "model": "gemini-flash-lite-latest",
            "provider": "google",
            "prompt_version": PROMPT_VERSION,
            "timestamp": __import__("datetime").datetime.now().isoformat(),
            "pipeline_version": "v3",
        },
        "chapter": data.get("chapter", 1),
        "total_panels": len(all_panels),
        "cache_stats": {"hits": cache_hits, "misses": cache_misses},
        "panels": all_results,
    }

    out_dir = Path(output_dir) if output_dir else Path(panels_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "vision_detail.json"
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"[VISION] {len(all_results)} paneles analizados → {output_path}")
    logger.info(f"[VISION] Cache: {cache_hits} hits, {cache_misses} misses")

    return output


if __name__ == "__main__":
    import argparse
    from llm.factory import LLMProviderFactory

    parser = argparse.ArgumentParser(description="Vision Analyzer: extraccion visual con Gemini")
    parser.add_argument("--panels-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_INPUT_TOKENS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    provider = LLMProviderFactory.create(
        "gemini",
        model="gemini-flash-lite-latest",
        timeout=120,
    )
    analyze_panels(
        panels_dir=args.panels_dir,
        provider=provider,
        output_dir=args.output_dir,
        max_tokens=args.max_tokens,
        force=args.force,
    )
