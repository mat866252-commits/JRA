"""Curador V5: Seleccion narrativa de escenas con DeepSeek.

Lee interpretation.json + scenes.json y selecciona 15-25 escenas
optimas para narracion basado en importancia narrativa y cobertura
de la trama.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger("curador_v5")

TARGET_SCENES_MIN = 15
TARGET_SCENES_MAX = 25


PROMPT_CURATE = """Eres un CURADOR NARRATIVO experto en manhwa. Debes seleccionar entre {MIN} y {MAX} escenas de las {TOTAL} disponibles para crear un video narrado de 8-12 minutos.

DATOS DE CADA ESCENA:
{SCENES_DATA}

INSTRUCCIONES:
- Selecciona las escenas MAS IMPORTANTES para la narrativa principal
- Prioriza escenas con: dialogo clave, revelaciones, desarrollo de personajes, accion importante
- Omite escenas redundantes, transiciones largas, o relleno
- Debes seleccionar entre {MIN} y {MAX} escenas
- Tu seleccion debe contar la historia completa de forma coherente

Responde EXACTAMENTE este JSON sin explicaciones ni markdown:

{{
  "selected_scenes": [1, 3, 5, ...],
  "reasoning": "explicacion breve de tu criterio de seleccion",
  "narrative_arc": {
    "setup": "escenas que establecen el conflicto",
    "confrontation": "escenas de desarrollo/climax",
    "resolution": "escenas de resolucion"
  }
}}"""


def select_scenes(
    scenes_path: str,
    interpretation_path: str,
    provider,
    output_dir: str | None = None,
) -> dict:
    """Selecciona escenas usando DeepSeek.

    Args:
        scenes_path: Ruta a scenes.json.
        interpretation_path: Ruta a interpretation.json.
        provider: LLMProvider (DeepSeek).
        output_dir: Directorio de salida para selected_panels.json.

    Returns:
        dict con selected_panels.json.
    """
    scenes = json.loads(Path(scenes_path).read_text(encoding="utf-8"))
    interp = json.loads(Path(interpretation_path).read_text(encoding="utf-8"))

    interp_map = {p["panel_id"]: p["interpretation"] for p in interp.get("panels", [])}

    scenes_list = scenes.get("scenes", [])
    total_scenes = len(scenes_list)
    total_panels = sum(len(s.get("panel_ids", [])) for s in scenes_list)

    scenes_text_parts = []
    for scene in scenes_list:
        sid = scene["scene_id"]
        pids = scene.get("panel_ids", [])
        panel_lines = []
        for pid in pids:
            interp_data = interp_map.get(pid, {})
            summary = interp_data.get("summary", "")
            category = interp_data.get("inferred_category", {}).get("value", "unknown")
            importance = interp_data.get("narrative_importance", {}).get("value", 5)
            panel_lines.append(
                f"    Panel {pid}: [{category}] importancia={importance} | {summary[:120]}"
            )
        scenes_text_parts.append(
            f"  Escena {sid} ({len(pids)} paneles):\n" + "\n".join(panel_lines)
        )

    scenes_text = "\n".join(scenes_text_parts)

    effective_min = min(TARGET_SCENES_MIN, max(1, total_scenes))
    effective_max = min(TARGET_SCENES_MAX, total_scenes)

    prompt = PROMPT_CURATE.format(
        MIN=effective_min,
        MAX=effective_max,
        TOTAL=total_scenes,
        SCENES_DATA=scenes_text,
    )

    logger.info(f"[CURADOR] Enviando {total_scenes} escenas ({total_panels} paneles) a {provider.model if hasattr(provider, 'model') else 'DeepSeek'}...")
    response = provider.describe_text(prompt)

    try:
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
        else:
            parsed = json.loads(response)
    except (json.JSONDecodeError, Exception) as exc:
        logger.error(f"[CURADOR] Error parseando respuesta: {exc}")
        logger.debug(f"Respuesta: {response[:500]}")
        parsed = {"selected_scenes": [], "reasoning": "fallback: error de parseo", "narrative_arc": {}}

    selected_ids = parsed.get("selected_scenes", [])
    if not selected_ids:
        logger.warning("[CURADOR] No se seleccionaron escenas, usando heuristica de respaldo")
        scored = []
        for scene in scenes_list:
            sid = scene["scene_id"]
            avg_imp = 5
            if scene.get("panel_ids"):
                scores = [
                    interp_map.get(pid, {}).get("narrative_importance", {}).get("value", 5)
                    for pid in scene["panel_ids"]
                ]
                avg_imp = sum(scores) / len(scores)
            scored.append((sid, avg_imp))
        scored.sort(key=lambda x: -x[1])
        selected_ids = [s[0] for s in scored[:TARGET_SCENES_MAX]]

    selected_ids.sort()

    all_panels = []
    for scene in scenes_list:
        if scene["scene_id"] in selected_ids:
            for pid in scene.get("panel_ids", []):
                all_panels.append(pid)

    scene_map = {s["scene_id"]: s for s in scenes_list}
    selected_scenes_detail = []
    for sid in selected_ids:
        scene = scene_map.get(sid, {})
        selected_scenes_detail.append({
            "scene_id": sid,
            "panel_ids": scene.get("panel_ids", []),
            "num_panels": len(scene.get("panel_ids", [])),
        })

    output = {
        "schema_version": 2,
        "generated_by": {
            "model": provider.model if hasattr(provider, "model") else "deepseek-chat",
            "provider": "deepseek",
            "prompt_version": "2026-07-21",
            "timestamp": __import__("datetime").datetime.now().isoformat(),
        },
        "chapter": scenes.get("chapter", 1),
        "total_scenes_available": total_scenes,
        "total_panels_available": total_panels,
        "num_selected_scenes": len(selected_ids),
        "num_selected_panels": len(all_panels),
        "selected_scenes": selected_scenes_detail,
        "selected_panels": all_panels,
        "reasoning": parsed.get("reasoning", ""),
        "narrative_arc": parsed.get("narrative_arc", {}),
    }

    out_dir = Path(output_dir) if output_dir else Path(scenes_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "selected_panels.json"
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"[CURADOR] {len(selected_ids)} escenas seleccionadas → {output_path}")

    return output


if __name__ == "__main__":
    import argparse
    from llm.factory import LLMProviderFactory

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", required=True)
    parser.add_argument("--interpretation", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--text-provider", default="openrouter")
    parser.add_argument("--text-model", default="google/gemini-3.5-flash")
    args = parser.parse_args()

    provider = LLMProviderFactory.create(
        args.text_provider,
        model=args.text_model,
        timeout=180,
    )
    select_scenes(
        scenes_path=args.scenes,
        interpretation_path=args.interpretation,
        provider=provider,
        output_dir=args.output_dir,
    )
