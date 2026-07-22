"""Script Generator V3: Generacion de guion con DeepSeek.

Lee selected_panels.json + interpretation.json + vision_detail.json
y genera guion narrativo con segmentacion temporal.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger("script_generator_v3")

TARGET_DURATION_SEC = 600
TARGET_DURATION_TOLERANCE = 120
SECONDS_PER_PANEL = 4.5

PROMPT_SCRIPT = """Eres un NARRADOR Y GUIONISTA experto en manhwa. Recibes datos de escenas seleccionadas y debes generar:
1. Un GUION NARRATIVO optimizado para locucion en espanol latino (8-12 minutos)
2. La SEGMENTACION temporal de cada panel para la voz y el video

ESCENAS SELECCIONADAS ({NUM_SCENES} escenas, {NUM_PANELS} paneles, capitulo {CHAPTER}):
{SCENES_DETAIL}

Responde EXACTAMENTE este JSON sin explicaciones ni markdown:

{{
  "script": {{
    "intro": "parrafo de apertura (1-2 frases, ~15 seg)",
    "segments": [
      {{
        "scene_id": 1,
        "panel_ids": [1, 2, 3],
        "narration": "texto de narracion para esta escena (10-30 seg de locucion)",
        "estimated_duration_sec": 12,
        "tone": "narrativo|dramatico|intenso|reflexivo|descriptivo",
        "audio_cue": "sugerencia de fondo musical/cortina"
      }}
    ],
    "outro": "parrafo de cierre (1-2 frases, ~15 seg)"
  }},
  "segmentation": [
    {{
      "panel_id": 1,
      "scene_id": 1,
      "duration_sec": 4.5,
      "narration_text": "texto de la narracion para ESTE panel especifico",
      "transition": "cut|fade|dissolve|none",
      "audio_sync": 0
    }}
  ]
}}

REGLAS:
- La duracion TOTAL debe ser {TARGET_MIN}-{TARGET_MAX} segundos ({MIN_MINUTES}-{MAX_MINUTES} minutos)
- narration por escena: extenso y fluido, como una narracion natural
- narration_text por panel: corto, sincronizado con el panel especifico
- tone debe variar segun la escena (no todo dramatico ni todo descriptivo)
- Las transiciones deben tener sentido narrativo
- NO incluyas instrucciones de audio en narration ni narration_text
- NO uses markdown ni caracteres especiales en los textos"""


def generate_script(
    selected_panels_path: str,
    interpretation_path: str,
    vision_detail_path: str,
    provider,
    output_dir: str | None = None,
) -> dict:
    """Genera guion narrativo y segmentacion temporal.

    Args:
        selected_panels_path: Ruta a selected_panels.json.
        interpretation_path: Ruta a interpretation.json.
        vision_detail_path: Ruta a vision_detail.json.
        provider: LLMProvider (DeepSeek).
        output_dir: Directorio de salida.

    Returns:
        dict con guion.txt y segments.json generados.
    """
    sel = json.loads(Path(selected_panels_path).read_text(encoding="utf-8"))
    interp = json.loads(Path(interpretation_path).read_text(encoding="utf-8"))
    vision = json.loads(Path(vision_detail_path).read_text(encoding="utf-8"))

    interp_map = {p["panel_id"]: p["interpretation"] for p in interp.get("panels", [])}
    vision_map = {p["panel_id"]: p["visual"] for p in vision.get("panels", [])}

    scenes = sel.get("selected_scenes", [])
    total_panels = sum(s.get("num_panels", 0) for s in scenes)
    chapter = sel.get("chapter", 1)

    scenes_detail_parts = []
    for scene in scenes:
        sid = scene["scene_id"]
        pids = scene.get("panel_ids", [])
        panel_lines = []
        for pid in pids:
            vis = vision_map.get(pid, {})
            interp_data = interp_map.get(pid, {})
            summary = interp_data.get("summary", "")
            category = interp_data.get("inferred_category", {}).get("value", "unknown")
            importance = interp_data.get("narrative_importance", {}).get("value", 5)
            env = vis.get("environment", {}).get("value", "?") if isinstance(vis, dict) else "?"
            action = vis.get("action", {}).get("value", "?") if isinstance(vis, dict) else "?"
            chars = vis.get("characters", []) if isinstance(vis, dict) else []
            chars_str = "; ".join(
                f"{c.get('gender', {}).get('value', '?')}({c.get('emotion', {}).get('value', '?')})"
                for c in chars
            ) or "vacio"

            panel_lines.append(
                f"    Panel {pid}: [{category}] imp={importance} | "
                f"{env} | {action} | [{chars_str}] | {summary[:150]}"
            )

        scenes_detail_parts.append(
            f"  [ESCENA {sid}] ({len(pids)} paneles):\n" + "\n".join(panel_lines)
        )

    scenes_detail = "\n".join(scenes_detail_parts)

    est_duration = int(total_panels * SECONDS_PER_PANEL)
    target_min = max(480, est_duration - TARGET_DURATION_TOLERANCE)
    target_max = min(720, est_duration + TARGET_DURATION_TOLERANCE)
    min_minutes = target_min // 60
    max_minutes = target_max // 60

    prompt = PROMPT_SCRIPT.format(
        NUM_SCENES=len(scenes),
        NUM_PANELS=total_panels,
        CHAPTER=chapter,
        SCENES_DETAIL=scenes_detail,
        TARGET_MIN=target_min,
        TARGET_MAX=target_max,
        MIN_MINUTES=min_minutes,
        MAX_MINUTES=max_minutes,
    )

    logger.info(f"[SCRIPT] Generando guion para {len(scenes)} escenas, {total_panels} paneles...")
    response = provider.describe_text(prompt)

    try:
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
        else:
            parsed = json.loads(response)
    except (json.JSONDecodeError, Exception) as exc:
        logger.error(f"[SCRIPT] Error parseando respuesta: {exc}")
        parsed = {"script": {"intro": "", "segments": [], "outro": ""}, "segmentation": []}

    script_data = parsed.get("script", {})
    segmentation = parsed.get("segmentation", [])

    script_text_parts = []
    script_text_parts.append(script_data.get("intro", ""))
    for seg in script_data.get("segments", []):
        script_text_parts.append(seg.get("narration", ""))
    script_text_parts.append(script_data.get("outro", ""))
    script_text = "\n\n".join(p for p in script_text_parts if p)

    panelfile_map = {v["panel_id"]: v.get("file", f"escena_{v['panel_id']:04d}.png") for v in vision.get("panels", [])}

    enriched_segmentation = []
    for seg in segmentation:
        pid = seg.get("panel_id", 0)
        enriched_segmentation.append({
            "panel_id": pid,
            "panel_file": panelfile_map.get(pid, f"escena_{pid:04d}.png"),
            "panel_scene": seg.get("scene_id", 0),
            "text": seg.get("narration_text", ""),
            "duration_sec": seg.get("duration_sec", 4.5),
            "transition": seg.get("transition", "cut"),
            "audio_sync": seg.get("audio_sync", 0),
        })

    out_dir = Path(output_dir) if output_dir else Path(selected_panels_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    guion_path = out_dir / "guion.txt"
    guion_path.write_text(script_text.strip(), encoding="utf-8")

    segments_output = {
        "schema_version": 2,
        "generated_by": {
            "model": provider.model if hasattr(provider, "model") else "deepseek-chat",
            "provider": "deepseek",
            "prompt_version": "2026-07-21",
            "timestamp": __import__("datetime").datetime.now().isoformat(),
        },
        "chapter": chapter,
        "total_segments": len(enriched_segmentation),
        "total_duration_sec": sum(s.get("duration_sec", 4.5) for s in enriched_segmentation),
        "segments": enriched_segmentation,
        "script": script_data,
    }

    segments_path = out_dir / "segments.json"
    segments_path.write_text(json.dumps(segments_output, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(f"[SCRIPT] Guion generado → {guion_path} ({len(script_text)} chars)")
    logger.info(f"[SCRIPT] Segmentacion → {segments_path} ({len(enriched_segmentation)} segmentos)")

    return {
        "script_path": str(guion_path),
        "segments_path": str(segments_path),
        "script": script_text,
        "segments": segments_output,
    }


if __name__ == "__main__":
    import argparse
    from llm.factory import LLMProviderFactory

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-panels", required=True)
    parser.add_argument("--interpretation", required=True)
    parser.add_argument("--vision-detail", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--text-provider", default="openrouter")
    parser.add_argument("--text-model", default="google/gemini-3.5-flash")
    args = parser.parse_args()

    provider = LLMProviderFactory.create(
        args.text_provider,
        model=args.text_model,
        timeout=300,
    )
    generate_script(
        selected_panels_path=args.selected_panels,
        interpretation_path=args.interpretation,
        vision_detail_path=args.vision_detail,
        provider=provider,
        output_dir=args.output_dir,
    )
