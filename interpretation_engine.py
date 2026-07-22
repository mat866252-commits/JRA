"""Interpretation Engine V1: Razonamiento narrativo sobre datos visuales.

Lee vision_detail.json + scenes.json, usa DeepSeek para generar:
- resumen narrativo por panel
- importancia narrativa (1-10) con razon
- categoria (vocabulario cerrado)
- scores emocionales
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger("interpretation_engine")

PROMPT_INTERPRET = """Eres un ANALISTA NARRATIVO de manhwa. Recibes informacion visual detallada de paneles y debes interpretar su significado narrativo.

Para cada panel, responde con JSON EXACTO:

{
  "panels": [
    {
      "panel_id": 143,
      "interpretation": {
        "summary": "resumen narrativo de lo que ocurre en el panel (1-2 frases)",
        "narrative_importance": {
          "value": 1-10,
          "reason": "explicacion breve de por que este panel importa para la historia"
        },
        "inferred_category": {
          "value": "CATEGORIA",
          "confidence": 0.0-1.0
        },
        "scores": {
          "emotional_intensity": {"value": 1-10, "confidence": 0.0-1.0},
          "action": {"value": 1-10, "confidence": 0.0-1.0},
          "new_information": {"value": 1-10, "confidence": 0.0-1.0},
          "apparent_relevance": {"value": 1-10, "confidence": 0.0-1.0}
        }
      }
    }
  ]
}

CATEGORIAS PERMITIDAS (elige UNA y solo UNA):
- dialogue: conversaciones entre personajes
- action: movimiento, peleas, persecuciones
- reveal: revelaciones, sorpresas, descubrimientos
- reaction: reacciones emocionales a eventos
- travel: desplazamiento entre lugares
- combat: combate cuerpo a cuerpo o armado
- flashback: recuerdos, escenas del pasado
- transition: paneles de transicion, cambios de escena
- worldbuilding: establecimiento del mundo, escenarios amplios
- other: cualquier otra categoria no contemplada

REGLAS:
- narrative_importance debe ser 1-10, donde 10 = panel CRITICO para entender la historia
- El "reason" en importance debe ser especifico: "El protagonista descubre X" no "Es importante"
- new_information debe ser ALTO si el panel introduce personajes, lugares o giros nuevos
- Usa confidence para indicar que tan seguro estas de tu interpretacion
- Si la descripcion visual tiene poca calidad (vision_quality baja), baja la confianza
- NO inventes informacion que no este en la descripcion visual"""


def interpret_panels(
    vision_detail_path: str,
    scenes_path: str,
    provider,
    output_dir: str | None = None,
) -> dict:
    """Genera interpretacion narrativa para cada panel.

    Args:
        vision_detail_path: Ruta a vision_detail.json.
        scenes_path: Ruta a scenes.json.
        provider: LLMProvider (DeepSeek).
        output_dir: Directorio de salida para interpretation.json.

    Returns:
        dict con interpretation.json completo.
    """
    vision = json.loads(Path(vision_detail_path).read_text(encoding="utf-8"))
    scenes = json.loads(Path(scenes_path).read_text(encoding="utf-8"))

    panels = vision.get("panels", [])
    panel_map = {p["panel_id"]: p for p in panels}

    context_parts = []
    for scene in scenes.get("scenes", []):
        context_parts.append(f"[ESCENA {scene['scene_id']}]")
        for pid in scene.get("panel_ids", []):
            panel = panel_map.get(pid)
            if not panel:
                continue
            vis = panel.get("visual", {})
            chars_desc = "; ".join(
                f"{c.get('gender', {}).get('value', '?')}({c.get('emotion', {}).get('value', '?')})"
                for c in vis.get("characters", [])
            ) or "sin personajes"
            context_parts.append(
                f"  Panel {pid}: {vis.get('environment', {}).get('value', '?')} | "
                f"{vis.get('action', {}).get('value', '?')} | "
                f"personajes: {chars_desc} | "
                f"camara: {vis.get('camera', {}).get('value', '?')} | "
                f"dialogo: {vis.get('dialogue', {}).get('value', 'ninguno')} | "
                f"texto: {vis.get('visible_text', {}).get('value', 'ninguno')} | "
                f"calidad: {panel.get('vision_quality', 'N/A')}"
            )

    context = "\n".join(context_parts)

    prompt = (
        f"{PROMPT_INTERPRET}\n\n"
        f"Capitulo {vision.get('chapter', 1)} - "
        f"{len(panels)} paneles en {len(scenes.get('scenes', []))} escenas\n\n"
        f"DATOS VISUALES:\n{context}\n\n"
        "Responde SOLO el JSON con las interpretaciones de TODOS los paneles."
    )

    logger.info(f"[INTERPRET] Enviando {len(panels)} paneles a DeepSeek para interpretacion...")
    try:
        response = provider.describe_text(prompt)
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
        else:
            parsed = json.loads(response)
        interpretations = parsed.get("panels", [])
    except (json.JSONDecodeError, Exception) as exc:
        logger.error(f"[INTERPRET] Error: {exc}")
        interpretations = []

    interp_map = {i.get("panel_id"): i.get("interpretation", {}) for i in interpretations}

    full_results = []
    for panel in panels:
        pid = panel["panel_id"]
        interp = interp_map.get(pid, {})
        full_results.append({
            "panel_id": pid,
            "interpretation": {
                "summary": interp.get("summary", ""),
                "narrative_importance": interp.get("narrative_importance", {"value": 5, "reason": ""}),
                "inferred_category": interp.get("inferred_category", {"value": "other", "confidence": 0.0}),
                "scores": interp.get("scores", {
                    "emotional_intensity": {"value": 5, "confidence": 0.0},
                    "action": {"value": 5, "confidence": 0.0},
                    "new_information": {"value": 5, "confidence": 0.0},
                    "apparent_relevance": {"value": 5, "confidence": 0.0},
                }),
            }
        })

    output = {
        "schema_version": 1,
        "generated_by": {
            "model": provider.model if hasattr(provider, "model") else "deepseek-chat",
            "provider": "deepseek",
            "prompt_version": "2026-07-21",
            "timestamp": __import__("datetime").datetime.now().isoformat(),
        },
        "chapter": vision.get("chapter", 1),
        "panels": full_results,
    }

    out_dir = Path(output_dir) if output_dir else Path(vision_detail_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "interpretation.json"
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"[INTERPRET] {len(full_results)} paneles interpretados → {output_path}")

    return output


if __name__ == "__main__":
    import argparse
    from llm.factory import LLMProviderFactory

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--vision-detail", required=True)
    parser.add_argument("--scenes", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--text-provider", default="openrouter")
    parser.add_argument("--text-model", default="google/gemini-3.5-flash")
    args = parser.parse_args()

    provider = LLMProviderFactory.create(
        args.text_provider,
        model=args.text_model,
        timeout=180,
    )
    interpret_panels(
        vision_detail_path=args.vision_detail,
        scenes_path=args.scenes,
        provider=provider,
        output_dir=args.output_dir,
    )
