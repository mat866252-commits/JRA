"""Scene Grouper V1: Agrupacion de paneles en escenas.

Placeholder que asigna un scene_id por panel (1 panel = 1 escena).
Interfaz estable para futura implementacion con embeddings/clustering.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger("scene_grouper")


def group_scenes(
    vision_detail_path: str,
    output_dir: str | None = None,
) -> dict:
    """Agrupa paneles en escenas basado en analisis visual.

    Actuamente asigna un scene_id incremental (1 panel = 1 escena).
    En el futuro: clustering por similitud visual, deteccion de cambios
    de escena, embeddings, etc.

    Args:
        vision_detail_path: Ruta a vision_detail.json.
        output_dir: Directorio de salida para scenes.json.

    Returns:
        dict con scenes.json completo.
    """
    src = Path(vision_detail_path)
    if not src.is_file():
        raise RuntimeError(f"No se encontro {src}")

    data = json.loads(src.read_text(encoding="utf-8"))
    panels = data.get("panels", [])

    scenes = []
    current_scene_id = 1
    current_scene_panels = []

    for i, panel in enumerate(panels):
        current_scene_panels.append(panel["panel_id"])

        if i == len(panels) - 1:
            scenes.append({
                "scene_id": current_scene_id,
                "panel_ids": current_scene_panels,
                "start_index": i - len(current_scene_panels) + 1,
                "end_index": i,
            })
            current_scene_panels = []
            break

        vis = panel.get("visual", {})
        next_vis = panels[i + 1].get("visual", {})

        camera = vis.get("camera", {}).get("value", "")
        next_camera = next_vis.get("camera", {}).get("value", "")
        env = vis.get("environment", {}).get("value", "")
        next_env = next_vis.get("environment", {}).get("value", "")
        chars = [c.get("gender", {}).get("value") for c in vis.get("characters", [])]
        next_chars = [c.get("gender", {}).get("value") for c in next_vis.get("characters", [])]

        scene_break = (
            env != next_env
            or set(chars) != set(next_chars)
            or camera != next_camera
            or len(current_scene_panels) > 6
        )

        if scene_break and current_scene_panels:
            scenes.append({
                "scene_id": current_scene_id,
                "panel_ids": current_scene_panels,
                "start_index": i - len(current_scene_panels) + 1,
                "end_index": i,
            })
            current_scene_id += 1
            current_scene_panels = []

    if current_scene_panels:
        scenes.append({
            "scene_id": current_scene_id,
            "panel_ids": current_scene_panels,
            "start_index": len(panels) - len(current_scene_panels),
            "end_index": len(panels) - 1,
        })

    panel_map = {p["panel_id"]: p for p in panels}
    for scene in scenes:
        scene["panels"] = [panel_map.get(pid, {}) for pid in scene["panel_ids"]]

    output = {
        "schema_version": 1,
        "generated_by": {
            "module": "scene_grouper",
            "version": "v1-placeholder",
            "timestamp": __import__("datetime").datetime.now().isoformat(),
        },
        "chapter": data.get("chapter", 1),
        "total_scenes": len(scenes),
        "total_panels": len(panels),
        "scenes": scenes,
    }

    out_dir = Path(output_dir) if output_dir else src.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "scenes.json"
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"[SCENE] {len(scenes)} escenas agrupadas → {output_path}")

    return output


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--vision-detail", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    group_scenes(vision_detail_path=args.vision_detail, output_dir=args.output_dir)
