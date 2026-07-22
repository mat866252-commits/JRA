#!/usr/bin/env python3
"""Thumbnail God V1: Genera miniaturas automáticas para YouTube.

Toma el hook_panel de selected_panels.json, aplica tratamiento visual
agresivo (contraste + saturación), genera texto clickbait con Gemini,
y superpone el texto con estilo "MrBeast" para maximo CTR.

Uso:
  python thumbnail_god.py --panels-dir output/capitulo_1
  python thumbnail_god.py --panels-dir output/capitulo_1 --image mi_panel.png --text "TRAICION"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger("thumbnail_god")


def _get_font_path() -> str | None:
    """Busca fuentes gruesas (Impact, Burbank, Arial Black) en Windows."""
    candidates = [
        "C:/Windows/Fonts/impact.ttf",
        "C:/Windows/Fonts/ariblk.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Impact.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    for path in candidates:
        if Path(path).is_file():
            return path
    return None


def enhance_image(image_path: str, output_path: str) -> None:
    """Aplica mejora visual agresiva: contraste +20%, saturación +20%.

    Usa OpenCV para:
    1. Aumentar contraste con convertScaleAbs (alpha=1.2)
    2. Aumentar saturación en HSV (escala S * 1.2)
    """
    import cv2
    import numpy as np

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"No se pudo leer la imagen: {image_path}")

    # Contraste +20%
    enhanced = cv2.convertScaleAbs(img, alpha=1.2, beta=10)

    # Saturación +20%
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.2, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.05, 0, 255)
    enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Nitidez ligera
    sharpen = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) / 5.0
    enhanced = cv2.filter2D(enhanced, -1, sharpen)

    cv2.imwrite(output_path, enhanced, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    logger.info(f"[THUMBNAIL] Imagen mejorada guardada: {output_path}")


def generate_clickbait_text(panel_desc: str, panel_category: str, provider) -> str:
    """Usa Gemini para generar 1-2 palabras de clickbait impactante.

    Args:
        panel_desc: Descripción visual del panel (vision_description).
        panel_category: Categoría narrativa (accion, conversacion, etc).
        provider: Instancia de LLMProvider.

    Returns:
        Texto clickbait (max 15 chars, 1-2 palabras).
    """
    prompt = (
        "Eres un copywriter experto en miniaturas de YouTube.\n"
        "Genera 1 o 2 palabras de clickbait IMPACTANTE (maximo 15 caracteres)\n"
        "para una miniatura de video de resumen de manhwa.\n\n"
        "REGLAS:\n"
        "- Maximo 15 caracteres (incluyendo espacios)\n"
        "- 1 o 2 palabras. NADA MAS.\n"
        "- Debe generar URGENCIA o CURIOSIDAD inmediata\n"
        "- Usa mayusculas sostenidas\n"
        "- Termina con signo de exclamacion si aplica\n\n"
        "EJEMPLOS:\n"
        "  Panel de pelea -> 'PODER OCULTO'\n"
        "  Panel de traicion -> 'TRAICION!'\n"
        "  Panel dramatico -> 'SUPLICO'\n"
        "  Panel de revelacion -> 'EL SECRETO'\n"
        "  Panel de accion -> 'NO HUYAS'\n\n"
        f"CONTEXTO:\n"
        f"Categoria: {panel_category}\n"
        f"Descripcion del panel: {panel_desc[:300]}\n\n"
        "Responde SOLO con el texto del clickbait, sin comillas, sin explicaciones."
    )

    try:
        response = provider.describe_text(prompt)
        text = response.strip().strip('"\'.,!?')
        # Limita a 15 chars
        if len(text) > 15:
            text = text[:15].rsplit(" ", 1)[0] if " " in text[:15] else text[:15]
        # Asegura que termine con signo si es una palabra
        if len(text.split()) <= 2 and not text.endswith("!") and not text.endswith("?"):
            text = text.upper()
        else:
            text = text.upper()
        logger.info(f"[THUMBNAIL] Texto generado: '{text}'")
        return text
    except Exception as exc:
        logger.warning(f"[THUMBNAIL] Fallo generacion de texto: {exc}")
        return "IMPACTO"


def overlay_text(
    image_path: str, text: str, output_path: str,
    font_size: int = 120, border_size: int = 10,
) -> None:
    """Superpone texto estilo MrBeast en el centro inferior de la imagen.

    Args:
        image_path: Ruta de la imagen base (ya mejorada).
        text: Texto a superponer.
        output_path: Ruta de salida.
        font_size: Tamaño de fuente (default 120).
        border_size: Grosor del borde negro (default 10).
    """
    from PIL import Image, ImageDraw, ImageFont

    img = Image.open(image_path).convert("RGBA")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    font_path = _get_font_path()
    try:
        font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
    except (IOError, OSError):
        font = ImageFont.load_default()

    # Bounding box del texto
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Posicion: centrado, a 1/3 desde abajo
    x = (w - text_w) // 2
    y = (h * 3) // 4 - text_h // 2

    # Dibuja borde negro (OutlineStyle=10px)
    for dx in range(-border_size, border_size + 1):
        for dy in range(-border_size, border_size + 1):
            if dx * dx + dy * dy <= border_size * border_size:
                draw.text((x + dx, y + dy), text, font=font, fill="black")

    # Texto principal amarillo
    draw.text((x, y), text, font=font, fill="#FFD700")

    img.convert("RGB").save(output_path, "PNG")
    logger.info(f"[THUMBNAIL] Miniatura final: {output_path}")


def generate_thumbnail(
    panels_dir: str,
    output_path: str | None = None,
    provider=None,
    image_override: str | None = None,
    text_override: str | None = None,
) -> str:
    """Pipeline completo de generacion de miniatura.

    Args:
        panels_dir: Directorio con selected_panels.json y paneles.
        output_path: Ruta de salida (default: panels_dir/thumbnail_final.png).
        provider: LLMProvider para generar texto clickbait.
        image_override: Ruta directa a imagen (salta selected_panels.json).
        text_override: Texto directo (salta Gemini).

    Returns:
        Ruta de la miniatura generada.
    """
    # Determina imagen base
    if image_override:
        panel_path = Path(image_override)
        if not panel_path.is_file():
            raise FileNotFoundError(f"Imagen no encontrada: {image_override}")
    else:
        manifest_path = Path(panels_dir) / "selected_panels.json"
        if not manifest_path.is_file():
            manifest_path = Path(panels_dir) / "panels.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"No se encontro selected_panels.json ni panels.json en {panels_dir}")

        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        hook = data.get("hook_panel")
        panels = data.get("panels", data.get("selected_panels", []))

        if hook:
            panel_file = hook.get("file", f"escena_{hook['scene']:04d}.png")
        elif panels:
            # Toma el de mayor puntuacion
            best = max(panels, key=lambda p: p.get("stars", 1) * 10 + p.get("score_raw", 0))
            panel_file = best.get("file", f"escena_{best['scene']:04d}.png")
        else:
            raise ValueError("No hay hook_panel ni paneles disponibles")

        panel_path = Path(panels_dir) / panel_file
        if not panel_path.is_file():
            # Fallback: busca cualquier png
            pngs = sorted(Path(panels_dir).glob("escena_*.png"))
            if not pngs:
                raise FileNotFoundError(f"No se encontro imagen: {panel_path}")
            panel_path = pngs[0]

    # Determina texto clickbait
    if text_override:
        clickbait_text = text_override
    elif provider:
        if not image_override and hook:
            desc = hook.get("vision_description", hook.get("vision_error", ""))
            cat = hook.get("category", "accion")
        elif not image_override and panels:
            desc = panels[0].get("vision_description", "")
            cat = panels[0].get("category", "accion")
        else:
            desc = ""
            cat = "accion"
        clickbait_text = generate_clickbait_text(desc, cat, provider)
    else:
        clickbait_text = "IMPACTO"

    os.makedirs(os.path.dirname(output_path) if output_path else panels_dir, exist_ok=True)
    final_output = output_path or str(Path(panels_dir) / "thumbnail_final.png")

    enhanced_path = Path(final_output).with_suffix(".enhanced.png")
    enhance_image(str(panel_path), str(enhanced_path))
    overlay_text(str(enhanced_path), clickbait_text, final_output)

    # Limpia temporal
    try:
        enhanced_path.unlink()
    except OSError:
        pass

    size_kb = Path(final_output).stat().st_size / 1024
    logger.info(f"[THUMBNAIL] Miniatura: {final_output} ({size_kb:.0f} KB, texto='{clickbait_text}')")
    return final_output


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Thumbnail God: miniaturas automaticas para YouTube")
    parser.add_argument("--panels-dir", required=True, help="Directorio con selected_panels.json y paneles")
    parser.add_argument("--output", default=None, help="Ruta de salida (default: panels_dir/thumbnail_final.png)")
    parser.add_argument("--image", default=None, help="Override: ruta directa a imagen")
    parser.add_argument("--text", default=None, help="Override: texto clickbait directo")
    parser.add_argument("--text-provider", default="gemini", help="Proveedor LLM para texto (default: gemini)")
    parser.add_argument("--text-model", default="gemini-flash-lite-latest", help="Modelo de texto")
    parser.add_argument("--text-timeout", type=int, default=30)
    parser.add_argument("--font-size", type=int, default=120, help="Tamaño de fuente (default: 120)")
    parser.add_argument("--border-size", type=int, default=10, help="Grosor del borde (default: 10)")

    args = parser.parse_args()

    provider = None
    if not args.text:
        from llm.factory import LLMProviderFactory
        provider = LLMProviderFactory.create(
            args.text_provider,
            model=args.text_model,
            timeout=args.text_timeout,
        )

    try:
        output = generate_thumbnail(
            panels_dir=args.panels_dir,
            output_path=args.output,
            provider=provider,
            image_override=args.image,
            text_override=args.text,
        )
        print(f"[OK] Miniatura generada: {output}")
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        sys.exit(f"[ERROR] {exc}")


if __name__ == "__main__":
    main()
