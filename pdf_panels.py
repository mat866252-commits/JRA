#!/usr/bin/env python3
"""Recorta viñetas desde PDF con fusión entre páginas, scoring narrativo y validaciones automáticas.

Reemplaza a panel_crop.py en el pipeline v3. Procesa directamente PDFs con PyMuPDF,
detecta gutters con OpenCV, fusiona paneles partidos entre páginas mediante puntuación
de confianza multicriterio, asigna puntuación narrativa a cada viñeta y descarta las
de baja calidad automáticamente.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from html import escape
import hashlib
import json
import os
import sys
from pathlib import Path

import cv2
import fitz
import numpy as np

from vision import VisionError, create_vision


def _write_json(path: str, data: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def pdf_to_images(pdf_path: str, dpi: int = 200) -> list[np.ndarray]:
    """Convierte cada página del PDF a imagen numpy (BGR) sin archivos intermedios.
    
    Procesa en batches de 10 páginas para evitar consumo excesivo de RAM en CPUs sin GPU."""
    import gc
    images = []
    doc = None
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        batch_size = 10
        for batch_start in range(0, total_pages, batch_size):
            batch_end = min(batch_start + batch_size, total_pages)
            for page_num in range(batch_start, batch_end):
                try:
                    page = doc[page_num]
                    mat = page.get_pixmap(dpi=dpi)
                    rgb = np.frombuffer(mat.samples, dtype=np.uint8).reshape(mat.height, mat.width, mat.n)
                    if mat.n == 4:
                        rgb = rgb[:, :, :3]
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    images.append(bgr)
                except Exception as exc:
                    print(f"[AVISO] Página {page_num + 1} corrupta, saltando: {exc}")
                    images.append(np.zeros((100, 100, 3), dtype=np.uint8))
    except Exception as exc:
        if doc:
            doc.close()
        raise RuntimeError(f"Error abriendo PDF {pdf_path}: {exc}") from exc
    if doc:
        doc.close()
    gc.collect()
    return images


def find_panels_by_content_distribution(gray: np.ndarray) -> list[tuple[int, int]]:
    """Detecta viñetas analizando la distribución de contenido (bordes Canny).

    A diferencia de find_gutter_splits (basado en filas blancas/negras),
    este método:
    1. Aplica Canny edge detection
    2. Calcula proyección horizontal de bordes
    3. Suaviza con GaussianBlur
    4. Encuentra valles (gutters) = filas sin contenido
    5. Entre valles = paneles

    Esto es mucho más robusto: un panel con arte oscuro ya no se confunde
    con un gutter, y los gutters reales (sin contenido) se detectan aunque
    no sean blancos puros.
    """
    edges = cv2.Canny(gray, 50, 150)
    horz_proj = edges.mean(axis=1)
    horz_proj = cv2.GaussianBlur(horz_proj, (1, 9), 0).flatten()

    threshold = horz_proj.mean() * 0.35
    blank_rows = horz_proj < threshold

    ranges, start = [], None
    for i, blank in enumerate(blank_rows):
        if blank and start is None:
            start = i
        elif not blank and start is not None:
            if i - start >= 12:
                ranges.append((start, i))
            start = None
    if start is not None and len(blank_rows) - start >= 12:
        ranges.append((start, len(blank_rows)))

    if not ranges:
        return []

    boundaries = [0]
    for gs, ge in ranges:
        mid = (gs + ge) // 2
        boundaries.append(mid)
    boundaries.append(gray.shape[0])

    panels = []
    for top, bottom in zip(boundaries, boundaries[1:]):
        h = bottom - top
        if h >= 300:
            panels.append((top, bottom))

    return panels


def verify_panel_has_content(panel: np.ndarray) -> bool:
    """Verifica que un panel tenga contenido narrativo real usando múltiples heurísticas."""
    h, w = panel.shape[:2]
    if h < 300 or w < 100:
        return False

    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)

    blank = ((gray > 240) | (gray < 10)).mean()
    if blank > 0.50:
        return False

    edges = cv2.Canny(gray, 50, 150)
    edge_density = edges.mean() / 255.0
    if edge_density < 0.02:
        return False

    faces = _detect_faces(panel)
    if faces == 0:
        laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian < 0.5 and edge_density < 0.04:
            return False

    return True


def detect_panels_from_page(page_image: np.ndarray) -> list[np.ndarray]:
    """Pipeline completo de detección de viñetas para una página.

    1. Convierte a escala de grises
    2. Detecta contornos con Canny
    3. Encuentra regiones de contenido mediante proyección horizontal
    4. Verifica cada candidato con verify_panel_has_content
    5. Devuelve solo paneles con contenido narrativo real
    """
    gray = cv2.cvtColor(page_image, cv2.COLOR_BGR2GRAY)
    candidates = find_panels_by_content_distribution(gray)

    panels = []
    for top, bottom in candidates:
        panel = page_image[top:bottom, :]
        if verify_panel_has_content(panel):
            panels.append(panel)

    return panels


def is_valid_panel(panel: np.ndarray, min_height: int, max_blank_ratio: float) -> bool:
    """Descarta paneles demasiado pequeños o mayormente en blanco/negro."""
    if panel.shape[0] < min_height or panel.shape[1] < 100:
        return False
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    blank = ((gray > 240) | (gray < 15)).mean()
    return blank <= max_blank_ratio


def _heuristic_quality_check(heuristic_panels: list[np.ndarray],
                              page_image: np.ndarray) -> tuple[bool, str]:
    """Evalúa si el resultado heurístico es fiable o necesita IA.
    
    Returns: (needs_ai: bool, reason: str)
    """
    if not heuristic_panels:
        return True, "no_panels_detected"

    if len(heuristic_panels) > 8:
        return True, f"too_many_panels({len(heuristic_panels)})"
    
    # Paneles muy pequeños -> falsos positivos
    tiny = sum(1 for p in heuristic_panels if p.shape[0] < 150)
    if tiny > len(heuristic_panels) * 0.3:
        return True, f"tiny_panels({tiny}/{len(heuristic_panels)})"
    
    # Cobertura de página muy baja -> mal detection
    page_area = page_image.shape[0] * page_image.shape[1]
    panel_area = sum(p.shape[0] * p.shape[1] for p in heuristic_panels)
    if panel_area / page_area < 0.3:
        return True, f"low_coverage({panel_area/page_area:.2f})"
    
    return False, "ok"


def _detect_panels_with_ai(page_image: np.ndarray, page_num: int,
                            provider) -> list[np.ndarray]:
    """Usa Gemini para detectar viñetas reales en páginas problemáticas.
    
    Solo se llama cuando la heurística falla (~20-30% de páginas).
    Ahorra tokens al no enviar páginas con detección clara.
    """
    import tempfile, json as _json
    
    rgb = cv2.cvtColor(page_image, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
    
    from PIL import Image as PILImage
    import io
    
    pil_img = PILImage.fromarray(rgb)
    pil_img.thumbnail((1280, 1280))
    buf = io.BytesIO()
    pil_img.save(buf, format='JPEG', quality=88)
    
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"_ai_page_{page_num:04d}.jpg")
    cv2.imwrite(temp_path, page_image, [cv2.IMWRITE_JPEG_QUALITY, 88])
    
    prompt = (
        "Eres un extractor de viñetas de manhwa. Analiza esta página y devuelve "
        "SOLO las viñetas con CONTENIDO NARRATIVO REAL.\n\n"
        "REGLAS:\n"
        "- Ignora espacios en blanco, bordes, viñetas vacías o decorativas\n"
        "- Ignora gutters (franjas blancas/negras entre viñetas)\n"
        "- Si una viñeta está partida entre páginas: fusion_candidate=true\n"
        "- Devuelve coordenadas en PÍXELES de la imagen ORIGINAL\n\n"
        f"TAMAÑO ORIGINAL: {page_image.shape[1]}x{page_image.shape[0]}px\n\n"
        'SOLO JSON: {"panels": [{"bbox": [x1,y1,x2,y2], '
        '"has_content": true, "fusion_candidate": false}]}'
    )
    
    try:
        response = provider.describe(temp_path, prompt)
        parsed = _json.loads(response)
        panels = []
        for p in parsed.get("panels", []):
            x1, y1, x2, y2 = p["bbox"]
            if x2 - x1 > 50 and y2 - y1 > 50:
                panels.append(page_image[y1:y2, x1:x2])
        return panels
    except Exception as exc:
        print(f"[AVISO] IA falló en página {page_num}: {exc}")
        return []
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


def detect_panels_from_page_hybrid(page_image: np.ndarray, page_num: int,
                                     provider=None) -> tuple[list[np.ndarray], str]:
    """Detección de viñetas basada en contornos.

    Usa detección por distribución de contenido (Canny edges + proyección horizontal)
    en lugar del antiguo método de gutter por filas blancas.

    Si el proveedor de IA está disponible y la heurística encuentra
    resultados dudosos, usa Gemini como fallback para refinar.

    Returns: (panels: list[np.ndarray], status: str)
    """
    panels = detect_panels_from_page(page_image)

    if not panels:
        try:
            ai_panels = _detect_panels_with_ai(page_image, page_num, provider)
            if ai_panels:
                return ai_panels, "ai_fallback"
        except Exception:
            pass
        return [], "no_panels"

    needs_ai, reason = _heuristic_quality_check(panels, page_image)
    if needs_ai and provider:
        try:
            ai_panels = _detect_panels_with_ai(page_image, page_num, provider)
            if ai_panels:
                return ai_panels, f"ai_corrected({reason})"
        except Exception:
            pass

    return panels, "content_based"


def panel_metrics(panel: np.ndarray) -> dict:
    """Métricas base: blank_ratio, sharpness, quality_score."""
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    blank = float(((gray > 240) | (gray < 15)).mean())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    quality = max(0.0, min(1.0, (1 - blank) * min(1, sharpness / 120.0) * min(1, panel.shape[0] / 400)))
    return {"blank_ratio": round(blank, 4), "sharpness": round(sharpness, 2), "quality_score": round(quality, 3)}


def _detect_faces(panel: np.ndarray) -> int:
    """Detección de rostros con haarcascade para puntuar primeros planos.

    Si haarcascade no está disponible (opencv-python-headless), devuelve 0."""
    try:
        gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        cascade = cv2.CascadeClassifier(cascade_path)
        faces = cascade.detectMultiScale(gray, 1.1, 2, minSize=(40, 40))
        return len(faces)
    except (AttributeError, cv2.error):
        return 0


def _edge_density(panel: np.ndarray) -> float:
    """Densidad de bordes Canny — alta en acción, baja en fondos vacíos."""
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return float(edges.mean() / 255.0)


def _detect_speech_bubbles(panel: np.ndarray) -> float:
    """Estima proporción de área ocupada por bocadillos (zonas blancas estructuradas)."""
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bubble_area = 0.0
    total_area = panel.shape[0] * panel.shape[1]
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / max(h, 1)
        if 100 < area < total_area * 0.6 and 0.3 < aspect < 5.0 and h > 15 and w > 20:
            bubble_area += area
    return min(1.0, bubble_area / total_area)


def _content_cut_at_edge(panel: np.ndarray, edge: str = "bottom") -> bool:
    """Detecta si hay figuras/contornos cortados en el borde del panel."""
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_inv = cv2.bitwise_not(thresh)
    contours, _ = cv2.findContours(thresh_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h = panel.shape[0]
    margin = 8
    for cnt in contours:
        if cv2.contourArea(cnt) < 500:
            continue
        x, y, w_cnt, h_cnt = cv2.boundingRect(cnt)
        if edge == "bottom" and (y + h_cnt) >= h - margin:
            return True
        if edge == "top" and y <= margin:
            return True
        if edge == "left" and x <= margin:
            return True
        if edge == "right" and (x + w_cnt) >= panel.shape[1] - margin:
            return True
    return False


def panel_narrative_score(panel: np.ndarray) -> dict:
    """Puntuación narrativa de 1★ a 5★ basada en heurísticas visuales.

    Criterios:
      5★ — acción, expresiones, revelaciones, primeros planos
      3★ — conversaciones
      1★ — texto puro, fondos vacíos, paneles poco relevantes
    """
    faces = _detect_faces(panel)
    edges = _edge_density(panel)
    bubbles = _detect_speech_bubbles(panel)
    height = panel.shape[0]
    width = panel.shape[1]
    area = height * width

    score = 1.0
    category = "fondo/texto"
    reasons = []

    if faces >= 1:
        score += 1.5
        category = "primer_plano"
        reasons.append("caras_detectadas")

    if edges > 0.12:
        score += 1.2
        category = "accion"
        reasons.append("alta densidad bordes")

    if area > 400 * 400:
        score += 0.6
        reasons.append("gran area")

    if height > width * 1.5:
        score += 0.4
        reasons.append("panel vertical")

    if faces >= 2 and edges > 0.10:
        score += 1.0
        category = "accion_grupal"
        reasons.append("multiple caras con accion")

    if 0.05 < bubbles < 0.40 and edges < 0.10 and faces >= 1:
        score += 0.4
        if category in ("fondo/texto", "primer_plano"):
            category = "conversacion"
        reasons.append("bocadillos moderados")

    if 0.05 < bubbles < 0.50 and faces == 0:
        score += 0.2
        if category == "fondo/texto" and edges < 0.08:
            category = "conversacion"
        reasons.append("solo bocadillos")

    if bubbles > 0.50:
        score -= 0.8
        category = "texto_puro"
        reasons.append("exceso bocadillos")

    if edges < 0.03 and bubbles < 0.02 and faces == 0:
        score -= 1.0
        category = "fondo_vacio"
        reasons.append("sin contenido relevante")

    stars = max(1, min(5, round(score)))
    return {
        "stars": stars,
        "category": category,
        "score_raw": round(score, 2),
        "faces": faces,
        "edge_density": round(edges, 4),
        "bubble_ratio": round(bubbles, 4),
        "reasons": reasons,
    }


def trim_margins(panel: np.ndarray, threshold: int = 245, padding: int = 4) -> np.ndarray:
    """Recorta márgenes blancos verticales."""
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    content = np.where((gray < threshold).any(axis=1))[0]
    if not len(content):
        return panel
    top = max(0, content[0] - padding)
    bottom = min(panel.shape[0], content[-1] + padding + 1)
    return panel[top:bottom, :]


# ─── Fusión entre páginas ────────────────────────────────────────────────────

def _edge_continuity(panel_a_bottom: np.ndarray, panel_b_top: np.ndarray) -> float:
    """Compara bordes Canny entre el final de una página y el inicio de la siguiente."""
    if panel_a_bottom.shape != panel_b_top.shape:
        min_h = min(panel_a_bottom.shape[0], panel_b_top.shape[0])
        min_w = min(panel_a_bottom.shape[1], panel_b_top.shape[1])
        panel_a_bottom = panel_a_bottom[:min_h, :min_w]
        panel_b_top = panel_b_top[:min_h, :min_w]
    gray_a = cv2.cvtColor(panel_a_bottom, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(panel_b_top, cv2.COLOR_BGR2GRAY)
    edges_a = cv2.Canny(gray_a, 50, 150)
    edges_b = cv2.Canny(gray_b, 50, 150)
    intersection = float(np.sum((edges_a > 0) & (edges_b > 0)))
    union = float(np.sum((edges_a > 0) | (edges_b > 0)))
    if union == 0:
        return 0.0
    return intersection / union


def _color_continuity(panel_a_bottom: np.ndarray, panel_b_top: np.ndarray) -> float:
    """Compara diferencia media de color absoluto entre franjas."""
    if panel_a_bottom.shape != panel_b_top.shape:
        min_h = min(panel_a_bottom.shape[0], panel_b_top.shape[0])
        min_w = min(panel_a_bottom.shape[1], panel_b_top.shape[1])
        panel_a_bottom = panel_a_bottom[:min_h, :min_w]
        panel_b_top = panel_b_top[:min_h, :min_w]
    diff = float(np.abs(panel_a_bottom.astype(np.float32) - panel_b_top.astype(np.float32)).mean())
    return max(0.0, 1.0 - diff / 255.0)


def _content_continuity(panel_a: np.ndarray, panel_b: np.ndarray) -> float:
    """Detecta figuras cortadas en el borde inferior de A y superior de B."""
    cut_a = _content_cut_at_edge(panel_a, "bottom")
    cut_b = _content_cut_at_edge(panel_b, "top")
    if cut_a and cut_b:
        return 0.7
    if cut_a or cut_b:
        return 0.4
    return 0.0


def fusion_confidence(panel_a: np.ndarray, panel_b: np.ndarray,
                      text_cut_a: float = 0.0, text_cut_b: float = 0.0,
                      strip_height: int = 20) -> dict:
    """Calcula puntuación de confianza para fusión de dos paneles entre páginas consecutivas.

    Criterios:
      1. Panel A toca borde inferior Y panel B empieza desde borde superior (imprescindible).
      2. Continuidad visual (Canny edge matching) — la señal principal.
      3. Continuidad de color (diferencia media).
      4. Contenido cortado (figuras/contornos en bordes opuestos).
      5. Forma del panel (sin bordes de viñeta visibles) — bonus.
      6. Texto/bocadillos cortados (requiere OCR externo) — apoyo.
    """
    ha = panel_a.shape[0]
    hb = panel_b.shape[0]
    strip = min(strip_height, ha, hb)
    strip_a = panel_a[ha - strip:ha, :]
    strip_b = panel_b[0:strip, :]

    edge_cont = _edge_continuity(strip_a, strip_b)
    color_cont = _color_continuity(strip_a, strip_b)
    content_cont = _content_continuity(panel_a, panel_b)

    touches_bottom = _content_cut_at_edge(panel_a, "bottom")
    touches_top = _content_cut_at_edge(panel_b, "top")

    without_border_a = not _has_panel_border(panel_a, "bottom")
    without_border_b = not _has_panel_border(panel_b, "top")
    shape_bonus = 0.15 if (without_border_a and without_border_b) else 0.05 if (without_border_a or without_border_b) else 0.0

    text_signal = max(text_cut_a, text_cut_b) * 0.15

    weights = {
        "edge": 0.45,
        "color": 0.15,
        "content": 0.20,
        "text": 0.05,
        "shape": 0.15,
    }

    score = (
        edge_cont * weights["edge"]
        + color_cont * weights["color"]
        + content_cont * weights["content"]
        + text_signal * weights["text"]
        + shape_bonus * weights["shape"]
    )

    return {
        "confidence": round(score, 4),
        "touches_bottom": touches_bottom,
        "touches_top": touches_top,
        "edge_continuity": round(edge_cont, 4),
        "color_continuity": round(color_cont, 4),
        "content_continuity": round(content_cont, 4),
        "without_border_a": without_border_a,
        "without_border_b": without_border_b,
    }


def _has_panel_border(panel: np.ndarray, edge: str = "bottom") -> bool:
    """Detecta si el panel tiene un borde de viñeta visible (línea negra gruesa)."""
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    if edge == "bottom":
        strip = gray[-8:, :]
    elif edge == "top":
        strip = gray[:8, :]
    elif edge == "left":
        strip = gray[:, :8]
    else:
        strip = gray[:, -8:]
    dark_pct = (strip < 30).mean()
    return dark_pct > 0.55


# ─── Validaciones automáticas (Prioridad 6) ──────────────────────────────────

def validate_panels(manifest: list[dict], min_width: int = 80, min_height: int = 100,
                    max_text_ratio: float = 0.60) -> list[dict]:
    """Ejecuta 9 comprobaciones automáticas sobre el manifiesto de paneles."""
    warnings = []
    seen_hashes = {}

    for entry in manifest:
        scene = entry["scene"]
        w, h = entry["width"], entry["height"]

        if w < min_width or h < min_height:
            warnings.append({"scene": scene, "issue": "panel_demasiado_pequeno", "width": w, "height": h})

        if entry.get("blank_ratio", 0) > 0.75:
            warnings.append({"scene": scene, "issue": "panel_vacio", "blank_ratio": entry["blank_ratio"]})

        if entry.get("sharpness", 999) < 0.15:
            warnings.append({"scene": scene, "issue": "posible_corrupto", "sharpness": entry["sharpness"]})

        sha = entry.get("sha256", "")
        if sha and sha in seen_hashes:
            warnings.append({"scene": scene, "issue": "duplicado", "duplicate_of": seen_hashes[sha]})
        elif sha:
            seen_hashes[sha] = scene

        bubble = entry.get("bubble_ratio", 0)
        if bubble > max_text_ratio:
            warnings.append({"scene": scene, "issue": "exceso_texto", "bubble_ratio": bubble})

        stars = entry.get("stars", 3)
        if stars <= 2:
            warnings.append({"scene": scene, "issue": "puntuacion_baja", "stars": stars})

        if entry.get("fusion_confidence") is not None and 0.5 <= entry["fusion_confidence"] < 0.95:
            warnings.append({"scene": scene, "issue": "fusion_confianza_media",
                             "confidence": entry["fusion_confidence"]})

    for i in range(len(manifest) - 1):
        h1 = manifest[i]["height"]
        h2 = manifest[i + 1]["height"]
        area1 = manifest[i]["width"] * h1
        area2 = manifest[i + 1]["width"] * h2
        if min(area1, area2) > 0 and max(area1 / max(area2, 1), area2 / max(area1, 1)) > 8:
            warnings.append({"scene": manifest[i]["scene"],
                             "issue": "cambio_brusco_tamano",
                             "adjacent": manifest[i + 1]["scene"]})

    return warnings


def _ocr_text_for_panel(panel: np.ndarray, reader) -> str:
    """Extrae texto del panel con EasyOCR, orientado a bocadillos."""
    try:
        gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        results = reader.readtext(thresh, detail=0, paragraph=True)
        return " ".join(results).strip()
    except Exception:
        return ""


# ─── Pipeline principal ──────────────────────────────────────────────────────

def crop_panels_from_pdf(pdf_path: str, args, review_dir: str | None = None) -> tuple[list[dict], list[dict], list[dict]]:
    """Procesa PDF completo: extrae páginas, recorta viñetas, fusiona entre páginas, puntúa y valida."""
    pages = pdf_to_images(pdf_path, dpi=args.dpi)
    all_panels: list[np.ndarray] = []
    panel_sources: list[dict] = []
    diagnostics: list[dict] = []
    discarded_total = 0
    uncertain: list[str] = []
    discarded_set: set[tuple[int, int]] = set()

    provider = None
    if args.vision_provider != "disabled":
        try:
            provider = create_vision(args.vision_provider, args.vision_model, args.ollama_host,
                                      args.vision_timeout, 2, args.vision_max_side)
        except VisionError:
            provider = None

    for page_idx, image in enumerate(pages):
        page_label = f"p{page_idx + 1:04d}"

        if args.detection_mode == "ai" and provider:
            result = _detect_panels_with_ai(image, page_idx, provider)
            page_panels, status = result or [], "ai_only"
        else:
            page_panels, status = detect_panels_from_page_hybrid(image, page_idx, provider)

        page_discarded = 0
        if page_panels:
            valid_panels = []
            for pos, panel in enumerate(page_panels):
                if is_valid_panel(panel, args.min_height, args.max_blank_ratio):
                    valid_panels.append(panel)
                    panel_sources.append({"source_page": page_idx + 1, "source_pos": pos,
                                          "width": int(panel.shape[1]), "height": int(panel.shape[0])})
                else:
                    discard_key = (page_idx, pos)
                    if discard_key not in discarded_set:
                        discarded_set.add(discard_key)
                        discarded_total += 1
                        page_discarded += 1
                    if review_dir:
                        cv2.imwrite(os.path.join(review_dir, f"{page_label}_{pos:02d}.png"), panel)
            page_panels = valid_panels
        else:
            page_discarded = 0

        if not page_panels and args.on_uncertain == "full":
            all_panels.append(image)
            panel_sources.append({"source_page": page_idx + 1, "source_pos": -1, "width": int(image.shape[1]),
                                  "height": int(image.shape[0])})
            diagnostics.append({"source": page_label, "status": f"fallback_full_image({status})", "gutters": 0,
                                "saved": 1, "discarded": page_discarded})
        elif page_panels:
            all_panels.extend(page_panels)
            diagnostics.append({"source": page_label, "status": status, "gutters": 0,
                                "saved": len(page_panels), "discarded": page_discarded})
        else:
            diagnostics.append({"source": page_label, "status": f"{status}_empty", "gutters": 0,
                                "saved": 0, "discarded": page_discarded})
            uncertain.append(page_label)
            discarded_total += page_discarded

    fusion_log: list[dict] = []
    merged_manifest: list[dict] = []
    scene_counter: int = 0
    hashes: dict[str, int] = {}
    idx = 0
    fusion_id = 0
    while idx < len(all_panels):
        panel_a = all_panels[idx]
        src_a = panel_sources[idx]

        if idx + 1 < len(all_panels) and src_a["source_page"] != panel_sources[idx + 1]["source_page"]:
            panel_b = all_panels[idx + 1]
            src_b = panel_sources[idx + 1]
            result = fusion_confidence(panel_a, panel_b, strip_height=args.fusion_strip)

            fusion_entry = {
                "fusion_id": fusion_id,
                "page_a": src_a["source_page"],
                "pos_a": src_a.get("source_pos", -1),
                "page_b": src_b["source_page"],
                "pos_b": src_b.get("source_pos", -1),
                **result,
            }

            if result["confidence"] >= args.fusion_min_confidence:
                fused_panel = np.vstack([panel_a, panel_b])
                all_panels[idx] = fused_panel
                all_panels.pop(idx + 1)
                panel_sources[idx] = {**src_a, "fusion_page_b": src_b["source_page"],
                                      "fusion_id": fusion_id}
                panel_sources.pop(idx + 1)
                fusion_entry["action"] = "auto"
                fusion_entry["fused"] = True
                fusion_log.append(fusion_entry)
                fusion_id += 1
                continue
            elif result["confidence"] >= 0.5:
                fused_panel = np.vstack([panel_a, panel_b])
                all_panels[idx] = fused_panel
                all_panels.pop(idx + 1)
                panel_sources[idx] = {**src_a, "fusion_page_b": src_b["source_page"],
                                      "fusion_id": fusion_id}
                panel_sources.pop(idx + 1)
                fusion_entry["action"] = "fusion_revisar"
                fusion_entry["fused"] = True
                fusion_log.append(fusion_entry)
                fusion_id += 1
                continue
            else:
                fusion_entry["action"] = "separados"
                fusion_entry["fused"] = False
                fusion_log.append(fusion_entry)
                fusion_id += 1

        idx += 1

    for idx, panel in enumerate(all_panels):
        scene_counter += 1
        scene_num = scene_counter
        src = panel_sources[idx]

        if args.trim_margins:
            panel = trim_margins(panel, args.trim_threshold, args.trim_padding)

        extension = "webp" if args.webp else "png"
        name = f"escena_{scene_num:04d}.{extension}"
        target = os.path.join(args.output, name)
        imwrite_params = [cv2.IMWRITE_WEBP_QUALITY, 92] if args.webp else []
        if not cv2.imwrite(target, panel, imwrite_params):
            sys.exit(f"[ERROR] No se pudo guardar {target}")

        metrics = panel_metrics(panel)
        narrative = panel_narrative_score(panel)
        digest = hashlib.sha256(panel.tobytes()).hexdigest()

        score = narrative["score_raw"]
        if score < args.discard_score_threshold:
            if review_dir:
                cv2.imwrite(os.path.join(review_dir, name), panel)
            discarded_total += 1
            continue

        entry = {
            "scene": scene_num,
            "file": name,
            "source_page": src["source_page"],
            "source_pos": src.get("source_pos", -1),
            "width": int(panel.shape[1]),
            "height": int(panel.shape[0]),
            "sha256": digest,
            **metrics,
            **narrative,
        }
        if src.get("fusion_page_b"):
            entry["fused_from_page"] = src["fusion_page_b"]
            entry["fusion_confidence"] = next(
                (f["confidence"] for f in fusion_log
                 if f["fusion_id"] == src.get("fusion_id", -1)),
                None,
            )

        if digest in hashes:
            entry["duplicate_of"] = hashes[digest]
        else:
            hashes[digest] = scene_num

        entry["low_confidence"] = (
            entry.get("blank_ratio", 0) > 0.70
            or entry.get("quality_score", 1) < 0.25
            or entry.get("sharpness", 999) < 0.5
            or entry.get("stars", 3) <= 1
            or (entry.get("fusion_confidence") is not None and entry["fusion_confidence"] < 0.5)
        )

        merged_manifest.append(entry)

    return merged_manifest, diagnostics, fusion_log, discarded_total


def main() -> None:
    parser = argparse.ArgumentParser(description="Recorte inteligente de viñetas desde PDF con fusión entre páginas.")
    parser.add_argument("--pdf", required=True, help="Archivo PDF del capítulo.")
    parser.add_argument("--output", required=True, help="Carpeta de salida para viñetas PNG/WebP.")
    parser.add_argument("--dpi", type=int, default=200, help="Resolución de extracción de páginas (default 200).")
    parser.add_argument("--detection-mode", choices=("auto", "heuristic", "ai"), default="auto",
                        help="Modo de detección de viñetas: auto (híbrido), heuristic (solo heurística, más barato), ai (solo IA, más preciso). Default auto.")
    parser.add_argument("--min-panel-height", type=int, default=250)
    parser.add_argument("--min-height", type=int, default=400)
    parser.add_argument("--max-blank-ratio", type=float, default=0.50)
    parser.add_argument("--gutter-min-height", type=int, default=35)
    parser.add_argument("--gutter-std-threshold", type=float, default=12.0)
    parser.add_argument("--gutter-blank-ratio", type=float, default=.995)
    parser.add_argument("--fusion-min-confidence", type=float, default=0.97,
                        help="Umbral de confianza para fusión automática (0.0-1.0). Default 0.97.")
    parser.add_argument("--fusion-strip", type=int, default=25,
                        help="Altura en píxeles de la franja de comparación entre páginas.")
    parser.add_argument("--discard-score-threshold", type=float, default=2.5,
                        help="Puntuación narrativa por debajo de la cual se descarta la viñeta.")
    parser.add_argument("--no-discard", action="store_true", default=False,
                        help="Desactiva el descarte por puntuación baja. Por defecto se descartan viñetas de baja calidad.")
    parser.add_argument("--no-filter", action="store_true")
    parser.add_argument("--review", action="store_true")
    parser.add_argument("--on-uncertain", choices=("error", "full"), default="error")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--diagnostics", default=None)
    parser.add_argument("--fusion-log", default=None)
    parser.add_argument("--vision-provider", choices=("disabled", "ollama", "huggingface"), default="disabled")
    parser.add_argument("--vision-model", default="gemma3")
    parser.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    parser.add_argument("--vision-timeout", type=int, default=90)
    parser.add_argument("--vision-workers", type=int, default=2)
    parser.add_argument("--vision-max-side", type=int, default=1280)
    parser.add_argument("--expected-scenes", type=int, default=None)
    parser.add_argument("--trim-margins", action="store_true")
    parser.add_argument("--trim-threshold", type=int, default=245)
    parser.add_argument("--trim-padding", type=int, default=4)
    parser.add_argument("--webp", action="store_true")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--html-report", default=None)
    parser.add_argument("--ocr", action="store_true", help="Extrae texto de bocadillos con EasyOCR.")
    parser.add_argument("--ocr-languages", nargs="+", default=["es", "en"],
                        help="Idiomas para EasyOCR (default: es en).")
    args = parser.parse_args()

    if args.no_filter:
        args.min_height, args.max_blank_ratio = 0, 1.0
    if args.no_discard:
        args.discard_score_threshold = -999
    if args.gutter_min_height < 1 or args.gutter_std_threshold < 0 or not 0 < args.gutter_blank_ratio <= 1:
        parser.error("Los umbrales de gutter no son válidos.")
    if not 0 <= args.fusion_min_confidence <= 1:
        parser.error("--fusion-min-confidence debe estar entre 0.0 y 1.0.")

    try:
        vision = create_vision(args.vision_provider, args.vision_model, args.ollama_host,
                               args.vision_timeout, 2, args.vision_max_side)
    except VisionError as exc:
        sys.exit(f"[ERROR] Visión no disponible: {exc}")

    pdf_path = os.path.abspath(args.pdf)
    if not os.path.isfile(pdf_path):
        sys.exit(f"[ERROR] No se encontró el PDF: {pdf_path}")

    destination = os.path.abspath(args.output)
    os.makedirs(destination, exist_ok=True)

    if args.replace:
        for item in os.listdir(destination):
            if item.startswith("escena_") and item.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                os.unlink(os.path.join(destination, item))

    review_dir = os.path.join(destination, "descartadas") if args.review else None
    if review_dir:
        os.makedirs(review_dir, exist_ok=True)

    print(f"[INFO] Procesando PDF: {pdf_path}")
    manifest, diagnostics, fusion_log, discarded_total = crop_panels_from_pdf(pdf_path, args, review_dir)

    warnings = validate_panels(manifest)
    if warnings:
        print(f"[QC] {len(warnings)} advertencias de calidad detectadas:")
        for w in warnings[:20]:
            print(f"  - escena {w['scene']:04d}: {w['issue']}")

    if args.ocr:
        try:
            import easyocr
            reader = easyocr.Reader(args.ocr_languages, gpu=True)
            print("[INFO] Extrayendo texto con EasyOCR...")
            for entry in manifest:
                panel_path = os.path.join(destination, entry["file"])
                panel = cv2.imread(panel_path)
                if panel is not None:
                    entry["ocr_text"] = _ocr_text_for_panel(panel, reader)
        except ImportError:
            print("[AVISO] EasyOCR no instalado; OCR desactivado.")
        except Exception as exc:
            print(f"[ERROR] OCR falló: {exc}")

    if vision and manifest:
        prompt = "Describe la viñeta en español en una frase: personajes, acción, lugar, expresiones."
        with ThreadPoolExecutor(max_workers=args.vision_workers) as executor:
            futures = {executor.submit(vision.describe, os.path.join(destination, entry["file"]), prompt): entry
                       for entry in manifest}
            for future in as_completed(futures):
                entry = futures[future]
                try:
                    entry["vision_description"] = future.result()
                except VisionError as exc:
                    entry["vision_error"] = str(exc)
                except Exception as exc:
                    entry["vision_error"] = f"Error inesperado: {exc}"

    manifest_path = args.manifest or os.path.join(destination, "panels.json")
    _write_json(manifest_path, {
        "panels": manifest,
        "discarded": discarded_total,
        "discarded_gutter": sum(d["discarded"] for d in diagnostics),
        "fusions": len([f for f in fusion_log if f.get("fused")]),
        "qc_warnings": len(warnings),
        "qc_details": warnings,
        "settings": {k: v for k, v in vars(args).items() if k not in ("vision_provider", "vision_model", "ollama_host")},
    })

    diagnostics_path = args.diagnostics or os.path.join(destination, "crop_diagnostics.jsonl")
    with open(diagnostics_path, "w", encoding="utf-8") as handle:
        for item in diagnostics:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    fusion_path = args.fusion_log or os.path.join(destination, "fusion_log.jsonl")
    with open(fusion_path, "w", encoding="utf-8") as handle:
        for item in fusion_log:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    csv_path = args.csv or os.path.join(destination, "panels.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        fields = ["scene", "file", "source_page", "width", "height", "quality_score", "stars", "category",
                  "score_raw", "blank_ratio", "sharpness", "faces", "edge_density", "bubble_ratio",
                  "fused_from_page", "fusion_confidence", "ocr_text", "vision_description"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(manifest)

    report_path = args.html_report or os.path.join(destination, "crop_report.html")
    rows = "\n".join(
        f"<tr><td>{entry['scene']:04d}</td><td><img src='{escape(entry['file'], quote=True)}'></td>"
        f"<td>p{entry.get('source_page', '?')}</td><td>{entry.get('stars', '?')}★</td>"
        f"<td>{entry.get('category', '?')}</td><td>{entry.get('quality_score', 0):.2f}</td>"
        f"<td>{escape(str(entry.get('duplicate_of', '')))}</td>"
        f"<td>{escape(str(entry.get('fusion_confidence', '') or ''))}</td>"
        f"<td>{escape(entry.get('vision_description', entry.get('vision_error', '')))}</td></tr>"
        for entry in manifest
    )
    Path(report_path).write_text(
        "<!doctype html><meta charset='utf-8'>"
        "<style>body{font-family:sans-serif;margin:20px}img{max-width:180px;max-height:160px}"
        "td{padding:6px;border-bottom:1px solid #ddd;vertical-align:top}"
        "th{background:#333;color:#fff;padding:8px}</style>"
        "<h1>Diagnóstico de viñetas (PDF)</h1>"
        "<p>⭐ Puntuación narrativa | "
        f"Paneles: {len(manifest)} | Fusiones: {len([f for f in fusion_log if f.get('fused')])} | "
        f"QC warnings: {len(warnings)}</p>"
        "<table><tr><th>Escena</th><th>Viñeta</th><th>Página</th><th>Estrellas</th><th>Categoría</th>"
        "<th>Calidad</th><th>Duplicada</th><th>Fusión</th><th>Descripción</th></tr>"
        + rows + "</table>",
        encoding="utf-8",
    )

    total_fusions = len([f for f in fusion_log if f.get("fused")])
    total_revisar = len([f for f in fusion_log if f.get("action") == "fusion_revisar"])
    print(f"\nTotal: {len(manifest)} paneles. Fusiones: {total_fusions - total_revisar} (auto), {total_revisar} (revisar).")
    print(f"Manifiesto: {manifest_path}")
    print(f"Diagnóstico: {diagnostics_path}")
    print(f"Fusiones: {fusion_path}")
    print(f"QC: {len(warnings)} advertencias")

    if diagnostics and any(d["status"] in ("uncertain_split", "unreadable") for d in diagnostics):
        bad = [d["source"] for d in diagnostics if d["status"] in ("uncertain_split", "unreadable")]
        print(f"[AVISO] Páginas con recorte incierto: {', '.join(bad)}")

    if args.expected_scenes is not None and len(manifest) != args.expected_scenes:
        sys.exit(f"[ERROR] Se esperaban {args.expected_scenes} escenas, se obtuvieron {len(manifest)} paneles.")


if __name__ == "__main__":
    main()