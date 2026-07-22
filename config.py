# config.py
import re
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class VideoConfig:
    """Configuración inmutable para el ensamblaje de video."""
    width: int
    height: int
    fps: int
    audio_codec: str = "aac"

    def __post_init__(self):
        if not (100 <= self.width <= 4096):
            raise ValueError(f"Ancho inválido: {self.width}. Debe estar entre 100 y 4096.")
        if not (100 <= self.height <= 4096):
            raise ValueError(f"Alto inválido: {self.height}. Debe estar entre 100 y 4096.")
        if not (1 <= self.fps <= 120):
            raise ValueError(f"FPS inválidos: {self.fps}. Debe estar entre 1 y 120.")

def parse_resolution(res_str: str) -> tuple[int, int]:
    """Parsea y valida una cadena de resolución tipo '1920x1080'."""
    match = re.match(r"^(\d{3,4})x(\d{3,4})$", res_str.strip())
    if not match:
        raise ValueError(f"Formato de resolución inválido: '{res_str}'. Usa el formato 'ANCHOxALTO' (ej: 1920x1080).")
    return int(match.group(1)), int(match.group(2))
