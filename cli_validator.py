import argparse
import re
import sys
from pathlib import Path
from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator, model_validator

from exceptions import ConfigurationError


class PipelineArgs(BaseModel):
    project: str = Field(..., min_length=1, max_length=100)
    input_dir: Path = Field(...)
    script: Optional[Path] = Field(default=None)
    output_dir: Path = Field(default=Path("output"))
    resolution: str = Field(default="1920x1080")
    fps: int = Field(default=30, ge=1, le=120)
    parallel: bool = Field(default=False)
    max_workers: int = Field(default=4, ge=1, le=32)
    skip: list[str] = Field(default_factory=list)
    chapter: int = Field(default=1, ge=1)
    text_provider: Literal["ollama", "gemini", "openrouter"] = Field(default="gemini")
    text_model: str = Field(default="gemini-flash-lite-latest")
    vision_provider: Literal["ollama", "gemini", "openrouter", "disabled"] = Field(default="gemini")
    text_timeout: int = Field(default=120, ge=30, le=600)
    vision_timeout: int = Field(default=120, ge=30, le=600)
    providers_config: Optional[Path] = Field(default=None)
    crf: int = Field(default=23, ge=0, le=51)
    preset: str = Field(default="medium")
    encoder: str = Field(default="libx264")
    music: Optional[Path] = Field(default=None)
    music_base_dir: Optional[Path] = Field(default=None)
    music_volume: float = Field(default=-22.0)
    crossfade: int = Field(default=12, ge=0, le=60,
                            description="Frames de crossfade entre segmentos (0=desactivado)")
    target_duration: float = Field(default=600.0, ge=120, le=3600,
                                    description="Duración objetivo en segundos")
    target_panels_min: int = Field(default=25, ge=10, le=60)
    target_panels_max: int = Field(default=60, ge=15, le=80)
    quality_audit: bool = Field(default=False, description="Ejecuta Quality Auditor post-renderizado")
    quality_audit_fix: bool = Field(default=False, description="Auto-corrige paneles incongruentes")
    thumbnail: bool = Field(default=False, description="Genera miniatura automatica con Thumbnail God")
    skip_low_confidence: bool = Field(default=True, description="Salta paneles de baja confianza")
    force: bool = Field(default=False, description="Re-ejecuta todas las fases ignorando outputs existentes")

    @field_validator("project")
    @classmethod
    def validate_project_name(cls, v: str) -> str:
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"Nombre de proyecto invalido: '{v}'. Solo se permiten letras, numeros, guiones y guiones bajos.")
        return v

    @field_validator("resolution")
    @classmethod
    def validate_resolution(cls, v: str) -> str:
        if not re.match(r"^\d{3,4}x\d{3,4}$", v):
            raise ValueError(f"Resolucion invalida: '{v}'. Usa el formato ANCHOxALTO (ej: 1920x1080).")
        return v

    @model_validator(mode="after")
    def validate_paths_exist(self):
        if not self.input_dir.exists():
            raise ValueError(f"El directorio de entrada no existe: {self.input_dir}")
        if not self.input_dir.is_dir():
            raise ValueError(f"La ruta de entrada no es un directorio: {self.input_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self


def parse_and_validate_args() -> PipelineArgs:
    parser = argparse.ArgumentParser(
        description="Pipeline de resumenes de manhwa (v2)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--project", type=str, required=True)
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Carpeta con los PDFs de los capítulos.")
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--resolution", type=str, default="1920x1080")
    parser.add_argument("--fps", type=int, default=30, choices=[24, 25, 30, 60])
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--skip", type=str, nargs="*", default=[])
    parser.add_argument("--chapter", type=int, default=1, help="Número de capítulo.")
    parser.add_argument("--text-provider", type=str, choices=["ollama", "gemini", "openrouter"], default="gemini",
                        help="Proveedor LLM. 'gemini' requiere GEMINI_API_KEY.")
    parser.add_argument("--text-model", type=str, default="gemini-flash-lite-latest",
                        help="Modelo de texto (default: gemini-flash-lite-latest).")
    parser.add_argument("--vision-provider", type=str, choices=["ollama", "gemini", "openrouter", "disabled"], default="gemini",
                        help="Proveedor de visión para descripción de viñetas.")
    parser.add_argument("--text-timeout", type=int, default=120,
                        help="Timeout en segundos para llamadas de texto (generación de guion).")
    parser.add_argument("--vision-timeout", type=int, default=120,
                        help="Timeout en segundos para llamadas de visión.")
    parser.add_argument("--crf", type=int, default=23,
                        help="Calidad de video: menor = mejor calidad (0-51).")
    parser.add_argument("--preset", type=str, default="medium",
                        help="Preset de FFmpeg: ultrafast, fast, medium, slow, veryslow.")
    parser.add_argument("--encoder", type=str, default="libx264",
                        help="Codificador de video FFmpeg.")
    parser.add_argument("--music", type=Path, default=None,
                        help="Archivo de música de fondo (mp3/wav) con ducking automático.")
    parser.add_argument("--music-base-dir", type=Path, default=None,
                        help="Carpeta con música por categoría (tension/, action/, sad/).")
    parser.add_argument("--music-volume", type=float, default=-22.0,
                        help="Volumen de la música en dB (antes del ducking).")
    parser.add_argument("--crossfade", type=int, default=12,
                        help="Frames de crossfade entre segmentos (0 = desactivado, default 12).")
    parser.add_argument("--target-duration", type=float, default=600.0,
                        help="Duración objetivo del video en segundos (default 600 = 10 min).")
    parser.add_argument("--target-panels-min", type=int, default=25,
                        help="Mínimo de paneles seleccionados (default 25).")
    parser.add_argument("--target-panels-max", type=int, default=60,
                        help="Máximo de paneles seleccionados (default 60).")
    parser.add_argument("--quality-audit", action="store_true",
                        help="Ejecuta Quality Auditor para validar coherencia narrativa.")
    parser.add_argument("--quality-audit-fix", action="store_true",
                        help="Auto-corrige paneles incongruentes (requiere --quality-audit).")
    parser.add_argument("--thumbnail", action="store_true",
                        help="Genera miniatura automatica con Thumbnail God.")
    parser.add_argument("--skip-low-confidence", action="store_true", default=True,
                        help="Salta paneles de baja confianza (default: True).")
    parser.add_argument("--no-skip-low-confidence", action="store_false", dest="skip_low_confidence",
                        help="No saltar paneles de baja confianza.")
    parser.add_argument("--script", type=Path, default=None,
                        help="Ruta al guion existente (si no se usa el generado por script_generator_v3).")
    parser.add_argument("--providers-config", type=Path, default=None,
                        help="Ruta al archivo providers.yaml (nuevo sistema de orquestación).")
    parser.add_argument("--force", action="store_true",
                        help="Re-ejecuta todas las fases ignorando outputs existentes.")

    raw_args = parser.parse_args()

    args_dict = {
        "project": raw_args.project,
        "input_dir": Path(raw_args.input_dir),
        "output_dir": Path(raw_args.output_dir),
        "resolution": raw_args.resolution,
        "fps": raw_args.fps,
        "parallel": raw_args.parallel,
        "max_workers": raw_args.max_workers,
        "skip": raw_args.skip or [],
        "chapter": raw_args.chapter,
        "text_provider": raw_args.text_provider,
        "text_model": raw_args.text_model,
        "vision_provider": raw_args.vision_provider,
        "text_timeout": raw_args.text_timeout,
        "vision_timeout": raw_args.vision_timeout,
        "crf": raw_args.crf,
        "preset": raw_args.preset,
        "encoder": raw_args.encoder,
        "music": raw_args.music,
        "music_base_dir": raw_args.music_base_dir,
        "music_volume": raw_args.music_volume,
        "crossfade": raw_args.crossfade,
        "target_duration": raw_args.target_duration,
        "target_panels_min": raw_args.target_panels_min,
        "target_panels_max": raw_args.target_panels_max,
        "quality_audit": raw_args.quality_audit,
        "quality_audit_fix": raw_args.quality_audit_fix,
        "thumbnail": raw_args.thumbnail,
        "skip_low_confidence": getattr(raw_args, 'skip_low_confidence', True),
        "providers_config": getattr(raw_args, 'providers_config', None),
        "script": getattr(raw_args, 'script', None),
        "force": getattr(raw_args, 'force', False),
    }

    try:
        return PipelineArgs(**args_dict)
    except Exception as e:
        raise ConfigurationError(f"Validacion de argumentos fallo: {e}") from e
