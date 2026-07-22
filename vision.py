"""Adaptadores opcionales de visión para describir viñetas.

No hay dependencia obligatoria: Ollama se consulta por HTTP local y Hugging
Face se importa de forma perezosa. Las descripciones se guardan en manifests
para diagnosticar y revisar; nunca modifican un panel por sí solas.
"""

from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
import hashlib
import os
from pathlib import Path

# ===== COMPATIBILITY LAYER =====
try:
    from llm.provider import LLMProvider
    from llm.factory import LLMProviderFactory
except ImportError:
    LLMProvider = None
    LLMProviderFactory = None


class VisionCache:
    """Caché persistente para almacenar las descripciones e inferencias de visión."""
    def __init__(self, cache_file: str | None = None):
        if cache_file is None:
            # Usar la carpeta memory si existe
            if os.path.isdir("memory"):
                self.cache_file = os.path.join("memory", "vision_cache.json")
            else:
                self.cache_file = ".vision_cache.json"
        else:
            self.cache_file = cache_file
            
        self.cache = {}
        self.load()
        
    def load(self) -> None:
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            except Exception as exc:
                print(f"[AVISO] No se pudo cargar caché de visión: {exc}")
                self.cache = {}
                
    def save(self) -> None:
        try:
            parent_dir = os.path.dirname(os.path.abspath(self.cache_file))
            os.makedirs(parent_dir, exist_ok=True)
            tmp_path = self.cache_file + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self.cache_file)
        except Exception as exc:
            print(f"[AVISO] No se pudo guardar caché de visión: {exc}")
            
    def get_key(self, model: str, prompt: str, image_paths: list[str] | None = None) -> str:
        hasher = hashlib.md5(usedforsecurity=False)
        hasher.update(model.encode("utf-8"))
        hasher.update(prompt.encode("utf-8"))
        if image_paths:
            for path in sorted(image_paths):
                p = Path(path)
                if p.exists():
                    stat = p.stat()
                    # Hash de la ruta, tamaño y fecha de modificación
                    hasher.update(f"{p.name}:{stat.st_size}:{stat.st_mtime}".encode("utf-8"))
                else:
                    hasher.update(f"{p.name}:missing".encode("utf-8"))
        return hasher.hexdigest()
        
    def get(self, key: str) -> str | None:
        return self.cache.get(key)
        
    def set(self, key: str, value: str) -> None:
        self.cache[key] = value
        self.save()


def encoded_image(path: str, max_side: int) -> str:
    """Codifica una versión reducida para no enviar megapíxeles innecesarios."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise VisionError(f"Pillow no está instalado. Instálalo con: pip install Pillow") from exc
    
    try:
        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail((max_side, max_side))
            from io import BytesIO
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=88, optimize=True)
            data = buffer.getvalue()
    except Exception as exc:
        raise VisionError(f"No se pudo preparar {Path(path).name} para visión: {exc}") from exc
    return base64.b64encode(data).decode("ascii")


class VisionError(RuntimeError):
    pass


class OllamaVision:
    """
    Adaptador de compatibilidad para OllamaProvider.

    Mantiene interfaz antigua pero delega a LLMProvider.
    Codigo antiguo que usa OllamaVision directamente sigue funcionando.
    """

    def __init__(self, model: str, host: str = "http://127.0.0.1:11434", timeout: int = 90, retries: int = 2, max_side: int = 1280):
        from llm.ollama import OllamaProvider

        self._provider = OllamaProvider(
            model=model,
            host=host,
            timeout=timeout,
            retries=retries,
            max_side=max_side
        )
        self.model = model
        self.host = host
        self.timeout = timeout
        self.retries = retries
        self.max_side = max_side
        self.cache = VisionCache()

    def describe(self, image_path: str, prompt: str) -> str:
        key = self.cache.get_key(self.model, prompt, [image_path])
        cached = self.cache.get(key)
        if cached:
            print(f"[CACHE HIT] Usando respuesta guardada para {Path(image_path).name}")
            return cached

        result = self._provider.describe(image_path, prompt)
        self.cache.set(key, result)
        return result

    def describe_text(self, prompt: str) -> str:
        key = self.cache.get_key(self.model, prompt)
        cached = self.cache.get(key)
        if cached:
            return cached

        result = self._provider.describe_text(prompt)
        self.cache.set(key, result)
        return result

    def describe_multi(self, image_paths: list[str], prompt: str) -> str:
        key = self.cache.get_key(self.model, prompt, image_paths)
        cached = self.cache.get(key)
        if cached:
            names = ", ".join(Path(p).name for p in image_paths[:3])
            if len(image_paths) > 3:
                names += f" y {len(image_paths)-3} mas"
            print(f"[CACHE HIT] Usando respuesta guardada para multiples imagenes ({names})")
            return cached

        result = self._provider.describe_multi(image_paths, prompt)
        self.cache.set(key, result)
        return result

    def health_check(self) -> bool:
        try:
            self._provider.health_check()
            return True
        except Exception as exc:
            raise VisionError(str(exc)) from exc


class LLMVisionProvider:
    """Generic wrapper to use any LLMProvider (Gemini, OpenRouter) as a vision provider."""

    def __init__(self, provider_name: str, model: str | None = None, timeout: int = 90, retries: int = 2, max_side: int = 1280):
        if LLMProviderFactory is None:
            raise VisionError("Sistema LLM no disponible. Revisa que llm/ exista.")
        kw = {"timeout": timeout, "retries": retries, "max_side": max_side}
        if model is not None:
            kw["model"] = model
        self.provider = LLMProviderFactory.create(provider_name, **kw)
        self.model_name = f"{provider_name}/{self.provider.model}"
        self.cache = VisionCache()
        self.max_side = max_side

    def describe(self, image_path: str, prompt: str) -> str:
        key = self.cache.get_key(self.model_name, prompt, [image_path])
        cached = self.cache.get(key)
        if cached:
            print(f"[CACHE HIT] Usando respuesta guardada para {Path(image_path).name}")
            return cached

        result = self.provider.describe(image_path, prompt)
        self.cache.set(key, result)
        return result

    def describe_text(self, prompt: str) -> str:
        key = self.cache.get_key(self.model_name, prompt)
        cached = self.cache.get(key)
        if cached:
            return cached

        result = self.provider.describe_text(prompt)
        self.cache.set(key, result)
        return result

    def health_check(self) -> bool:
        try:
            self.provider.health_check()
            return True
        except Exception as exc:
            raise VisionError(str(exc)) from exc


class HuggingFaceVision:
    def __init__(self, model: str):
        try:
            from transformers import AutoProcessor, AutoModelForImageTextToText
        except ImportError as exc:
            raise VisionError("Instala transformers y torch para usar Hugging Face Vision.") from exc
        try:
            self.processor = AutoProcessor.from_pretrained(model)
            self.model = AutoModelForImageTextToText.from_pretrained(model)
            self.model_name = model
            self.cache = VisionCache()
        except Exception as exc:
            raise VisionError(f"No se pudo cargar el modelo {model}: {exc}") from exc

    def describe(self, image_path: str, prompt: str) -> str:
        key = self.cache.get_key(self.model_name, prompt, [image_path])
        cached = self.cache.get(key)
        if cached:
            print(f"[CACHE HIT] Usando respuesta guardada para {Path(image_path).name}")
            return cached

        try:
            from PIL import Image
        except ImportError as exc:
            raise VisionError("Instala Pillow para usar Hugging Face Vision.") from exc
        
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        
        try:
            import torch
            with torch.inference_mode():
                outputs = self.model.generate(**inputs, max_new_tokens=100)
            text = self.processor.decode(outputs[0], skip_special_tokens=True).strip()
            self.cache.set(key, text)
            return text
        except Exception as exc:
            raise VisionError(f"Hugging Face no pudo describir {Path(image_path).name}: {exc}") from exc


def create_vision(provider: str, model: str, ollama_host: str, timeout: int, retries: int, max_side: int = 1280):
    provider = provider.lower()
    if provider == "disabled":
        return None
    if provider == "ollama":
        vision = OllamaVision(model, ollama_host, timeout, retries, max_side)
        vision.health_check()
        return vision
    if provider == "huggingface":
        return HuggingFaceVision(model)
    if provider in ("gemini", "openrouter"):
        kw_model = model if model else None  # None deja que el factory use su default
        vision = LLMVisionProvider(provider, kw_model, timeout, retries, max_side)
        vision.health_check()
        return vision
    raise ValueError(f"Proveedor de visión no válido: {provider}")
