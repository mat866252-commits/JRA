import os
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class SecretsManager:
    def __init__(self):
        self._secrets = {}
        self._load_from_env()
        self._load_from_file()

    def _load_from_env(self):
        env_vars = [
            "OPENAI_API_KEY",
            "GEMINI_API_KEY",
            "GEMINI_API_KEY_2",
            "GROQ_API_KEY",
            "DEEPSEEK_API_KEY",
            "OPENROUTER_API_KEY",
            "AZURE_OPENAI_API_KEY",
            "NVIDIA_API_KEY",
            "YOUTUBE_CLIENT_ID",
            "YOUTUBE_CLIENT_SECRET",
            "HUGGINGFACE_TOKEN"
        ]
        for var in env_vars:
            value = os.getenv(var)
            if value:
                self._secrets[var] = value
                logger.debug(f"Secreto cargado desde entorno: {var}")

    def _load_from_file(self):
        env_file = Path(__file__).resolve().parent / ".env"
        if env_file.exists():
            logger.info(f"Cargando secretos desde {env_file}")
            with open(env_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        # Las variables de entorno reales (p.ej. inyectadas
                        # por CI/CD o Docker en produccion) tienen prioridad
                        # sobre .env; si no, un .env desactualizado en disco
                        # podria pisar en silencio un secreto correcto que
                        # ya venia del entorno del proceso.
                        if key not in self._secrets:
                            self._secrets[key] = value.strip()

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        return self._secrets.get(key, default)

    def get_required(self, key: str) -> str:
        value = self.get(key)
        if not value:
            from exceptions import ConfigurationError
            raise ConfigurationError(
                f"El secreto '{key}' es requerido pero no esta configurado.\n"
                f"Configuralo en el archivo .env o como variable de entorno."
            )
        return value

    def has(self, key: str) -> bool:
        return key in self._secrets

    def mask(self, value: str) -> str:
        if not value or len(value) < 8:
            return "***"
        return value[:4] + "***" + value[-4:]


secrets = SecretsManager()
