import signal
import sys
import logging
from typing import Callable

logger = logging.getLogger(__name__)


class GracefulShutdown:
    def __init__(self):
        self.shutdown_requested = False
        self.cleanup_callbacks: list[Callable] = []
        self.cleanup_failed = False

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        sig_name = signal.Signals(signum).name
        logger.warning(f"\nSenal {sig_name} recibida. Iniciando shutdown limpio...")
        self.shutdown_requested = True

        for callback in reversed(self.cleanup_callbacks):
            try:
                callback()
            except Exception as e:
                logger.error(f"Error en callback de limpieza: {e}")
                self.cleanup_failed = True

        logger.info("Shutdown limpio completado")
        sys.exit(1 if self.cleanup_failed else 0)

    def register_cleanup(self, callback: Callable):
        self.cleanup_callbacks.append(callback)

    def is_shutdown_requested(self) -> bool:
        return self.shutdown_requested


shutdown_handler = GracefulShutdown()
