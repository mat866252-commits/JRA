#!/usr/bin/env python3
"""Wrapper para ejecutar el pipeline en Google Colab.

Uso en Colab:
    !python run_colab.py --project "Capitulo_001" --input-dir ./input --chapter 1

Requiere GEMINI_API_KEY configurada como secret de Colab o en .env
"""

import os
import sys
from pathlib import Path

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)

sys.path.insert(0, str(PROJECT_ROOT))

if __name__ == "__main__":
    from cli.main import app
    app()
