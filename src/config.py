"""Carga la configuración compartida del proyecto desde el archivo .env."""

from pathlib import Path
import os

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
