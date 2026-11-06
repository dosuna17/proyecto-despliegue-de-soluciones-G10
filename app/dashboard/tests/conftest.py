"""Hace importables los módulos del tablero (se ejecutan como scripts de Streamlit)."""

import sys
from pathlib import Path

CARPETA_TABLERO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CARPETA_TABLERO))
sys.path.insert(0, str(CARPETA_TABLERO / "tests"))
