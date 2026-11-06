"""Utilidades compartidas por las pantallas."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import logging

import streamlit as st

import textos

LOGGER = logging.getLogger("tablero")


def tema() -> str:
    """'oscuro' o 'claro' según el tema activo de Streamlit."""

    try:
        return "oscuro" if st.context.theme.type == "dark" else "claro"
    except Exception:  # noqa: BLE001 - versiones sin st.context.theme
        return "claro"


def hora_epoca(inicio: Optional[datetime], epoca: int, duracion_epoca_s: float, segundos: bool = False) -> str:
    """Hora del reloj de una época (la de la grabación) o tiempo transcurrido."""

    desfase = epoca * duracion_epoca_s
    if inicio is not None:
        momento = inicio.replace(tzinfo=None) + timedelta(seconds=desfase)
        return momento.strftime("%H:%M:%S" if segundos else "%H:%M")
    horas, resto = divmod(int(desfase), 3600)
    minutos, segs = divmod(resto, 60)
    return f"{horas:d} h {minutos:02d} min" + (f" {segs:02d} s" if segundos else "")


def minutos_a_texto(minutos: Optional[float]) -> str:
    if minutos is None:
        return "—"
    horas, mins = divmod(int(round(minutos)), 60)
    return f"{horas} h {mins:02d} min" if horas else f"{mins} min"


def aviso_apoyo() -> None:
    st.warning(textos.AVISO_APOYO, icon="⚕️")


def mostrar_error(mensaje: str, detalle: str = "") -> None:
    """Muestra un error en lenguaje llano.

    El detalle técnico va solo al registro del servidor (``docker logs``):
    puede contener direcciones internas o rutas que el usuario no debe ver.
    """

    st.error(mensaje)
    if detalle:
        LOGGER.error("%s | %s", mensaje, detalle)


def reiniciar() -> None:
    """Vuelve a la pantalla de carga conservando la configuración."""

    for clave in ("registro", "resultado", "resultado_comparacion", "analisis", "seleccion",
                  "epoca_actual", "error_inferencia"):
        st.session_state.pop(clave, None)
    st.session_state["etapa"] = "carga"
