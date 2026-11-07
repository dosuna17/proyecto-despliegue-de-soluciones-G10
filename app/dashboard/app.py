"""Tablero de clasificación de estadios del sueño (Streamlit).

Flujo en tres etapas, guardado en st.session_state["etapa"]:
    carga -> inferencia -> resultados

Ejecutar desde la raíz del repositorio:
    streamlit run app/dashboard/app.py
"""

from __future__ import annotations

import logging

import streamlit as st

import textos
from api_client import ClienteAPI
from config import obtener_config
from pantallas import carga, inferencia, resultados
from pantallas.comun import aviso_apoyo, mostrar_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

st.set_page_config(page_title="Clasificación del sueño", page_icon="🌙", layout="wide")


@st.cache_resource(show_spinner=False)
def _cliente() -> ClienteAPI:
    return ClienteAPI(obtener_config())


def main() -> None:
    try:
        config = obtener_config()
    except (FileNotFoundError, ValueError) as error:
        mostrar_error("El tablero no está bien configurado. Avise al equipo de soporte.", str(error))
        return

    st.title(textos.TITULO, anchor=False)
    st.caption(textos.SUBTITULO)
    aviso_apoyo()

    etapa = st.session_state.setdefault("etapa", "carga")
    pasos = {"carga": "1 · Carga del registro", "inferencia": "2 · Clasificación", "resultados": "3 · Resultados"}
    st.markdown(
        "  ›  ".join(f"**{t}**" if clave == etapa else f"<span style='opacity:.5'>{t}</span>"
                     for clave, t in pasos.items()),
        unsafe_allow_html=True,
    )

    cliente = _cliente()
    # Un solo contenedor para la etapa, que se reemplaza al cambiar de etapa.
    principal = st.empty()
    with principal.container():
        if etapa == "carga":
            carga.mostrar(config, cliente)
        elif etapa == "inferencia":
            inferencia.mostrar(config, cliente)
        elif etapa == "resultados" and "analisis" in st.session_state:
            resultados.mostrar(config)
        else:
            st.session_state["etapa"] = "carga"
            st.rerun()


main()
