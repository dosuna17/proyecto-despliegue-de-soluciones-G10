"""Tablero de clasificación de estadios del sueño (Streamlit).

Flujo en tres etapas, guardado en st.session_state["etapa"]:
    carga -> inferencia -> resultados

Ejecutar desde la raíz del repositorio:
    streamlit run app/dashboard/app.py
"""

from __future__ import annotations

import logging
from html import escape

import requests
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


@st.cache_data(ttl=60, show_spinner=False)
def _estado_modelo(api_url: str, timeout: int) -> dict:
    try:
        respuesta = requests.get(f"{api_url}/health", timeout=timeout)
        respuesta.raise_for_status()
        return respuesta.json()
    except requests.RequestException:
        return {}


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

    estado = _estado_modelo(
        config["api"]["url"], config["api"]["timeout_consulta_s"]
    )
    run_name = estado.get("model_run_name")
    run_id = estado.get("model_run_id")
    if run_name and run_id:
        etiqueta_run = f"{run_name} · {run_id}"
    elif run_name or run_id:
        etiqueta_run = run_name or run_id
    elif estado.get("model_use_mlflow"):
        etiqueta_run = "no se encontró un run con champion=True y el artefacto configurado"
    else:
        etiqueta_run = "no disponible en la metadata del checkpoint"
    st.markdown(
        """
        <style>
        .modelo-footer {
            margin-top: 2rem; padding: .7rem 0; text-align: center; font-size: .8rem;
            border-top: 1px solid rgba(128, 128, 128, .25);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="modelo-footer">Run del modelo activo: <code>{escape(str(etiqueta_run))}</code></div>',
        unsafe_allow_html=True,
    )


main()
