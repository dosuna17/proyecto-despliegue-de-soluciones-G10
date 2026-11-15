"""Etapa 2: aviso de procesamiento mientras se genera la clasificación."""

from __future__ import annotations

import time
from typing import Any, Dict

import streamlit as st

from api_client import ClienteAPI, ErrorAPI
from pantallas.comun import minutos_a_texto, mostrar_error, reiniciar
from procesamiento import calcular_indicadores, comparar_con_experto, periodo_principal_sueno
from textos import ERROR_GENERICO


def _analizar(config: Dict[str, Any], registro, resultado) -> Dict[str, Any]:
    """Recorte al periodo principal de sueño e indicadores, a partir de la predicción."""

    duracion_epoca = config["senales"]["duracion_epoca_s"]
    epocas_por_min = 60 / duracion_epoca
    rec = config["recorte"]
    ventana = periodo_principal_sueno(
        resultado.estadios,
        margen_epocas=int(rec["margen_min"] * epocas_por_min),
        max_interrupcion_epocas=int(rec["max_interrupcion_min"] * epocas_por_min),
        min_sueno_epocas=int(rec["min_sueno_periodo_min"] * epocas_por_min),
        min_bloque_epocas=int(rec["min_bloque_sueno_min"] * epocas_por_min),
    )
    clases = [c for c in resultado.clases]
    indicadores = calcular_indicadores(
        resultado.estadios, resultado.confianzas, ventana,
        config["interpretacion"]["umbral_confianza"], duracion_epoca, clases,
    )
    concordancia = None
    if registro.tiene_hipnograma:
        concordancia = comparar_con_experto(resultado.estadios, registro.etiquetas_experto, ventana, clases)
    return {"ventana": ventana, "indicadores": indicadores, "concordancia": concordancia}


def mostrar(config: Dict[str, Any], cliente: ClienteAPI) -> None:
    registro = st.session_state.get("registro")
    seleccion = st.session_state.get("seleccion")
    modelo = st.session_state.get("modelo")
    if registro is None or not seleccion or modelo is None:
        reiniciar()
        st.rerun()

    duracion_epoca = config["senales"]["duracion_epoca_s"]
    n_epocas = int(registro.duracion_s // duracion_epoca)

    # Mientras corre la clasificación, Streamlit deja atenuada la pantalla anterior
    # debajo del aviso; se oculta para que el usuario solo vea el progreso.
    st.html("<style>div[data-stale='true']{display:none !important;}</style>")
    st.subheader("Estamos generando la clasificación", anchor=False)
    st.markdown(
        f"Se están analizando **{n_epocas:,} épocas de 30 segundos** "
        f"({minutos_a_texto(n_epocas * duracion_epoca / 60)}) de **{registro.nombre}** con "
        f"**{' + '.join(seleccion)}**. No cierre ni recargue esta página.".replace(",", ".")
    )
    pasos = st.empty()
    barra = st.progress(0.0, text="Preparando las señales…")
    comienzo = time.perf_counter()

    def progreso(hechos: int, total: int) -> None:
        transcurrido = time.perf_counter() - comienzo
        restante = transcurrido / hechos * (total - hechos) if hechos else 0
        texto = f"Clasificando: lote {hechos} de {total}"
        if hechos >= 2 and restante > 5:
            texto += f" · faltan unos {max(1, round(restante / 60))} min" if restante >= 60 else f" · faltan unos {restante:.0f} s"
        barra.progress(hechos / total, text=texto)

    try:
        pasos.markdown(
            "① Pasada preliminar para ubicar el sueño  \n"
            "② Reclasificando el tramo con 30 min de margen  \n"
            "③ Calculando los indicadores"
        )
        resultado = cliente.clasificar_con_recorte_preliminar(
            registro.senales, seleccion, modelo, n_epocas, progreso
        )
        st.session_state["resultado"] = resultado

        barra.progress(1.0, text="Calculando los indicadores…")
        st.session_state["analisis"] = _analizar(config, registro, resultado)
    except ErrorAPI as error:
        barra.empty()
        pasos.empty()
        mostrar_error(error.mensaje, error.detalle_tecnico)
        if st.button("Volver a la carga del registro", type="primary"):
            st.session_state["etapa"] = "carga"
            st.rerun()
        return
    except Exception as error:  # noqa: BLE001 - ningún error debe mostrarse como traza
        barra.empty()
        mostrar_error(ERROR_GENERICO, repr(error))
        if st.button("Volver a la carga del registro", type="primary"):
            st.session_state["etapa"] = "carga"
            st.rerun()
        return

    st.session_state["etapa"] = "resultados"
    st.session_state["epoca_actual"] = None
    st.rerun()
