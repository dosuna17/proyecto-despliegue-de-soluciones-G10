"""Etapa 1: cargar el registro y elegir las señales."""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List

import streamlit as st

import textos
from api_client import ClienteAPI, ErrorAPI, InfoModelo
from pantallas.comun import hora_epoca, minutos_a_texto, mostrar_error
from procesamiento import CANALES_EEG, numero_combinacion, validar_combinacion
from registro import ErrorRegistro, Registro, leer_demo, leer_edf_subido, listar_demos

CANALES_INTERFAZ = ("EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal", "EMG submental")


@st.cache_data(show_spinner=False, max_entries=4)
def _leer_demo_cache(ruta: str) -> Registro:
    return leer_demo(ruta)


@st.cache_data(show_spinner=False, ttl=600)
def _info_modelo_cache(_cliente_id: str, url: str, _cliente: ClienteAPI) -> InfoModelo:
    return _cliente.info_modelo()


def _cargar_desde_archivo(config: Dict[str, Any]) -> Registro | None:
    limite = config["carga"]["tamano_maximo_mb"]
    col1, col2 = st.columns(2)
    with col1:
        psg = st.file_uploader(
            "Archivo de señales (PSG) · obligatorio",
            type=["edf"],
            help=f"Archivo EDF con las señales de la polisomnografía, p. ej. SC4001E0-PSG.edf. Máximo {limite} MB.",
        )
    with col2:
        hipno = st.file_uploader(
            "Hipnograma del experto · opcional",
            type=["edf"],
            help="Si lo sube, el tablero compara la clasificación del modelo con la del experto.",
        )
    if psg is None:
        st.caption(
            "Formatos admitidos: EDF/EDF+ con al menos un EEG (Fpz-Cz o Pz-Oz). "
            "Los archivos se procesan solo durante su sesión y no se guardan."
        )
        return None
    if psg.size > limite * 1024 * 1024:
        st.error(f"El archivo pesa {psg.size / 1e6:.0f} MB y el máximo permitido es {limite} MB.")
        return None

    contenido_psg = psg.getvalue()
    contenido_hipno = hipno.getvalue() if hipno is not None else None
    huella = hashlib.sha1(contenido_psg[:1_000_000] + (contenido_hipno or b"") + str(psg.size).encode()).hexdigest()
    if st.session_state.get("huella_carga") == huella and "registro" in st.session_state:
        return st.session_state["registro"]

    with st.spinner("Leyendo el archivo y verificando las señales…"):
        try:
            registro = leer_edf_subido(
                contenido_psg, contenido_hipno, psg.name.rsplit(".", 1)[0], config["senales"]["alias"],
                config["senales"]["frecuencia_hz"], config["senales"]["duracion_epoca_s"],
            )
        except ErrorRegistro as error:
            mostrar_error(str(error), repr(error.__cause__))
            return None
    st.session_state["huella_carga"] = huella
    return registro


def _cargar_demo(config: Dict[str, Any]) -> Registro | None:
    demos = listar_demos(config["estudios_demo"]["carpeta"])
    if not demos:
        st.info(
            "No hay estudios de demostración instalados en este servidor. Puede subir su propio "
            "registro en la otra opción."
        )
        return None
    etiquetas = {d["ruta"]: f"{d['nombre']} · {d.get('descripcion', '')}" for d in demos}
    ruta = st.selectbox("Estudio", list(etiquetas), format_func=etiquetas.get)
    st.caption(
        "Registros reales de Sleep-EDFx (Sleep Cassette) de personas que **no** se usaron para "
        "entrenar el modelo, con el hipnograma de su experto para comparar."
    )
    with st.spinner("Cargando el estudio…"):
        return _leer_demo_cache(ruta)


def _resumen_registro(registro: Registro, config: Dict[str, Any]) -> None:
    duracion_epoca = config["senales"]["duracion_epoca_s"]
    n_epocas = int(registro.duracion_s // duracion_epoca)
    c1, c2, c3 = st.columns(3)
    c1.metric("Duración del registro", minutos_a_texto(registro.duracion_s / 60))
    c2.metric(
        "Inicio de la grabación",
        hora_epoca(registro.inicio, 0, duracion_epoca) if registro.inicio else "No indicado",
    )
    c3.metric("Hipnograma del experto", "Sí" if registro.tiene_hipnograma else "No")
    st.caption(f"{n_epocas:,} épocas de 30 segundos.".replace(",", "."))
    for aviso in registro.avisos:
        st.info(aviso)
    if n_epocas < config["carga"]["min_epocas"]:
        st.error(
            f"El registro dura menos de {config['carga']['min_epocas'] * duracion_epoca / 60:.1f} minutos; "
            "el modelo necesita al menos ese tiempo para clasificar."
        )


def _selector_canales(registro: Registro, modelo: InfoModelo) -> List[str]:
    soportados = {c for combo in modelo.combinaciones for c in combo}
    seleccion: List[str] = []
    st.markdown("**Señales para el análisis**")
    columnas = st.columns(len(CANALES_INTERFAZ))
    for columna, canal in zip(columnas, CANALES_INTERFAZ):
        en_registro = canal in registro.canales_disponibles
        en_modelo = canal in soportados
        if not en_registro:
            estado = "No está en el archivo"
        elif not en_modelo:
            estado = "El modelo aún no la usa"
        else:
            estado = "Disponible"
        habilitado = en_registro and en_modelo
        with columna:
            marcado = st.checkbox(
                textos.NOMBRES_CANAL[canal],
                value=habilitado,
                disabled=not habilitado,
                key=f"canal_{canal}_{registro.nombre}",
                help=textos.AYUDA_CANAL[canal],
            )
            icono = "✅" if habilitado else "⛔"
            st.caption(f"{icono} {estado}")
        if marcado and habilitado:
            seleccion.append(canal)
    return seleccion


def _tarjeta_modelo(config: Dict[str, Any], modelo: InfoModelo) -> None:
    ref = config["modelo"].get("desempeno_referencia") or {}
    with st.container(border=True):
        st.markdown(f"**Modelo de clasificación: {modelo.nombre}**")
        if modelo.descripcion:
            st.caption(modelo.descripcion)
        if ref.get("exactitud") is not None and ref.get("kappa") is not None:
            texto = (
                f"En personas que no participaron en su entrenamiento coincidió con el experto en "
                f"el {ref['exactitud']:.0%} de las épocas (kappa de Cohen {ref['kappa']:.2f}; dos "
                f"expertos humanos suelen lograr {config['interpretacion']['kappa_entre_expertos']:.2f})."
            )
            if ref.get("descripcion_evaluacion"):
                texto += f" {ref['descripcion_evaluacion']}"
            st.caption(texto)


def mostrar(config: Dict[str, Any], cliente: ClienteAPI) -> None:
    with st.expander("¿Cómo se usa?", expanded="registro" not in st.session_state):
        columnas = st.columns(len(textos.PASOS))
        for columna, (titulo, descripcion) in zip(columnas, textos.PASOS):
            columna.markdown(f"**{titulo}**")
            columna.caption(descripcion)

    try:
        modelo = _info_modelo_cache(str(id(cliente)), cliente.url, cliente)
    except ErrorAPI as error:
        mostrar_error(error.mensaje, error.detalle_tecnico)
        if st.button("Reintentar conexión"):
            _info_modelo_cache.clear()
            st.rerun()
        return

    st.subheader("1. Registro", anchor=False)
    fuente = st.radio(
        "¿Qué registro desea analizar?",
        ["Estudio de demostración", "Subir mi registro (EDF)"],
        horizontal=True,
        label_visibility="collapsed",
    )
    registro = _cargar_demo(config) if fuente == "Estudio de demostración" else _cargar_desde_archivo(config)
    if registro is None:
        st.session_state.pop("registro", None)
        return
    st.session_state["registro"] = registro
    _resumen_registro(registro, config)

    st.subheader("2. Señales", anchor=False)
    seleccion = _selector_canales(registro, modelo)
    problema = validar_combinacion(seleccion, registro.canales_disponibles, modelo.combinaciones)
    if not any(c in registro.canales_disponibles for c in CANALES_EEG):
        st.error("Este registro no contiene ninguna señal EEG (Fpz-Cz o Pz-Oz), que el modelo necesita para clasificar.")
    elif problema:
        st.warning(problema)
    else:
        numero = numero_combinacion(seleccion)
        st.success(f"Combinación {numero} de 12: {' + '.join(seleccion)}.")
    _tarjeta_modelo(config, modelo)

    n_epocas = int(registro.duracion_s // config["senales"]["duracion_epoca_s"])
    listo = problema is None and n_epocas >= config["carga"]["min_epocas"]
    if st.button("Generar clasificación", type="primary", disabled=not listo, use_container_width=True):
        st.session_state["seleccion"] = seleccion
        st.session_state["modelo"] = modelo
        st.session_state["etapa"] = "inferencia"
        st.rerun()
