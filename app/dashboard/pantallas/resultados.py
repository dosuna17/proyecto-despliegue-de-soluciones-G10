"""Etapa 3: resultados del análisis."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import streamlit as st

import graficas
import informe
import textos
from pantallas.comun import hora_epoca, minutos_a_texto, reiniciar, tema
from procesamiento import comparar_con_experto, describir_umbrales, numero_combinacion, para_revisar
from registro import Registro


def _datos_informe(config, registro: Registro, resultado, analisis, seleccion, modelo) -> informe.DatosInforme:
    return informe.DatosInforme(
        nombre_registro=registro.nombre,
        inicio=registro.inicio,
        canales=list(seleccion),
        modelo=modelo.nombre,
        estadios=np.asarray(resultado.estadios, dtype=object),
        confianzas=np.asarray(resultado.confianzas, dtype=float),
        probabilidades=np.asarray(resultado.probabilidades, dtype=float),
        clases=list(resultado.clases),
        ventana=analisis["ventana"],
        indicadores=analisis["indicadores"],
        umbrales=config["interpretacion"]["umbrales_confianza"],
        kappa_referencia=config["interpretacion"]["kappa_entre_expertos"],
        duracion_epoca_s=config["senales"]["duracion_epoca_s"],
        experto=None if not registro.tiene_hipnograma else np.asarray(registro.etiquetas_experto[: len(resultado.estadios)], dtype=object),
        concordancia=analisis["concordancia"],
        recorte=dict(config["recorte"]),
    )


def _huella(datos: informe.DatosInforme) -> str:
    """Identifica el análisis para no regenerar los archivos en cada interacción."""

    return f"{datos.nombre_registro}|{'+'.join(datos.canales)}|{len(datos.estadios)}|{float(datos.confianzas.sum()):.6f}|{sorted(datos.umbrales.items())}"


@st.cache_data(show_spinner="Preparando el informe PDF…", max_entries=4)
def _pdf_cache(_nombre: str, huella: str, _datos: informe.DatosInforme) -> bytes:
    return informe.generar_pdf(_datos)


@st.cache_data(show_spinner="Preparando el archivo de Excel…", max_entries=4)
def _excel_cache(_nombre: str, huella: str, _datos: informe.DatosInforme) -> bytes:
    return informe.generar_excel(_datos)


def _ficha(config, registro, seleccion, modelo, ventana, indicadores, n_total) -> None:
    duracion_epoca = config["senales"]["duracion_epoca_s"]
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"**Registro**  \n{registro.nombre}")
        c2.markdown(
            f"**Señales usadas** (combinación {numero_combinacion(seleccion)} de 12)  \n{' + '.join(seleccion)}"
        )
        c3.markdown(f"**Modelo**  \n{modelo.nombre}")
        if ventana.metodo == "prediccion":
            fuera = n_total - indicadores.epocas_analizadas
            st.markdown(
                f"**Ventana analizada:** {hora_epoca(registro.inicio, ventana.inicio, duracion_epoca)} – "
                f"{hora_epoca(registro.inicio, ventana.fin, duracion_epoca)} "
                f"({minutos_a_texto(indicadores.minutos_analizados)} de "
                f"{minutos_a_texto(n_total * duracion_epoca / 60)} grabados). "
                f"Se dejaron fuera {textos.epocas(fuera)} de vigilia antes y después del sueño.".replace(",", "."),
                help=textos.AYUDA["ventana"],
            )
        elif ventana.metodo == "sin_sueno":
            st.warning("El modelo no encontró sueño en este registro; se muestra el registro completo.")
        else:
            st.warning(
                "No se encontró un periodo de sueño claro (menos de "
                f"{config['recorte']['min_sueno_periodo_min']} minutos seguidos); se muestra el registro completo."
            )


def _indicadores(config, registro, indicadores) -> None:
    duracion_epoca = config["senales"]["duracion_epoca_s"]
    umbrales = config["interpretacion"]["umbrales_confianza"]
    st.subheader("Resumen del sueño (estimado por el modelo)", anchor=False)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Tiempo total de sueño", minutos_a_texto(indicadores.tiempo_total_sueno_min),
              help=textos.AYUDA["tiempo_total_sueno"])
    if indicadores.inicio_sueno_epoca is not None:
        periodo = (f"{hora_epoca(registro.inicio, indicadores.inicio_sueno_epoca, duracion_epoca)} – "
                   f"{hora_epoca(registro.inicio, indicadores.fin_sueno_epoca + 1, duracion_epoca)}")
    else:
        periodo = "—"
    c2.metric("Inicio – fin del sueño", periodo, help=textos.AYUDA["inicio_fin"])
    c3.metric("Despierto tras dormirse", minutos_a_texto(indicadores.vigilia_tras_inicio_min),
              help=textos.AYUDA["vigilia_tras_inicio"])
    c4.metric("Latencia de REM", minutos_a_texto(indicadores.latencia_rem_min) if indicadores.latencia_rem_min is not None else "Sin REM",
              help=textos.AYUDA["latencia_rem"])
    porcentaje = indicadores.epocas_baja_confianza / max(indicadores.epocas_analizadas, 1)
    c5.metric("Épocas para revisar", f"{indicadores.epocas_baja_confianza:,}".replace(",", "."),
              help=textos.AYUDA["baja_confianza"].format(umbrales=describir_umbrales(umbrales)))
    c5.caption(f"{porcentaje:.0%} de las analizadas, con confianza menor al umbral de su estadio")
    _estadios_compactos(indicadores)


def _estadios_compactos(indicadores) -> None:
    """Tiempo en cada estadio como recuadros pequeños (dato estándar de un informe de polisomnografía)."""

    clases = [c for c in ("N1", "N2", "N3", "REM") if c in indicadores.porcentaje_sueno]
    if not clases or indicadores.tiempo_total_sueno_min <= 0:
        return
    st.markdown("**Tiempo en cada estadio**", help=textos.AYUDA["distribucion"])
    colores = graficas.COLORES[tema()]

    def tarjeta(clase: str, principal: str, detalle: str) -> str:
        return (f"""<div style="flex:1;min-width:120px;border:1px solid rgba(128,128,128,.28);
                    border-left:6px solid {colores[clase]};border-radius:8px;padding:8px 12px;">
                  <div style="font-size:.82rem;opacity:.75;">{textos.NOMBRES_ESTADIO[clase]}</div>
                  <div style="font-size:1.35rem;font-weight:600;line-height:1.3;">{principal}</div>
                  <div style="font-size:.82rem;opacity:.75;">{detalle}</div>
                </div>""")

    tarjetas = "".join(
        tarjeta(c, f"{indicadores.porcentaje_sueno[c]:.0f} %", minutos_a_texto(indicadores.minutos_por_estadio[c]))
        for c in clases
    )
    # Movimiento (sexta clase): se informa en minutos, porque no cuenta como tiempo de sueño.
    if indicadores.minutos_por_estadio.get("M", 0) > 0:
        tarjetas += tarjeta("M", minutos_a_texto(indicadores.minutos_por_estadio["M"]), "no cuenta como sueño")
    st.markdown(f'<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:4px;">{tarjetas}</div>',
                unsafe_allow_html=True)
    st.caption(
        "Porcentaje del tiempo total de sueño. Vigilia dentro de la ventana analizada: "
        f"{minutos_a_texto(indicadores.minutos_por_estadio.get('W', 0))}."
    )


def _visor_senal(config, registro, resultado, seleccion) -> None:
    duracion_epoca = config["senales"]["duracion_epoca_s"]
    umbrales = config["interpretacion"]["umbrales_confianza"]
    ventana = st.session_state["analisis"]["ventana"]
    n = len(resultado.estadios)
    baja = np.flatnonzero(para_revisar(resultado.estadios, resultado.confianzas, umbrales))

    st.subheader("Señal y clasificación por época", anchor=False, help=textos.AYUDA["senal"])
    if st.session_state.get("epoca_actual") is None:
        inicial = int(baja[baja >= ventana.inicio][0]) if np.any(baja >= ventana.inicio) else ventana.inicio
        st.session_state["epoca_actual"] = inicial

    def mover(destino: int) -> None:
        st.session_state["epoca_actual"] = int(np.clip(destino, 0, n - 1))

    actual = st.session_state["epoca_actual"]
    b1, b2, b3, b4, b5 = st.columns([1, 1, 1.4, 1.4, 2])
    b1.button("◀ Anterior", on_click=mover, args=(actual - 1,), use_container_width=True)
    b2.button("Siguiente ▶", on_click=mover, args=(actual + 1,), use_container_width=True)
    previas = baja[baja < actual]
    siguientes = baja[baja > actual]
    b3.button("◀ Baja confianza", on_click=mover, args=(int(previas[-1]) if len(previas) else actual,),
              disabled=not len(previas), use_container_width=True,
              help="Va a la época anterior en la que el modelo tuvo baja confianza.")
    b4.button("Baja confianza ▶", on_click=mover, args=(int(siguientes[0]) if len(siguientes) else actual,),
              disabled=not len(siguientes), use_container_width=True,
              help="Va a la siguiente época en la que el modelo tuvo baja confianza.")
    with b5:
        contexto = st.radio("Ancho", ["30 s", "5 min"], horizontal=True, label_visibility="collapsed",
                            help="30 s muestra una sola época; 5 min muestra 10 épocas para ver el contexto.")

    # El deslizador muestra la época numerada desde 1, igual que el resto de la
    # pantalla y el CSV; internamente se guarda el índice desde 0.
    def desde_deslizador() -> None:
        st.session_state["epoca_actual"] = st.session_state["deslizador_epoca"] - 1

    st.session_state["deslizador_epoca"] = st.session_state["epoca_actual"] + 1
    st.slider(
        "Desplácese a cualquier momento del registro (número de época)", 1, n,
        key="deslizador_epoca", on_change=desde_deslizador,
    )
    actual = st.session_state["epoca_actual"]
    ancho = 1 if contexto == "30 s" else 10
    primera = int(np.clip(actual - ancho // 2, 0, max(n - ancho, 0)))

    fig = graficas.senal_con_estadios(
        registro.senales, seleccion, resultado.estadios, resultado.confianzas, primera, ancho,
        registro.frecuencia_hz, duracion_epoca, registro.inicio, tema(), umbrales, epoca_actual=actual,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

    estadio = resultado.estadios[actual]
    confianza = resultado.confianzas[actual]
    probabilidades = resultado.probabilidades[actual]
    orden = np.argsort(probabilidades)[::-1]
    segunda = resultado.clases[int(orden[1])] if len(orden) > 1 else None

    with st.container(border=True):
        st.markdown(f"**Época {actual + 1:,} · {hora_epoca(registro.inicio, actual, duracion_epoca, True)}**".replace(",", "."))
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"Clasificación del modelo  \n**{textos.NOMBRES_ESTADIO.get(estadio, estadio)}**")
        if estadio == "?":
            c2.markdown("Confianza  \n**—**")
            c3.markdown("Sin clasificar")
        else:
            c3.markdown(
                f"Segunda opción del modelo  \n**{textos.NOMBRES_ESTADIO.get(segunda, segunda)}** "
                f"({probabilidades[orden[1]]:.0%})" if segunda is not None else "Segunda opción  \n**—**",
                help=textos.AYUDA["segunda_opcion"] if segunda is not None else None,
            )
            c2.markdown(
                f"Confianza  \n**{confianza:.0%}**" + (" · ⚠ conviene revisarla" if confianza < umbrales.get(estadio, 0.0) else ""),
                help=textos.AYUDA["confianza"],
            )
        if registro.tiene_hipnograma:
            experto = registro.etiquetas_experto[actual]
            coincide = "✅ coincide" if experto == estadio else "❌ no coincide"
            st.markdown(f"Experto: **{textos.NOMBRES_ESTADIO.get(experto, experto)}** ({coincide})")
        if not (ventana.inicio <= actual < ventana.fin):
            st.caption("Esta época está fuera de la ventana analizada y no cuenta en los indicadores.")


def _concordancia(config, concordancia) -> None:
    kappa_ref = config["interpretacion"]["kappa_entre_expertos"]
    st.subheader("Comparación con el experto", anchor=False, help=textos.AYUDA["concordancia"].format(kappa_ref=kappa_ref))
    c1, c2, c3 = st.columns(3)
    c1.metric("Coincidencia con el experto", f"{concordancia.exactitud:.0%}")
    c2.metric("Kappa de Cohen", f"{concordancia.kappa:.2f}")
    c2.caption(f"Referencia entre dos expertos humanos: {kappa_ref:.2f}")
    c3.metric("Épocas comparadas", f"{concordancia.epocas_comparadas:,}".replace(",", "."))
    if concordancia.epocas_excluidas:
        st.caption(
            f"{'Se excluyó' if concordancia.epocas_excluidas == 1 else 'Se excluyeron'} "
            f"{textos.epocas(concordancia.epocas_excluidas)} que el experto marcó como movimiento "
            "o no puntuó, porque el modelo no predice esas categorías."
        )
    st.caption(textos.AYUDA["matriz"])
    st.plotly_chart(graficas.matriz_confusion(concordancia.matriz, concordancia.clases, tema()),
                    use_container_width=True, config={"displayModeBar": False})


def mostrar(config: Dict[str, Any]) -> None:
    registro: Registro = st.session_state["registro"]
    resultado = st.session_state["resultado"]
    analisis = st.session_state["analisis"]
    seleccion = st.session_state["seleccion"]
    modelo = st.session_state["modelo"]
    ventana, indicadores = analisis["ventana"], analisis["indicadores"]
    duracion_epoca = config["senales"]["duracion_epoca_s"]
    umbrales = config["interpretacion"]["umbrales_confianza"]
    n_total = len(resultado.estadios)

    cabecera, boton = st.columns([4, 1])
    cabecera.subheader("Resultado del análisis", anchor=False)
    boton.button("Analizar otro registro", on_click=reiniciar, use_container_width=True)
    if np.any(np.asarray(resultado.estadios, dtype=object) == "?"):
        st.caption(
            "Las épocas «?» del final del registro quedaron sin clasificar porque no completan "
            "una secuencia de 15 épocas, igual que en la evaluación del modelo."
        )

    _ficha(config, registro, seleccion, modelo, ventana, indicadores, n_total)
    _indicadores(config, registro, indicadores)
    datos_informe = _datos_informe(config, registro, resultado, analisis, seleccion, modelo)

    st.subheader("Hipnograma", anchor=False, help=textos.AYUDA["hipnograma"])
    ver_todo = st.toggle("Ver el registro completo", value=False,
                         help="Por defecto se muestra solo la ventana analizada.")
    inicio, fin = (0, n_total) if ver_todo else (ventana.inicio, ventana.fin)
    comparar = False
    if registro.tiene_hipnograma:
        comparar = st.toggle("Mostrar el hipnograma del experto y las coincidencias", value=True)
    tiempos = graficas.eje_tiempo(n_total, registro.inicio.replace(tzinfo=None) if registro.inicio else None, duracion_epoca)
    extra = [c for c in resultado.clases if c not in ("W", "N1", "N2", "N3", "REM")]
    st.plotly_chart(
        graficas.hipnograma(
            resultado.estadios[inicio:fin], resultado.confianzas[inicio:fin], tiempos[inicio:fin], umbrales, tema(),
            experto=registro.etiquetas_experto[inicio:fin] if comparar else None, clases_extra=extra,
            duracion_epoca_s=duracion_epoca,
        ),
        use_container_width=True, config={"displaylogo": False},
    )

    _visor_senal(config, registro, resultado, seleccion)

    if analisis["concordancia"] is not None:
        _concordancia(config, analisis["concordancia"])

    st.subheader("Descargar resultados", anchor=False, help=textos.AYUDA["descarga"])
    col_pdf, col_excel = st.columns(2)
    with col_pdf:
        st.download_button(
            "Descargar informe (PDF)",
            _pdf_cache(registro.nombre, _huella(datos_informe), datos_informe),
            file_name=f"informe_sueno_{registro.nombre}.pdf",
            mime="application/pdf", type="primary", use_container_width=True,
        )
        st.caption("Para leer, imprimir o adjuntar: resumen, hallazgos, hipnograma, comparación y épocas para revisar.")
    with col_excel:
        st.download_button(
            "Descargar datos por época (Excel)",
            _excel_cache(registro.nombre, _huella(datos_informe), datos_informe),
            file_name=f"clasificacion_{registro.nombre}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True,
        )
        st.caption("Para analizar: resumen, una fila por época con filtros, lista de épocas para revisar y guía de lectura.")
    st.caption(f"Clasificación generada en {resultado.segundos:.0f} s.")
