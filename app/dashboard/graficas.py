"""Figuras del tablero (Plotly).

Colores por estadio: las cinco primeras posiciones de una paleta categórica
validada para daltonismo en el orden W, N1, N2, N3, REM (orden en que se
suceden en la noche, que es también el orden en que quedan adyacentes). El
color nunca es la única pista: el hipnograma codifica el estadio por su
posición vertical y en la vista de señal cada franja lleva su etiqueta.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence

import numpy as np
import plotly.graph_objects as go

from procesamiento import ORDEN_HIPNOGRAMA, para_revisar
from textos import NOMBRES_ESTADIO

COLORES = {
    "claro": {"W": "#2a78d6", "N1": "#eb6834", "N2": "#1baf7a", "N3": "#eda100", "REM": "#e87ba4",
              "M": "#008300", "?": "#9b9a95"},
    "oscuro": {"W": "#3987e5", "N1": "#d95926", "N2": "#199e70", "N3": "#c98500", "REM": "#d55181",
               "M": "#008300", "?": "#6f6e69"},
}
TINTA = {"claro": {"primaria": "#0b0b0b", "secundaria": "#52514e", "rejilla": "#e6e5e0"},
         "oscuro": {"primaria": "#f0efec", "secundaria": "#c3c2b7", "rejilla": "#3a3936"}}


def _base(fig: go.Figure, tema: str, alto: int) -> go.Figure:
    tinta = TINTA[tema]
    fig.update_layout(
        height=alto,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=tinta["primaria"], size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(color=tinta["secundaria"])),
        hoverlabel=dict(font_size=13),
    )
    fig.update_xaxes(gridcolor=tinta["rejilla"], linecolor=tinta["rejilla"], tickfont=dict(color=tinta["secundaria"]))
    fig.update_yaxes(gridcolor=tinta["rejilla"], linecolor=tinta["rejilla"], tickfont=dict(color=tinta["secundaria"]))
    return fig


def eje_tiempo(n_epocas: int, inicio: Optional[datetime], duracion_epoca_s: float) -> List:
    """Hora del reloj de cada época si se conoce el inicio; si no, horas desde el inicio."""

    if inicio is not None:
        return [inicio + timedelta(seconds=i * duracion_epoca_s) for i in range(n_epocas)]
    return list(np.arange(n_epocas) * duracion_epoca_s / 3600.0)


def tramos(estadios: Sequence) -> List[tuple]:
    """Agrupa épocas consecutivas iguales: [(inicio, fin, valor)] con fin excluido."""

    valores = list(estadios)
    salida, inicio = [], 0
    for k in range(1, len(valores) + 1):
        if k == len(valores) or valores[k] != valores[inicio]:
            salida.append((inicio, k, valores[inicio]))
            inicio = k
    return salida


def _bloques(fig, estadios, tiempos, duracion, posicion, colores, fila, nombre_serie, es_fecha):
    """Un bloque de color por tramo, agrupado en una traza por estadio (pocas trazas = gráfica fluida)."""

    escala = 1000.0 * duracion if es_fecha else duracion / 3600.0  # ms en eje de fecha, horas si no
    for clase in posicion:
        seleccion = [(a, b) for a, b, c in tramos(estadios) if c == clase]
        if not seleccion:
            continue
        inicio = [tiempos[a] for a, _ in seleccion]
        largo = [(b - a) * escala for a, b in seleccion]
        minutos = [(b - a) * duracion / 60 for a, b in seleccion]
        fig.add_trace(go.Bar(
            base=inicio, x=largo, y=[posicion[clase]] * len(seleccion), orientation="h", width=0.62,
            marker=dict(color=colores.get(clase, "#999999"), line=dict(width=0)),
            name=NOMBRES_ESTADIO.get(clase, clase), legendgroup=clase, showlegend=False,
            customdata=minutos,
            hovertemplate=f"{nombre_serie}: {NOMBRES_ESTADIO.get(clase, clase)} · %{{customdata:.1f}} min<extra></extra>",
        ), row=fila, col=1)


def _linea(fig, estadios, tiempos, duracion, posicion, color, fila, es_fecha):
    """Línea escalonada negra que une los estadios (cómo se mueve la persona entre ellos)."""

    y = [posicion.get(e, None) for e in estadios]
    if es_fecha:
        fin = tiempos[-1] + timedelta(seconds=duracion)
    else:
        fin = tiempos[-1] + duracion / 3600.0
    fig.add_trace(go.Scatter(
        x=list(tiempos) + [fin], y=y + [y[-1]], mode="lines", line=dict(shape="hv", width=1.3, color=color),
        showlegend=False, hoverinfo="skip",
    ), row=fila, col=1)


def hipnograma(
    estadios: Sequence[str],
    confianzas: Sequence[float],
    tiempos: Sequence,
    umbrales: Dict[str, float],
    tema: str,
    experto: Optional[Sequence[str]] = None,
    clases_extra: Sequence[str] = (),
    duracion_epoca_s: float = 30.0,
    titulo_modelo: str = "Clasificación del modelo",
) -> go.Figure:
    """Hipnograma con bloques de color por estadio y línea escalonada.

    Panel 1: modelo, con una marca gris por época de baja confianza.
    Si hay experto: panel 2 con su hipnograma y una franja que marca en rojo
    las épocas en que modelo y experto no coinciden.
    """

    from plotly.subplots import make_subplots

    tinta = TINTA[tema]
    colores = COLORES[tema]
    orden = list(ORDEN_HIPNOGRAMA) + [c for c in clases_extra if c not in ORDEN_HIPNOGRAMA]
    posicion = {c: len(orden) - 1 - i for i, c in enumerate(orden)}  # W arriba, N3 abajo
    es_fecha = bool(len(tiempos)) and isinstance(tiempos[0], datetime)
    estadios = list(estadios)
    confianzas = np.asarray(confianzas, dtype=float)

    if experto is not None:
        experto = list(experto)
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07, row_heights=[0.44, 0.44, 0.12],
                            subplot_titles=(titulo_modelo, "Hipnograma del experto", "¿Coinciden modelo y experto?"))
    else:
        fig = make_subplots(rows=1, cols=1, subplot_titles=(titulo_modelo,))

    # Leyenda: una entrada por estadio, en el orden del eje.
    for clase in orden:
        if clase in estadios or (experto is not None and clase in experto):
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=NOMBRES_ESTADIO.get(clase, clase).split(" ·")[0],
                                     marker=dict(symbol="square", size=12, color=colores.get(clase, "#999"))),
                          row=1, col=1)

    _bloques(fig, estadios, tiempos, duracion_epoca_s, posicion, colores, 1, "Modelo", es_fecha)
    _linea(fig, estadios, tiempos, duracion_epoca_s, posicion, tinta["primaria"], 1, es_fecha)

    baja = para_revisar(estadios, confianzas, umbrales)
    if baja.any():
        texto = [f"{NOMBRES_ESTADIO.get(e, e)} · confianza {c:.0%}" for e, c in zip(estadios, confianzas)]
        fig.add_trace(go.Scatter(
            x=np.asarray(tiempos, dtype=object)[baja], y=np.full(int(baja.sum()), -0.78), mode="markers",
            marker=dict(symbol="line-ns", size=9, line=dict(width=2, color=tinta["secundaria"])),
            name="Época para revisar (confianza menor al umbral de su estadio)",
            text=np.asarray(texto, dtype=object)[baja], hovertemplate="%{x}<br>%{text}<extra></extra>",
        ), row=1, col=1)

    ejes_y = dict(tickvals=[posicion[c] for c in orden], ticktext=[NOMBRES_ESTADIO.get(c, c).split(" ·")[0] for c in orden],
                  range=[-1.0, len(orden) - 0.45], showgrid=False, fixedrange=True)
    fig.update_yaxes(**ejes_y, row=1, col=1)

    if experto is not None:
        _bloques(fig, experto, tiempos, duracion_epoca_s, posicion, colores, 2, "Experto", es_fecha)
        _linea(fig, experto, tiempos, duracion_epoca_s, posicion, tinta["primaria"], 2, es_fecha)
        fig.update_yaxes(**{**ejes_y, "range": [-0.6, len(orden) - 0.45]}, row=2, col=1)
        validas = np.array([e in posicion for e in experto])
        coincide = np.where(~validas, "excluida", np.where(np.array(estadios, dtype=object) == np.array(experto, dtype=object),
                                                           "si", "no"))
        escala = 1000.0 * duracion_epoca_s if es_fecha else duracion_epoca_s / 3600.0
        estilos = {"si": ("Coinciden", tinta["rejilla"]), "no": ("No coinciden", "#d03b3b"),
                   "excluida": ("Sin puntuar por el experto", "#ffffff" if tema == "claro" else "#1e1e1c")}
        for valor, (nombre, color) in estilos.items():
            seleccion = [(a, b) for a, b, c in tramos(coincide) if c == valor]
            if not seleccion:
                continue
            fig.add_trace(go.Bar(
                base=[tiempos[a] for a, _ in seleccion], x=[(b - a) * escala for a, b in seleccion],
                y=[0.5] * len(seleccion), orientation="h", width=1.0, marker=dict(color=color, line=dict(width=0)),
                name=nombre, showlegend=valor == "no", hovertemplate=f"{nombre}<extra></extra>",
            ), row=3, col=1)
        fig.update_yaxes(showticklabels=False, showgrid=False, range=[0, 1], fixedrange=True, row=3, col=1)

    alto = 640 if experto is not None else 330
    _base(fig, tema, alto)
    # La leyenda va por encima del título del primer panel.
    fig.update_layout(barmode="overlay", bargap=0, margin=dict(l=10, r=10, t=75, b=10),
                      legend=dict(orientation="h", yanchor="bottom", x=0, traceorder="normal",
                                  y=1.045 if experto is not None else 1.12))
    for anotacion in fig.layout.annotations:
        anotacion.update(x=0, xanchor="left", font=dict(size=13, color=tinta["secundaria"]))
    if es_fecha:
        fig.update_xaxes(type="date", tickformat="%H:%M")
    else:
        fig.update_xaxes(ticksuffix=" h")
    return fig


# Un color oscuro por señal, distinto de los colores de estadio, para que las
# trazas se distingan entre sí sin confundirse con la franja de estadios.
COLORES_SENAL = {
    "claro": {"EEG Fpz-Cz": "#1f4e79", "EEG Pz-Oz": "#6b3fa0", "EOG horizontal": "#0e6f6b",
              "EMG submental": "#8a4b0f"},
    "oscuro": {"EEG Fpz-Cz": "#8cb8e8", "EEG Pz-Oz": "#c3a6ef", "EOG horizontal": "#7fd3cc",
               "EMG submental": "#e7b27a"},
}


def senal_con_estadios(
    senales: Dict[str, np.ndarray],
    canales: Sequence[str],
    estadios: Sequence[str],
    confianzas: Sequence[float],
    primera_epoca: int,
    n_epocas: int,
    frecuencia: int,
    duracion_epoca_s: float,
    inicio: Optional[datetime],
    tema: str,
    umbrales: Dict[str, float],
    epoca_actual: Optional[int] = None,
) -> go.Figure:
    """Señales apiladas sobre fondo limpio, con el estadio de cada época en una franja superior.

    - Cada señal tiene su color y su nombre a la izquierda.
    - Líneas grises verticales separan las épocas de 30 s.
    - La franja superior muestra el estadio (color + nombre, ⚠ si la confianza es baja).
    - En la vista de varias épocas, la época seleccionada se resalta con un fondo gris suave.
    """

    tinta = TINTA[tema]
    colores_senal = COLORES_SENAL[tema]
    muestras = int(frecuencia * duracion_epoca_s)
    a, b = primera_epoca * muestras, (primera_epoca + n_epocas) * muestras
    paso = max(1, (b - a) // 6000)
    segundos = np.arange(a, b, paso) / frecuencia
    base = inicio.replace(tzinfo=None) if inicio is not None else None

    def eje_x(seg):
        if base is not None:
            return base + timedelta(seconds=float(seg))
        return float(seg) / 3600.0

    x = [eje_x(v) for v in segundos] if base is not None else segundos / 3600.0

    escalas = [float(np.percentile(np.abs(senales[c][a:b][::paso]), 99) or 1.0) for c in canales]
    separacion = 2.4 * max(escalas) if escalas else 1.0
    alto_franja = 0.38 * separacion
    techo = 0.5 * separacion + alto_franja * 1.25

    fig = go.Figure()
    for k, canal in enumerate(canales):
        tramo = senales[canal][a:b][::paso]
        desplazamiento = -k * separacion
        color = colores_senal.get(canal, tinta["primaria"])
        fig.add_trace(go.Scattergl(
            x=x, y=tramo + desplazamiento, mode="lines", line=dict(width=1, color=color),
            name=canal, customdata=tramo, hovertemplate=f"{canal}: %{{customdata:.1f}} µV<extra></extra>",
            showlegend=False,
        ))
        fig.add_annotation(
            x=0, xref="paper", y=desplazamiento + separacion * 0.42, text=f"<b style='color:{color}'>━</b> {canal}",
            showarrow=False, xanchor="left", font=dict(size=12, color=tinta["secundaria"]),
            bgcolor="rgba(255,255,255,0.75)" if tema == "claro" else "rgba(14,17,23,0.75)",
        )

    y_franja0 = 0.5 * separacion
    y_franja1 = y_franja0 + alto_franja
    revisar = para_revisar(estadios, confianzas, umbrales)
    for i in range(primera_epoca, min(primera_epoca + n_epocas, len(estadios))):
        x0, x1 = eje_x(i * duracion_epoca_s), eje_x((i + 1) * duracion_epoca_s)
        estadio = estadios[i]
        if epoca_actual is not None and n_epocas > 1 and i == epoca_actual:
            fig.add_shape(type="rect", x0=x0, x1=x1, y0=-(len(canales) - 0.5) * separacion, y1=y_franja0,
                          fillcolor=tinta["rejilla"], opacity=0.55, line_width=0, layer="below")
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y_franja0, y1=y_franja1,
                      fillcolor=COLORES[tema].get(estadio, "#999999"), line=dict(width=1, color="#ffffff" if tema == "claro" else "#0e1117"),
                      layer="above")
        fig.add_annotation(
            x=x0 + (x1 - x0) / 2, y=(y_franja0 + y_franja1) / 2, text=estadio + (" ⚠" if revisar[i] else ""),
            showarrow=False, font=dict(size=12, color="#ffffff"),
        )
        if i > primera_epoca:
            fig.add_shape(type="line", x0=x0, x1=x0, y0=-(len(canales) - 0.5) * separacion, y1=y_franja0,
                          line=dict(width=1, color=tinta["rejilla"]), layer="below")

    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False,
                     range=[-(len(canales) - 0.5) * separacion, techo])
    fig.update_xaxes(showgrid=False, tickformat="%H:%M:%S" if base is not None else None,
                     ticksuffix="" if base is not None else " h")
    figura = _base(fig, tema, 150 + 115 * len(canales))
    figura.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    return figura


def matriz_confusion(matriz: np.ndarray, clases: Sequence[str], tema: str) -> go.Figure:
    filas = matriz.sum(axis=1, keepdims=True)
    porcentaje = np.divide(matriz, filas, out=np.zeros_like(matriz, dtype=float), where=filas > 0) * 100
    def color_texto(valor: float) -> str:
        oscuro_fondo = valor >= 50 if tema == "claro" else valor < 50
        return "#ffffff" if oscuro_fondo else TINTA[tema]["primaria"]

    texto = [[f"<span style='color:{color_texto(porcentaje[i, j])}'><b>{porcentaje[i, j]:.0f} %</b>"
              f"<br>{matriz[i, j]} épocas</span>"
              for j in range(len(clases))] for i in range(len(clases))]
    escala = [[0, "#cde2fb"], [0.5, "#5598e7"], [1, "#104281"]] if tema == "claro" else \
        [[0, "#1f2a38"], [0.5, "#256abf"], [1, "#86b6ef"]]
    fig = go.Figure(go.Heatmap(
        z=porcentaje, x=list(clases), y=list(clases), colorscale=escala, zmin=0, zmax=100,
        text=texto, texttemplate="%{text}", textfont=dict(size=14), showscale=False, xgap=2, ygap=2,
        hovertemplate="Experto %{y} → modelo %{x}: %{z:.0f} %<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", title="Experto", fixedrange=True)
    fig.update_xaxes(title="Modelo", side="top", fixedrange=True)
    return _base(fig, tema, 340)
