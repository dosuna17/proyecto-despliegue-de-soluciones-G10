"""Informe descargable del análisis: PDF (para leer o imprimir) y Excel (para analizar).

Ambos contienen lo mismo que se ve en la pantalla de resultados, más un texto
de hallazgos generado a partir de los indicadores. No depende de Streamlit:
se prueba en tests/test_informe.py.

Fuente: Century Gothic si está instalada; si no (servidor Linux), URW Gothic,
la versión libre del mismo diseño (ver fuentes/README.md).
"""

from __future__ import annotations

import io
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

import textos
from procesamiento import (
    ESTADIOS_SUENO,
    ORDEN_HIPNOGRAMA,
    Concordancia,
    Indicadores,
    Ventana,
    describir_umbrales,
    numero_combinacion,
    para_revisar,
)

CARPETA_FUENTES = Path(__file__).resolve().parent / "fuentes"
FUENTE_EXCEL = "Century Gothic"

COLORES = {"W": "#2a78d6", "N1": "#eb6834", "N2": "#1baf7a", "N3": "#eda100", "REM": "#e87ba4",
           "M": "#008300", "?": "#9b9a95"}
# Versiones claras de cada color para el fondo de las celdas del Excel.
COLORES_SUAVES = {"W": "D5E4F7", "N1": "FBE1D6", "N2": "D1F0E4", "N3": "FBECC7", "REM": "F9E0EA",
                  "M": "CCE6CC", "?": "EEEEEC"}
TINTA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"
REJILLA = "#e6e5e0"
NO_COINCIDE = "#d03b3b"
REM_TIPICA_MIN = (70, 120)  # latencia de REM habitual en adultos, en minutos


# --------------------------------------------------------------------------
# Datos del informe
# --------------------------------------------------------------------------

@dataclass
class DatosInforme:
    nombre_registro: str
    inicio: Optional[datetime]
    canales: List[str]
    modelo: str
    estadios: np.ndarray
    confianzas: np.ndarray
    probabilidades: np.ndarray
    clases: List[str]
    ventana: Ventana
    indicadores: Indicadores
    umbrales: Dict[str, float]
    kappa_referencia: float
    duracion_epoca_s: float
    experto: Optional[np.ndarray] = None
    concordancia: Optional[Concordancia] = None
    generado: datetime = field(default_factory=datetime.now)
    recorte: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_total(self) -> int:
        return len(self.estadios)

    @property
    def revisar(self) -> np.ndarray:
        """True en las épocas con confianza menor al umbral de su estadio."""
        return para_revisar(self.estadios, self.confianzas, self.umbrales)

    @property
    def umbrales_texto(self) -> str:
        return describir_umbrales(self.umbrales)

    def hora(self, epoca: int, segundos: bool = False) -> str:
        desfase = epoca * self.duracion_epoca_s
        if self.inicio is not None:
            momento = self.inicio.replace(tzinfo=None) + timedelta(seconds=desfase)
            return momento.strftime("%H:%M:%S" if segundos else "%H:%M")
        horas, resto = divmod(int(desfase), 3600)
        minutos, segs = divmod(resto, 60)
        return f"{horas}:{minutos:02d}:{segs:02d}" if segundos else f"{horas} h {minutos:02d}"

    def tiempos(self) -> List:
        if self.inicio is not None:
            base = self.inicio.replace(tzinfo=None)
            return [base + timedelta(seconds=i * self.duracion_epoca_s) for i in range(self.n_total)]
        return list(np.arange(self.n_total) * self.duracion_epoca_s / 3600.0)


def _min(minutos: Optional[float]) -> str:
    if minutos is None:
        return "—"
    horas, mins = divmod(int(round(minutos)), 60)
    return f"{horas} h {mins:02d} min" if horas else f"{mins} min"


def _pct(valor: float) -> str:
    return f"{valor * 100:.0f} %"


def _nombre(clase: str) -> str:
    """'N2 (sueño ligero)' en lugar de 'N2 · sueño ligero', para usar dentro de una frase."""

    nombre = textos.NOMBRES_ESTADIO.get(clase, clase)
    return nombre.replace(" · ", " (") + ")" if " · " in nombre else nombre


# --------------------------------------------------------------------------
# Hallazgos: texto que resume lo que muestra el análisis
# --------------------------------------------------------------------------

def hallazgos(d: DatosInforme) -> List[str]:
    """Párrafos cortos y descriptivos (no diagnósticos) sobre el análisis."""

    ind, v = d.indicadores, d.ventana
    parrafos: List[str] = []

    if v.metodo == "sin_sueno" or ind.tiempo_total_sueno_min == 0:
        parrafos.append("El modelo no encontró sueño en este registro; se analizó el registro completo.")
    else:
        texto = (f"El modelo estimó {_min(ind.tiempo_total_sueno_min)} de sueño, entre las "
                 f"{d.hora(ind.inicio_sueno_epoca)} y las {d.hora(ind.fin_sueno_epoca + 1)}.")
        if v.metodo == "prediccion":
            texto += (f" Se analizaron {_min(ind.minutos_analizados)} de los "
                      f"{_min(d.n_total * d.duracion_epoca_s / 60)} grabados: el periodo de sueño más "
                      f"{d.recorte.get('margen_min', 30)} minutos antes y después.")
        else:
            texto += " No se encontró un periodo de sueño claro, así que se analizó el registro completo."
        parrafos.append(texto)

        pct = ind.porcentaje_sueno
        principal = max(pct, key=pct.get)
        parrafos.append(
            f"El estadio más frecuente fue {_nombre(principal)}, con el {pct[principal]:.0f} % del sueño. "
            f"El sueño profundo (N3) ocupó el {pct.get('N3', 0):.0f} %, el REM el {pct.get('REM', 0):.0f} % "
            f"y el sueño ligero (N1 + N2) el {pct.get('N1', 0) + pct.get('N2', 0):.0f} %."
        )

        continuidad = f"Después de dormirse, la persona estuvo despierta {_min(ind.vigilia_tras_inicio_min)} en total."
        if ind.latencia_rem_min is None:
            continuidad += " El modelo no detectó sueño REM."
        else:
            continuidad += f" El primer sueño REM apareció {_min(ind.latencia_rem_min)} después de dormirse"
            bajo, alto = REM_TIPICA_MIN
            if bajo <= ind.latencia_rem_min <= alto:
                continuidad += f", dentro del rango habitual en adultos ({bajo} a {alto} min)."
            else:
                continuidad += (f", fuera del rango habitual en adultos ({bajo} a {alto} min); conviene "
                                "verificarlo en el hipnograma.")
        parrafos.append(continuidad)

    tramo = slice(v.inicio, v.fin)
    baja = d.revisar[tramo]
    if baja.any():
        estadios_baja = np.asarray(d.estadios[tramo], dtype=object)[baja]
        valores, cuentas = np.unique(estadios_baja, return_counts=True)
        mas = valores[int(np.argmax(cuentas))]
        parrafos.append(
            f"{textos.epocas(baja.sum())} ({_pct(baja.mean())} de las analizadas) "
            f"{'tiene' if int(baja.sum()) == 1 else 'tienen'} una confianza menor al "
            f"umbral de su estadio ({d.umbrales_texto}) y conviene revisarlas; la mayoría fueron clasificadas como "
            f"{_nombre(mas)}: {textos.epocas(cuentas.max())}. Están en la lista de épocas para revisar."
        )
    else:
        parrafos.append(f"Ninguna época analizada quedó por debajo del umbral de su estadio ({d.umbrales_texto}).")

    c = d.concordancia
    if c is not None:
        diferencia = c.kappa - d.kappa_referencia
        if abs(diferencia) <= 0.03:
            relacion = "similar al"
        elif diferencia > 0:
            relacion = "por encima del"
        else:
            relacion = "por debajo del"
        texto = (f"Frente al hipnograma del experto, el modelo coincidió en el {_pct(c.exactitud)} de "
                 f"{textos.epocas(c.epocas_comparadas)} (kappa de Cohen {c.kappa:.2f}, {relacion} acuerdo típico entre "
                 f"dos expertos humanos, {d.kappa_referencia:.2f}).")
        fuera = c.matriz.copy()
        np.fill_diagonal(fuera, 0)
        if fuera.sum():
            i, j = np.unravel_index(int(np.argmax(fuera)), fuera.shape)
            texto += (f" La diferencia más frecuente: el experto marcó {c.clases[i]} y el modelo {c.clases[j]} "
                      f"({textos.epocas(fuera[i, j])}).")
        parrafos.append(texto)

    parrafos.append("Estos resultados son una estimación automática y deben ser revisados por un profesional del sueño.")
    return parrafos


# --------------------------------------------------------------------------
# Tablas comunes
# --------------------------------------------------------------------------

def filas_indicadores(d: DatosInforme) -> List[tuple]:
    ind = d.indicadores
    if ind.inicio_sueno_epoca is not None:
        periodo = f"{d.hora(ind.inicio_sueno_epoca)} – {d.hora(ind.fin_sueno_epoca + 1)}"
    else:
        periodo = "—"
    baja = f"{ind.epocas_baja_confianza} ({_pct(ind.epocas_baja_confianza / max(ind.epocas_analizadas, 1))})"
    return [
        ("Tiempo total de sueño", _min(ind.tiempo_total_sueno_min), textos.AYUDA["tiempo_total_sueno"]),
        ("Inicio – fin del sueño", periodo, textos.AYUDA["inicio_fin"]),
        ("Despierto tras dormirse", _min(ind.vigilia_tras_inicio_min), textos.AYUDA["vigilia_tras_inicio"]),
        ("Latencia de REM", _min(ind.latencia_rem_min) if ind.latencia_rem_min is not None else "Sin REM",
         textos.AYUDA["latencia_rem"]),
        ("Épocas para revisar", baja, textos.AYUDA["baja_confianza"].format(umbrales=d.umbrales_texto)),
    ]


def filas_ficha(d: DatosInforme) -> List[tuple]:
    v = d.ventana
    return [
        ("Registro", d.nombre_registro),
        ("Señales usadas", f"{' + '.join(d.canales)} (combinación {numero_combinacion(d.canales)} de 12)"),
        ("Modelo", d.modelo),
        ("Ventana analizada", f"{d.hora(v.inicio)} – {d.hora(v.fin)} ({_min(d.indicadores.minutos_analizados)})"),
        ("Duración del registro", _min(d.n_total * d.duracion_epoca_s / 60)),
        ("Umbrales de baja confianza", d.umbrales_texto),
        ("Generado", d.generado.strftime("%Y-%m-%d %H:%M")),
    ]


def epocas_para_revisar(d: DatosInforme, maximo: Optional[int] = None) -> List[int]:
    v = d.ventana
    revisar = d.revisar
    indices = [i for i in range(v.inicio, v.fin) if revisar[i]]
    indices.sort(key=lambda i: d.confianzas[i])
    return indices[:maximo] if maximo else indices


# --------------------------------------------------------------------------
# Matplotlib: fuente e hipnograma (mismo estilo que el tablero)
# --------------------------------------------------------------------------

def _preparar_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt

    disponibles = {f.name for f in fm.fontManager.ttflist}
    if "Century Gothic" in disponibles:
        familia, negrita = "Century Gothic", None
    else:
        for archivo in CARPETA_FUENTES.glob("*.otf"):
            try:
                fm.fontManager.addfont(str(archivo))
            except Exception:  # noqa: BLE001
                pass
        familia = "URW Gothic" if "URW Gothic" in {f.name for f in fm.fontManager.ttflist} else "DejaVu Sans"
        demi = CARPETA_FUENTES / "URWGothic-Demi.otf"
        negrita = fm.FontProperties(fname=str(demi)) if demi.is_file() and familia == "URW Gothic" else None
    plt.rcParams.update({
        "font.family": familia, "font.size": 9, "axes.edgecolor": REJILLA, "axes.labelcolor": TINTA_SECUNDARIA,
        "xtick.color": TINTA_SECUNDARIA, "ytick.color": TINTA_SECUNDARIA, "axes.spines.top": False,
        "axes.spines.right": False,
    })
    return plt, negrita


def _tramos(valores: Sequence) -> List[tuple]:
    valores = list(valores)
    salida, inicio = [], 0
    for k in range(1, len(valores) + 1):
        if k == len(valores) or valores[k] != valores[inicio]:
            salida.append((inicio, k, valores[inicio]))
            inicio = k
    return salida


def dibujar_hipnograma(d: DatosInforme, figura, rect, plt) -> None:
    """Hipnograma estilo tablero: bloques de color + línea negra; experto y franja si hay hipnograma."""

    import matplotlib.dates as mdates

    v = d.ventana
    tramo = slice(v.inicio, v.fin)
    estadios = list(d.estadios[tramo])
    experto = list(d.experto[tramo]) if d.experto is not None else None
    conf = d.confianzas[tramo]
    tiempos = d.tiempos()[tramo]
    es_fecha = d.inicio is not None
    paso = timedelta(seconds=d.duracion_epoca_s) if es_fecha else d.duracion_epoca_s / 3600.0
    x = mdates.date2num(tiempos) if es_fecha else np.asarray(tiempos, dtype=float)
    ancho = (mdates.date2num(tiempos[0] + paso) - x[0]) if es_fecha else paso

    orden = list(ORDEN_HIPNOGRAMA)
    if "M" in d.clases:
        orden.append("M")  # sexta clase: solo cuando el modelo la predice (igual que en el tablero)
    posicion = {c: len(orden) - 1 - i for i, c in enumerate(orden)}
    izquierda, abajo, ancho_rect, alto_rect = rect
    if experto is not None:
        alturas = [0.42, 0.42, 0.08]
        separacion = 0.035 * alto_rect
        util = alto_rect - 2 * separacion
        ejes = []
        y = abajo + alto_rect
        for h in alturas:
            y -= h * util
            ejes.append(figura.add_axes([izquierda, y, ancho_rect, h * util]))
            y -= separacion
        titulos = ["Clasificación del modelo", "Hipnograma del experto", "¿Coinciden modelo y experto?"]
        series = [estadios, experto]
    else:
        ejes = [figura.add_axes(rect)]
        titulos = ["Clasificación del modelo"]
        series = [estadios]

    for eje, serie, titulo in zip(ejes, series, titulos):
        for a, b, clase in _tramos(serie):
            if clase in posicion:
                eje.broken_barh([(x[a], (b - a) * ancho)], (posicion[clase] - 0.32, 0.64), facecolors=COLORES[clase],
                                linewidth=0)
        ys = [posicion.get(e, np.nan) for e in serie]
        eje.step(list(x) + [x[-1] + ancho], ys + [ys[-1]], where="post", color=TINTA, linewidth=0.7)
        eje.set_yticks([posicion[c] for c in orden], [textos.NOMBRES_ESTADIO[c].split(" ·")[0] for c in orden])
        eje.set_ylim(-0.9 if eje is ejes[0] else -0.6, len(orden) - 0.45)
        eje.set_title(titulo, loc="left", fontsize=8.5, color=TINTA_SECUNDARIA, pad=3)
        eje.grid(axis="x", color=REJILLA, linewidth=0.5)
        eje.tick_params(length=0, labelsize=8)

    baja = d.revisar[tramo]
    if baja.any():
        ejes[0].vlines(x[baja] + ancho / 2, -0.85, -0.55, color=TINTA_SECUNDARIA, linewidth=0.5)

    if experto is not None:
        franja = ejes[2]
        for a, b, valor in _tramos([("x" if e not in posicion else ("si" if e == m else "no"))
                                    for e, m in zip(experto, estadios)]):
            color = {"si": REJILLA, "no": NO_COINCIDE, "x": "#ffffff"}[valor]
            franja.broken_barh([(x[a], (b - a) * ancho)], (0, 1), facecolors=color, linewidth=0)
        franja.set_yticks([])
        franja.set_ylim(0, 1)
        franja.spines["left"].set_visible(False)
        franja.tick_params(length=0, labelsize=8)

    for eje in ejes:
        eje.set_xlim(x[0], x[-1] + ancho)
        if es_fecha:
            eje.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            eje.xaxis.set_major_locator(mdates.HourLocator())
        if eje is not ejes[-1]:
            eje.tick_params(labelbottom=False)

    # Leyenda propia, arriba del primer panel
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    elementos = [Patch(facecolor=COLORES[c], label=textos.NOMBRES_ESTADIO[c].split(" ·")[0]) for c in orden]
    elementos.append(Line2D([0], [0], color=TINTA_SECUNDARIA, linewidth=0.8, label="Época para revisar"))
    if experto is not None:
        elementos.append(Patch(facecolor=NO_COINCIDE, label="No coinciden"))
    ejes[0].legend(handles=elementos, loc="lower left", bbox_to_anchor=(0, 1.08), ncol=len(elementos), frameon=False,
                   fontsize=7.5, handlelength=1.2, columnspacing=1.0)


def imagen_hipnograma(d: DatosInforme, ancho_px: int = 1500) -> bytes:
    plt, _ = _preparar_matplotlib()
    alto = 5.2 if d.experto is not None else 2.8
    figura = plt.figure(figsize=(10, alto), dpi=ancho_px / 10)
    dibujar_hipnograma(d, figura, [0.08, 0.1, 0.9, 0.72 if d.experto is not None else 0.62], plt)
    salida = io.BytesIO()
    figura.savefig(salida, format="png", facecolor="white")
    plt.close(figura)
    return salida.getvalue()


# --------------------------------------------------------------------------
# PDF
# --------------------------------------------------------------------------

_NEGRITA = {"propiedades": None}


def _texto(figura, x, y, contenido, tamano=9, color=TINTA, fuerte=False, **extra):
    """Texto en coordenadas de la página; ``fuerte`` usa la versión en negrita de la fuente."""

    if fuerte and _NEGRITA["propiedades"] is not None:
        propiedades = _NEGRITA["propiedades"].copy()
        propiedades.set_size(tamano)
        return figura.text(x, y, contenido, color=color, fontproperties=propiedades, **extra)
    return figura.text(x, y, contenido, fontsize=tamano, color=color, fontweight="bold" if fuerte else "normal", **extra)


def _parrafo(figura, x, y, contenido, ancho_caracteres, tamano=8.8, interlineado=0.0145, color=TINTA):
    lineas = textwrap.wrap(contenido, ancho_caracteres)
    for linea in lineas:
        figura.text(x, y, linea, fontsize=tamano, color=color, va="top")
        y -= interlineado
    return y


def _caja(figura, x, y, ancho, alto, color, borde=None):
    from matplotlib.patches import FancyBboxPatch
    figura.patches.append(FancyBboxPatch((x, y), ancho, alto, boxstyle="round,pad=0,rounding_size=0.006",
                                         transform=figura.transFigure, facecolor=color,
                                         edgecolor=borde or color, linewidth=0.6))


def _encabezado(figura, d: DatosInforme, pagina: int, total: int) -> None:
    _texto(figura, 0.06, 0.955, "Informe de clasificación de estadios del sueño", 15, fuerte=True)
    _texto(figura, 0.06, 0.935, f"{d.nombre_registro} · generado el {d.generado.strftime('%Y-%m-%d %H:%M')}", 8.5,
           TINTA_SECUNDARIA)
    figura.text(0.94, 0.03, f"Página {pagina} de {total}", fontsize=7.5, color=TINTA_SECUNDARIA, ha="right")
    figura.text(0.06, 0.03, "Herramienta de apoyo, no de diagnóstico.", fontsize=7.5, color=TINTA_SECUNDARIA)


def _titulo_seccion(figura, y, texto, ayuda=None):
    _texto(figura, 0.06, y, texto, 10.5, fuerte=True)
    if ayuda:
        figura.text(0.06, y - 0.017, ayuda, fontsize=7.6, color=TINTA_SECUNDARIA)


def _pagina_resumen(f, d: DatosInforme) -> None:
    _caja(f, 0.06, 0.875, 0.88, 0.045, "#fff7d6", "#f0dc8a")
    _parrafo(f, 0.075, 0.910, textos.AVISO_APOYO.replace("**", ""), 118, 8.2, 0.013, "#5c4500")

    y = 0.85
    _titulo_seccion(f, y, "Datos del análisis")
    y -= 0.024
    for etiqueta, valor in filas_ficha(d):
        f.text(0.06, y, etiqueta, fontsize=8.5, color=TINTA_SECUNDARIA)
        f.text(0.30, y, valor, fontsize=8.5, color=TINTA)
        y -= 0.0165

    y -= 0.014
    _titulo_seccion(f, y, "Resumen del sueño (estimado por el modelo)")
    y -= 0.078
    ancho = 0.88 / 5
    for k, (nombre, valor, _) in enumerate(filas_indicadores(d)):
        x0 = 0.06 + k * ancho
        _caja(f, x0 + 0.003, y, ancho - 0.006, 0.062, "#f6f6f3")
        f.text(x0 + 0.012, y + 0.044, nombre, fontsize=7.4, color=TINTA_SECUNDARIA)
        _texto(f, x0 + 0.012, y + 0.014, valor, 11 if len(valor) < 12 else 9.5, fuerte=True)

    y -= 0.035
    _titulo_seccion(f, y, "Hallazgos")
    y -= 0.022
    for parrafo in hallazgos(d):
        f.text(0.065, y, "•", fontsize=8.8, color=TINTA, va="top")
        y = _parrafo(f, 0.08, y, parrafo, 112) - 0.007

    clases = [c for c in ESTADIOS_SUENO if c in d.indicadores.porcentaje_sueno]
    if d.indicadores.tiempo_total_sueno_min > 0 and clases:
        y -= 0.02
        _titulo_seccion(f, y, "Distribución de los estadios del sueño",
                        "Porcentaje sobre el tiempo total de sueño, con sus minutos.")
        alto = min(0.16, max(y - 0.06 - 0.09, 0.1))
        eje = f.add_axes([0.22, y - 0.035 - alto, 0.6, alto])
        pct = [d.indicadores.porcentaje_sueno[c] for c in clases]
        eje.barh([textos.NOMBRES_ESTADIO[c] for c in clases], pct, color=[COLORES[c] for c in clases], height=0.6)
        for k, c in enumerate(clases):
            eje.text(pct[k] + 1, k, f"{pct[k]:.0f} % · {d.indicadores.minutos_por_estadio[c]:.0f} min", va="center",
                     fontsize=8, color=TINTA)
        eje.invert_yaxis()
        eje.set_xlim(0, max(pct + [10]) * 1.4)
        eje.set_xticks([])
        eje.spines["bottom"].set_visible(False)
        eje.tick_params(length=0, labelsize=8.5)
        f.text(0.06, y - 0.05 - alto, f"Vigilia dentro de la ventana analizada: "
               f"{_min(d.indicadores.minutos_por_estadio.get('W', 0))}.", fontsize=7.6, color=TINTA_SECUNDARIA)


def _pagina_hipnograma(f, d: DatosInforme, plt) -> None:
    y = 0.90
    _titulo_seccion(f, y, "Hipnograma de la ventana analizada",
                    "Bloques de color: estadio de cada época. Línea negra: cambios entre estadios. "
                    "Marcas grises: épocas para revisar.")
    alto = 0.38 if d.experto is not None else 0.2
    dibujar_hipnograma(d, f, [0.12, y - 0.075 - alto, 0.82, alto], plt)
    y = y - 0.075 - alto - 0.05

    c = d.concordancia
    if c is None:
        _titulo_seccion(f, y, "Cómo leer el hipnograma")
        _parrafo(f, 0.06, y - 0.022, textos.AYUDA["hipnograma"] + " Durante la noche el sueño avanza en ciclos de "
                 "unos 90 minutos (N1, N2, N3 y luego REM); en un registro típico se ven entre 4 y 6 de estos ciclos, con "
                 "más N3 al inicio de la noche y más REM al final.", 118, 8, 0.013, TINTA_SECUNDARIA)
        return
    _titulo_seccion(f, y, "Comparación con el hipnograma del experto")
    y -= 0.024
    for etiqueta, valor in (("Coincidencia", _pct(c.exactitud)), ("Kappa de Cohen", f"{c.kappa:.2f}"),
                            ("Referencia entre dos expertos humanos", f"{d.kappa_referencia:.2f}"),
                            ("Épocas comparadas", str(c.epocas_comparadas))):
        f.text(0.06, y, etiqueta, fontsize=8.5, color=TINTA_SECUNDARIA)
        f.text(0.36, y, valor, fontsize=8.5, color=TINTA)
        y -= 0.0165
    if c.epocas_excluidas:
        f.text(0.06, y, f"{'Se excluyó' if c.epocas_excluidas == 1 else 'Se excluyeron'} {textos.epocas(c.epocas_excluidas)} "
               "que el experto no puntuó o marcó como movimiento.",
               fontsize=7.6, color=TINTA_SECUNDARIA)
        y -= 0.016
    # Matriz grande: ocupa la mitad izquierda de la página hasta el pie, para que se lean los números.
    alto_matriz = max(0.18, y - 0.03 - 0.075)
    eje = f.add_axes([0.11, y - 0.03 - alto_matriz, 0.46, alto_matriz])
    filas = c.matriz.sum(axis=1, keepdims=True)
    porcentaje = np.divide(c.matriz, filas, out=np.zeros_like(c.matriz, dtype=float), where=filas > 0)
    eje.imshow(porcentaje, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(len(c.clases)):
        for j in range(len(c.clases)):
            eje.text(j, i, f"{_pct(porcentaje[i, j])}\n{textos.epocas(c.matriz[i, j])}", ha="center", va="center", fontsize=8.5,
                     color="white" if porcentaje[i, j] > 0.5 else TINTA, linespacing=1.4)
    eje.set_xticks(range(len(c.clases)), c.clases)
    eje.set_yticks(range(len(c.clases)), c.clases)
    eje.set_xlabel("Modelo", fontsize=9)
    eje.set_ylabel("Experto", fontsize=9)
    eje.xaxis.set_label_position("top")
    eje.xaxis.tick_top()
    eje.tick_params(length=0, labelsize=9)
    eje.set_xticks(np.arange(-0.5, len(c.clases)), minor=True)
    eje.set_yticks(np.arange(-0.5, len(c.clases)), minor=True)
    eje.grid(which="minor", color="white", linewidth=2)
    eje.tick_params(which="minor", length=0)
    for borde in eje.spines.values():
        borde.set_visible(False)
    _parrafo(f, 0.62, y - 0.04, textos.AYUDA["matriz"] + " Los porcentajes se calculan por fila.", 45, 7.8, 0.0125,
             TINTA_SECUNDARIA)


def _pagina_revision(f, d: DatosInforme) -> None:
    y = 0.90
    total_revisar = epocas_para_revisar(d)
    revisar = total_revisar[:80]
    _titulo_seccion(f, y, "Épocas para revisar",
                    f"Confianza menor al umbral de su estadio ({d.umbrales_texto}), de la menos a la más confiable."
                    + (f" Se muestran {len(revisar)} de {len(total_revisar)}; la lista completa está en el Excel."
                       if len(total_revisar) > len(revisar) else ""))
    y -= 0.045
    if not revisar:
        f.text(0.06, y, "No hay épocas con confianza menor al umbral.", fontsize=8.5)
        y -= 0.03
    else:
        columnas = ["Época", "Hora", "Modelo", "Confianza"] + (["Experto"] if d.experto is not None else [])
        xs = [0.0, 0.065, 0.15, 0.225, 0.32]
        por_columna = 40
        alto_fila = 0.0145
        for bloque_inicio, x0 in ((0, 0.06), (por_columna, 0.53)):
            bloque = revisar[bloque_inicio:bloque_inicio + por_columna]
            if not bloque:
                continue
            yy = y
            for k, titulo in enumerate(columnas):
                _texto(f, x0 + xs[k], yy, titulo, 7.8, TINTA_SECUNDARIA, fuerte=True)
            yy -= alto_fila
            for i in bloque:
                valores = [str(i + 1), d.hora(i, True), str(d.estadios[i]), _pct(d.confianzas[i])]
                if d.experto is not None:
                    valores.append(str(d.experto[i]))
                for k, valor in enumerate(valores):
                    f.text(x0 + xs[k], yy, valor, fontsize=7.8, color=TINTA)
                yy -= alto_fila
        y -= alto_fila * (min(len(revisar), por_columna) + 1) + 0.03

    _titulo_seccion(f, y, "Cómo se generó este informe")
    metodo = (
        f"Cada época de 30 segundos se clasificó con el modelo {d.modelo} usando {' + '.join(d.canales)}. "
        f"Luego se ubicó el periodo principal de sueño a partir de la propia clasificación: solo cuentan los tramos de "
        f"sueño de al menos {d.recorte.get('min_bloque_sueno_min', 5)} min seguidos, y los despertares de menos de "
        f"{d.recorte.get('max_interrupcion_min', 60)} min se consideran parte de la noche. A ese periodo se le añadieron "
        f"{d.recorte.get('margen_min', 30)} min antes y después, como en el entrenamiento del modelo. Los indicadores "
        f"se calculan dentro de esa ventana. Una época se marca para revisar cuando la confianza del modelo es menor "
        f"al umbral del estadio que eligió ({d.umbrales_texto}); los umbrales se fijaron en la validación cruzada "
        f"para que las épocas no marcadas de cada estadio tengan una precisión cercana a la del estado del arte."
    )
    _parrafo(f, 0.06, y - 0.022, metodo, 118, 8, 0.013, TINTA_SECUNDARIA)


def generar_pdf(d: DatosInforme) -> bytes:
    """Informe de 3 páginas: resumen, hipnograma y comparación, épocas para revisar."""

    plt, negrita = _preparar_matplotlib()
    _NEGRITA["propiedades"] = negrita
    from matplotlib.backends.backend_pdf import PdfPages

    salida = io.BytesIO()
    paginas = [lambda f: _pagina_resumen(f, d), lambda f: _pagina_hipnograma(f, d, plt), lambda f: _pagina_revision(f, d)]
    metadatos = {"Title": f"Informe de sueño · {d.nombre_registro}", "Creator": "Tablero de clasificación del sueño"}
    with PdfPages(salida, metadata=metadatos) as pdf:
        for numero, dibujar in enumerate(paginas, start=1):
            figura = plt.figure(figsize=(8.27, 11.69))
            _encabezado(figura, d, numero, len(paginas))
            dibujar(figura)
            pdf.savefig(figura)
            plt.close(figura)
    return salida.getvalue()


# --------------------------------------------------------------------------
# Excel
# --------------------------------------------------------------------------

def generar_excel(d: DatosInforme) -> bytes:
    from openpyxl import Workbook
    from openpyxl.drawing.image import Image as ImagenExcel
    from openpyxl.formatting.rule import CellIsRule, DataBarRule, FormulaRule
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.table import Table, TableStyleInfo

    libro = Workbook()
    normal = libro._named_styles["Normal"]
    normal.font = Font(name=FUENTE_EXCEL, size=10)

    def fuente(**extra):
        return Font(name=FUENTE_EXCEL, **{"size": 10, **extra})

    titulo = fuente(size=16, bold=True)
    subtitulo = fuente(size=12, bold=True, color="0B0B0B")
    gris = fuente(color="52514E")
    negrita = fuente(bold=True)
    envolver = Alignment(wrap_text=True, vertical="top")
    encabezado_relleno = PatternFill("solid", fgColor="2A78D6")
    encabezado_fuente = fuente(bold=True, color="FFFFFF")
    linea = Side(style="thin", color="E6E5E0")
    borde = Border(bottom=linea)

    def encabezados(hoja, fila, nombres, columna=1):
        for k, nombre in enumerate(nombres):
            celda = hoja.cell(fila, columna + k, nombre)
            celda.font, celda.fill = encabezado_fuente, encabezado_relleno
            celda.alignment = Alignment(vertical="center", wrap_text=True)

    # ---------------- Hoja 1: Resumen ----------------
    h = libro.active
    h.title = "Resumen"
    h.sheet_view.showGridLines = False
    for columna, ancho in zip("ABCDEFGH", (28, 22, 16, 16, 16, 16, 16, 16)):
        h.column_dimensions[columna].width = ancho

    h["A1"] = "Informe de clasificación de estadios del sueño"
    h["A1"].font = titulo
    h["A2"] = f"{d.nombre_registro} · generado el {d.generado.strftime('%Y-%m-%d %H:%M')}"
    h["A2"].font = gris
    h.merge_cells("A3:H3")
    h["A3"] = textos.AVISO_APOYO.replace("**", "")
    h["A3"].font = fuente(color="5C4500")
    h["A3"].fill = PatternFill("solid", fgColor="FFF7D6")
    h["A3"].alignment = envolver
    h.row_dimensions[3].height = 32

    fila = 5
    h.cell(fila, 1, "Hallazgos").font = subtitulo
    fila += 1
    for parrafo in hallazgos(d):
        h.merge_cells(start_row=fila, start_column=1, end_row=fila, end_column=8)
        celda = h.cell(fila, 1, "• " + parrafo)
        celda.alignment = envolver
        h.row_dimensions[fila].height = 14 * (len(parrafo) // 150 + 1) + 4
        fila += 1

    fila += 1
    h.cell(fila, 1, "Datos del análisis").font = subtitulo
    fila += 1
    for etiqueta, valor in filas_ficha(d):
        h.cell(fila, 1, etiqueta).font = gris
        h.merge_cells(start_row=fila, start_column=2, end_row=fila, end_column=8)
        h.cell(fila, 2, valor)
        for col in range(1, 9):
            h.cell(fila, col).border = borde
        fila += 1

    fila += 1
    h.cell(fila, 1, "Resumen del sueño (estimado por el modelo)").font = subtitulo
    fila += 1
    encabezados(h, fila, ["Indicador", "Valor", "Qué mide"])
    h.merge_cells(start_row=fila, start_column=3, end_row=fila, end_column=8)
    fila += 1
    for nombre, valor, ayuda in filas_indicadores(d):
        h.cell(fila, 1, nombre).font = negrita
        h.cell(fila, 2, valor)
        h.merge_cells(start_row=fila, start_column=3, end_row=fila, end_column=8)
        h.cell(fila, 3, ayuda).alignment = envolver
        h.cell(fila, 1).alignment = h.cell(fila, 2).alignment = Alignment(vertical="top")
        h.row_dimensions[fila].height = 14 * (len(ayuda) // 110 + 1) + 4
        for col in range(1, 9):
            h.cell(fila, col).border = borde
        fila += 1

    fila += 1
    h.cell(fila, 1, "Distribución de los estadios del sueño").font = subtitulo
    fila += 1
    encabezados(h, fila, ["Estadio", "Minutos", "% del sueño"])
    fila += 1
    inicio_dist = fila
    for clase in ESTADIOS_SUENO:
        if clase not in d.indicadores.porcentaje_sueno:
            continue
        h.cell(fila, 1, textos.NOMBRES_ESTADIO[clase]).fill = PatternFill("solid", fgColor=COLORES_SUAVES[clase])
        h.cell(fila, 2, round(d.indicadores.minutos_por_estadio[clase], 1))
        celda = h.cell(fila, 3, d.indicadores.porcentaje_sueno[clase] / 100)
        celda.number_format = "0%"
        fila += 1
    h.conditional_formatting.add(f"C{inicio_dist}:C{fila - 1}",
                                 DataBarRule(start_type="num", start_value=0, end_type="num", end_value=1, color="5598E7"))
    h.cell(fila, 1, f"Vigilia dentro de la ventana: {_min(d.indicadores.minutos_por_estadio.get('W', 0))}.").font = gris
    fila += 2

    c = d.concordancia
    if c is not None:
        h.cell(fila, 1, "Comparación con el hipnograma del experto").font = subtitulo
        fila += 1
        for etiqueta, valor, formato in (("Coincidencia", c.exactitud, "0%"), ("Kappa de Cohen", round(c.kappa, 3), "0.00"),
                                         ("Referencia entre dos expertos humanos", d.kappa_referencia, "0.00"),
                                         ("Épocas comparadas", c.epocas_comparadas, "0")):
            h.cell(fila, 1, etiqueta).font = gris
            celda = h.cell(fila, 2, valor)
            celda.number_format = formato
            fila += 1
        fila += 1
        h.cell(fila, 1, "Matriz de confusión: filas = experto, columnas = modelo (número de épocas)").font = negrita
        fila += 1
        encabezados(h, fila, ["Experto \\ Modelo"] + list(c.clases))
        fila += 1
        inicio_matriz = fila
        for i, clase in enumerate(c.clases):
            h.cell(fila, 1, clase).font = negrita
            for j in range(len(c.clases)):
                celda = h.cell(fila, 2 + j, int(c.matriz[i, j]))
                celda.alignment = Alignment(horizontal="center")
                if i == j:
                    celda.font = negrita
            fila += 1
        from openpyxl.formatting.rule import ColorScaleRule
        rango = f"B{inicio_matriz}:{get_column_letter(1 + len(c.clases))}{fila - 1}"
        h.conditional_formatting.add(rango, ColorScaleRule(start_type="min", start_color="FFFFFF",
                                                           end_type="max", end_color="5598E7"))
        h.merge_cells(start_row=fila, start_column=1, end_row=fila, end_column=8)
        h.cell(fila, 1, textos.AYUDA["matriz"]).font = gris
        h.cell(fila, 1).alignment = envolver
        h.row_dimensions[fila].height = 30
        fila += 2

    h.cell(fila, 1, "Hipnograma de la ventana analizada").font = subtitulo
    fila += 1
    imagen = ImagenExcel(io.BytesIO(imagen_hipnograma(d, ancho_px=1500)))
    imagen.width, imagen.height = 900, int(900 * (0.52 if d.experto is not None else 0.28))
    h.add_image(imagen, f"A{fila}")

    # ---------------- Hoja 2: Épocas ----------------
    e = libro.create_sheet("Épocas")
    e.sheet_view.showGridLines = False
    columnas = ["Época", "Hora", "Estadio (modelo)", "Confianza", "¿Revisar?", "En ventana analizada"]
    columnas += [f"Prob. {clase}" for clase in d.clases]
    if d.experto is not None:
        columnas += ["Estadio (experto)", "¿Coincide?"]
    encabezados(e, 1, columnas)
    e.row_dimensions[1].height = 30
    revisar = d.revisar
    for i in range(d.n_total):
        en_ventana = d.ventana.inicio <= i < d.ventana.fin
        valores = [i + 1, d.hora(i, True), str(d.estadios[i]), float(d.confianzas[i]),
                   "Sí" if revisar[i] else "No", "Sí" if en_ventana else "No"]
        valores += [float(p) for p in d.probabilidades[i]]
        if d.experto is not None:
            exp = str(d.experto[i])
            valores += [exp, "—" if exp not in d.clases else ("Sí" if exp == d.estadios[i] else "No")]
        e.append(valores)
        e.cell(i + 2, 3).fill = PatternFill("solid", fgColor=COLORES_SUAVES.get(str(d.estadios[i]), "FFFFFF"))
        if d.experto is not None:
            e.cell(i + 2, len(columnas) - 1).fill = PatternFill("solid", fgColor=COLORES_SUAVES.get(str(d.experto[i]), "FFFFFF"))
    ultima = d.n_total + 1
    for col in range(4, 7 + len(d.clases)):
        if col == 4 or col >= 7:
            for fila_e in range(2, ultima + 1):
                e.cell(fila_e, col).number_format = "0%"
    anchos = [8, 10, 16, 11, 10, 12] + [9] * len(d.clases) + ([16, 11] if d.experto is not None else [])
    for k, ancho in enumerate(anchos, start=1):
        e.column_dimensions[get_column_letter(k)].width = ancho
    e.freeze_panes = "D2"
    tabla = Table(displayName="Epocas", ref=f"A1:{get_column_letter(len(columnas))}{ultima}")
    tabla.tableStyleInfo = TableStyleInfo(name="TableStyleLight1", showRowStripes=False)
    e.add_table(tabla)
    e.conditional_formatting.add(f"E2:E{ultima}", CellIsRule(operator="equal", formula=['"Sí"'],
                                                             font=Font(name=FUENTE_EXCEL, bold=True, color="9C5700"),
                                                             fill=PatternFill("solid", fgColor="FFEFC2")))
    if d.experto is not None:
        col_coincide = get_column_letter(len(columnas))
        e.conditional_formatting.add(f"{col_coincide}2:{col_coincide}{ultima}",
                                     CellIsRule(operator="equal", formula=['"No"'],
                                                font=Font(name=FUENTE_EXCEL, bold=True, color="9C0006"),
                                                fill=PatternFill("solid", fgColor="FFD7D5")))
    e.conditional_formatting.add(f"A2:{get_column_letter(len(columnas))}{ultima}",
                                 FormulaRule(formula=['$F2="No"'], font=Font(name=FUENTE_EXCEL, color="9B9A95")))

    # ---------------- Hoja 3: Para revisar ----------------
    r = libro.create_sheet("Para revisar")
    r.sheet_view.showGridLines = False
    r["A1"] = f"Épocas con confianza menor al umbral de su estadio ({d.umbrales_texto}), de la menos a la más confiable"
    r["A1"].font = subtitulo
    r["A2"] = ("Son las épocas en las que el modelo dudó más. Revíselas en el tablero (sección «Señal y clasificación "
               "por época») o en su software de polisomnografía.")
    r["A2"].font = gris
    columnas_r = ["Época", "Hora", "Estadio (modelo)", "Confianza", "Segunda opción del modelo", "Prob. segunda opción"]
    if d.experto is not None:
        columnas_r.append("Estadio (experto)")
    encabezados(r, 4, columnas_r)
    for fila_r, i in enumerate(epocas_para_revisar(d), start=5):
        orden_prob = np.argsort(d.probabilidades[i])[::-1]
        segunda = d.clases[int(orden_prob[1])] if len(orden_prob) > 1 else "—"
        valores = [i + 1, d.hora(i, True), str(d.estadios[i]), float(d.confianzas[i]), segunda,
                   float(d.probabilidades[i][orden_prob[1]]) if len(orden_prob) > 1 else 0.0]
        if d.experto is not None:
            valores.append(str(d.experto[i]))
        for k, valor in enumerate(valores, start=1):
            celda = r.cell(fila_r, k, valor)
            celda.border = borde
            if k in (4, 6):
                celda.number_format = "0%"
        r.cell(fila_r, 3).fill = PatternFill("solid", fgColor=COLORES_SUAVES.get(str(d.estadios[i]), "FFFFFF"))
    for k, ancho in enumerate([8, 11, 16, 11, 22, 18, 16], start=1):
        r.column_dimensions[get_column_letter(k)].width = ancho
    r.freeze_panes = "A5"

    # ---------------- Hoja 4: Cómo leer ----------------
    g = libro.create_sheet("Cómo leer")
    g.sheet_view.showGridLines = False
    g.column_dimensions["A"].width = 26
    g.column_dimensions["B"].width = 95
    g["A1"] = "Cómo leer este archivo"
    g["A1"].font = titulo
    explicaciones = [
        ("Época", "Periodo de 30 segundos. Es la unidad que clasifica el modelo y que usan los expertos."),
        ("Hora", "Hora del reloj en que empieza la época, según la grabación."),
        ("Estadio (modelo)", "W = vigilia, N1 y N2 = sueño ligero, N3 = sueño profundo, REM = sueño con movimientos "
                             "oculares rápidos."),
        ("Confianza", textos.AYUDA["confianza"]),
        ("¿Revisar?", f"«Sí» cuando la confianza es menor al umbral del estadio del modelo ({d.umbrales_texto})."),
        ("En ventana analizada", textos.AYUDA["ventana"]),
        ("Prob. W … Prob. REM", "Probabilidad que el modelo asignó a cada estadio en esa época; suman 100 %."),
        ("Estadio (experto)", "Etiqueta del hipnograma del experto, si se cargó. «?» = época no puntuada."),
        ("¿Coincide?", "«Sí» si modelo y experto eligieron el mismo estadio. «—» si el experto no la puntuó."),
        ("Kappa de Cohen", textos.AYUDA["concordancia"].format(kappa_ref=d.kappa_referencia)),
        ("Aviso", textos.AVISO_APOYO.replace("**", "")),
    ]
    for fila_g, (nombre, texto) in enumerate(explicaciones, start=3):
        g.cell(fila_g, 1, nombre).font = negrita
        g.cell(fila_g, 1).alignment = Alignment(vertical="top")
        g.cell(fila_g, 2, texto).alignment = envolver
        g.row_dimensions[fila_g].height = 15 * (len(texto) // 105 + 1) + 4

    # Century Gothic en todas las celdas (openpyxl no aplica el estilo Normal a las celdas ya creadas).
    from copy import copy
    for hoja in libro.worksheets:
        for fila_celdas in hoja.iter_rows():
            for celda in fila_celdas:
                if celda.font is None or celda.font.name != FUENTE_EXCEL:
                    nueva = copy(celda.font)
                    nueva.name = FUENTE_EXCEL
                    celda.font = nueva

    # Impresión: cada hoja cabe a lo ancho de una página horizontal.
    for hoja in libro.worksheets:
        hoja.page_setup.orientation = "landscape"
        hoja.page_setup.paperSize = hoja.PAPERSIZE_A4
        hoja.page_setup.fitToWidth = 1
        hoja.page_setup.fitToHeight = 0
        hoja.sheet_properties.pageSetUpPr.fitToPage = True
        hoja.print_options.horizontalCentered = True
    e.print_title_rows = "1:1"

    salida = io.BytesIO()
    libro.save(salida)
    return salida.getvalue()
