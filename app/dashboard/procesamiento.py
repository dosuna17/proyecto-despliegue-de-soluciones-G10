"""Lógica del análisis, independiente de Streamlit (y por eso probada en tests/).

Contiene:
- la regla de combinaciones válidas de canales (al menos un EEG),
- la preparación de los bloques que se envían a la API,
- el recorte al periodo principal de sueño usando la propia predicción,
- los indicadores del estudio y la comparación con el hipnograma experto.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

CANALES_EEG = ("EEG Fpz-Cz", "EEG Pz-Oz")
CANALES_OPCIONALES = ("EOG horizontal", "EMG submental")
ESTADIOS_SUENO = ("N1", "N2", "N3", "REM")
# Orden clínico del hipnograma, de arriba hacia abajo.
ORDEN_HIPNOGRAMA = ("W", "REM", "N1", "N2", "N3")


# --------------------------------------------------------------------------
# Combinaciones de canales
# --------------------------------------------------------------------------

def todas_las_combinaciones() -> List[Tuple[str, ...]]:
    """Las 12 combinaciones de la propuesta: 3 opciones de EEG x (EOG, EMG) opcionales."""

    opciones_eeg = [("EEG Fpz-Cz",), ("EEG Pz-Oz",), CANALES_EEG]
    opciones_extra = [()] + [c for n in (1, 2) for c in combinations(CANALES_OPCIONALES, n)]
    return [eeg + extra for eeg in opciones_eeg for extra in opciones_extra]


def numero_combinacion(canales: Sequence[str]) -> Optional[int]:
    """Posición (1 a 12) de una combinación, o None si no es válida."""

    objetivo = set(canales)
    for indice, combinacion in enumerate(todas_las_combinaciones(), start=1):
        if set(combinacion) == objetivo:
            return indice
    return None


def validar_combinacion(
    seleccion: Sequence[str],
    disponibles_en_registro: Sequence[str],
    combinaciones_modelo: Sequence[Sequence[str]],
) -> Optional[str]:
    """Devuelve un mensaje para el usuario si la selección no se puede analizar."""

    if not any(c in CANALES_EEG for c in seleccion):
        return "Seleccione al menos una señal EEG (Fpz-Cz o Pz-Oz): el modelo la necesita para clasificar."
    faltantes = [c for c in seleccion if c not in disponibles_en_registro]
    if faltantes:
        return f"El registro no contiene: {', '.join(faltantes)}."
    if not any(set(seleccion) == set(c) for c in combinaciones_modelo):
        return "El modelo todavía no admite esta combinación de señales."
    return None


# --------------------------------------------------------------------------
# Preparación de la entrada del modelo
# --------------------------------------------------------------------------

def numero_epocas(longitud_muestras: int, muestras_por_epoca: int) -> int:
    return int(longitud_muestras // muestras_por_epoca)


def bloques_secuencia(n_epocas: int, largo: int) -> List[Tuple[int, int, int]]:
    """Divide las épocas en bloques de ``largo`` para el modelo secuencial.

    Devuelve (inicio, fin, primera_nueva): el bloque cubre [inicio, fin) y solo
    se usan sus predicciones desde ``primera_nueva``. El último bloque se
    superpone con el anterior en lugar de descartar las épocas sobrantes, así
    se clasifican todas las épocas del registro.
    """

    if n_epocas < largo:
        return []
    bloques = []
    inicio = 0
    while inicio + largo <= n_epocas:
        bloques.append((inicio, inicio + largo, inicio))
        inicio += largo
    if inicio < n_epocas:
        bloques.append((n_epocas - largo, n_epocas, inicio))
    return bloques


def construir_bloque(
    senales: Dict[str, np.ndarray],
    canales_usados: Sequence[str],
    canales_modelo: Sequence[str],
    inicio: int,
    fin: int,
    muestras_por_epoca: int,
    con_presencia: bool = True,
) -> np.ndarray:
    """Arma un bloque (épocas, canales, muestras) en el orden fijo del modelo.

    Las señales no seleccionadas van en ceros y, si ``con_presencia``, se
    añade una marca 0/1 por señal (contrato de /predict-tinysleepnet).
    """

    n = fin - inicio
    n_canales = len(canales_modelo) * (2 if con_presencia else 1)
    bloque = np.zeros((n, n_canales, muestras_por_epoca), dtype=np.float32)
    a, b = inicio * muestras_por_epoca, fin * muestras_por_epoca
    for posicion, canal in enumerate(canales_modelo):
        if canal in canales_usados:
            bloque[:, posicion, :] = senales[canal][a:b].reshape(n, muestras_por_epoca)
            if con_presencia:
                bloque[:, len(canales_modelo) + posicion, :] = 1.0
    return bloque


# --------------------------------------------------------------------------
# Recorte al periodo principal de sueño
# --------------------------------------------------------------------------

@dataclass
class Ventana:
    inicio: int  # primera época analizada (incluida)
    fin: int  # última época analizada (excluida)
    inicio_sueno: Optional[int]
    fin_sueno: Optional[int]
    metodo: str


def bloques_sueno_sostenido(estadios: Sequence[str], min_bloque_epocas: int) -> List[Tuple[int, int]]:
    """Tramos de sueño continuo de al menos ``min_bloque_epocas`` épocas.

    Una época de sueño aislada en medio de horas de vigilia suele ser un
    error del modelo o un microsueño; no debe definir el inicio del sueño.
    Devuelve (inicio, fin) con ``fin`` incluido.
    """

    es_sueno = np.isin(np.asarray(estadios, dtype=object), ESTADIOS_SUENO)
    bloques: List[Tuple[int, int]] = []
    i, n = 0, len(es_sueno)
    while i < n:
        if not es_sueno[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and es_sueno[j + 1]:
            j += 1
        if j - i + 1 >= min_bloque_epocas:
            bloques.append((i, j))
        i = j + 1
    return bloques


def periodo_principal_sueno(
    estadios: Sequence[str],
    margen_epocas: int,
    max_interrupcion_epocas: int,
    min_sueno_epocas: int,
    min_bloque_epocas: int = 1,
    metodo: str = "prediccion",
) -> Ventana:
    """Ubica el periodo principal de sueño y le añade un margen de vigilia.

    1. Se toman solo los tramos de sueño sostenido (``min_bloque_epocas``
       seguidas), para que épocas sueltas no estiren el periodo.
    2. Se unen los tramos separados por menos de ``max_interrupcion_epocas``
       de vigilia (despertares nocturnos).
    3. Se elige el grupo con más sueño: así una siesta durante el día no
       se confunde con la noche.
    4. Se añaden ``margen_epocas`` antes y después, como en el entrenamiento.
    """

    estadios = np.asarray(estadios, dtype=object)
    n = len(estadios)
    if not np.isin(estadios, ESTADIOS_SUENO).any():
        return Ventana(0, n, None, None, "sin_sueno")
    bloques = bloques_sueno_sostenido(estadios, min_bloque_epocas)
    if not bloques:
        return Ventana(0, n, None, None, "sin_periodo_claro")

    grupos: List[List[int]] = []  # [inicio, fin, épocas de sueño]
    for inicio, fin in bloques:
        if grupos and inicio - grupos[-1][1] - 1 <= max_interrupcion_epocas:
            grupos[-1][1] = fin
            grupos[-1][2] += fin - inicio + 1
        else:
            grupos.append([inicio, fin, fin - inicio + 1])

    inicio_sueno, fin_sueno, sueno = max(grupos, key=lambda g: g[2])
    if sueno < min_sueno_epocas:
        return Ventana(0, n, int(inicio_sueno), int(fin_sueno), "sin_periodo_claro")
    return Ventana(
        inicio=max(0, int(inicio_sueno) - margen_epocas),
        fin=min(n, int(fin_sueno) + margen_epocas + 1),
        inicio_sueno=int(inicio_sueno),
        fin_sueno=int(fin_sueno),
        metodo=metodo,
    )


# --------------------------------------------------------------------------
# Indicadores del estudio
# --------------------------------------------------------------------------

@dataclass
class Indicadores:
    epocas_analizadas: int
    minutos_analizados: float
    tiempo_total_sueno_min: float
    vigilia_tras_inicio_min: Optional[float]
    latencia_rem_min: Optional[float]
    inicio_sueno_epoca: Optional[int]
    fin_sueno_epoca: Optional[int]
    minutos_por_estadio: Dict[str, float] = field(default_factory=dict)
    porcentaje_sueno: Dict[str, float] = field(default_factory=dict)
    epocas_baja_confianza: int = 0


def calcular_indicadores(
    estadios: Sequence[str],
    confianzas: Sequence[float],
    ventana: Ventana,
    umbral_confianza: float,
    duracion_epoca_s: float,
    clases: Sequence[str],
) -> Indicadores:
    """Indicadores estándar del hipnograma, calculados dentro de la ventana."""

    minutos = duracion_epoca_s / 60.0
    tramo = np.asarray(estadios, dtype=object)[ventana.inicio:ventana.fin]
    confianza = np.asarray(confianzas, dtype=float)[ventana.inicio:ventana.fin]
    es_sueno = np.isin(tramo, ESTADIOS_SUENO)

    minutos_estadio = {c: float(np.sum(tramo == c) * minutos) for c in clases}
    total_sueno = float(es_sueno.sum() * minutos)
    porcentaje = {
        c: (minutos_estadio[c] / total_sueno * 100.0 if total_sueno else 0.0)
        for c in clases
        if c in ESTADIOS_SUENO
    }

    inicio_rel = fin_rel = None
    vigilia = latencia_rem = None
    if ventana.inicio_sueno is not None and ventana.metodo == "prediccion":
        # Inicio y fin del sueño sostenido (ver periodo_principal_sueno).
        inicio_rel = ventana.inicio_sueno - ventana.inicio
        fin_rel = ventana.fin_sueno - ventana.inicio
    elif es_sueno.any():
        inicio_rel = int(np.flatnonzero(es_sueno)[0])
        fin_rel = int(np.flatnonzero(es_sueno)[-1])
    if inicio_rel is not None:
        periodo = tramo[inicio_rel:fin_rel + 1]
        vigilia = float(np.sum(periodo == "W") * minutos)
        rem = np.flatnonzero(periodo == "REM")
        latencia_rem = float(rem[0] * minutos) if len(rem) else None

    return Indicadores(
        epocas_analizadas=len(tramo),
        minutos_analizados=len(tramo) * minutos,
        tiempo_total_sueno_min=total_sueno,
        vigilia_tras_inicio_min=vigilia,
        latencia_rem_min=latencia_rem,
        inicio_sueno_epoca=None if inicio_rel is None else ventana.inicio + inicio_rel,
        fin_sueno_epoca=None if fin_rel is None else ventana.inicio + fin_rel,
        minutos_por_estadio=minutos_estadio,
        porcentaje_sueno=porcentaje,
        epocas_baja_confianza=int(np.sum(confianza < umbral_confianza)),
    )


# --------------------------------------------------------------------------
# Comparación con el hipnograma experto
# --------------------------------------------------------------------------

@dataclass
class Concordancia:
    epocas_comparadas: int
    epocas_excluidas: int
    exactitud: float
    kappa: float
    clases: List[str]
    matriz: np.ndarray  # filas: experto, columnas: modelo (conteos)


def kappa_cohen(matriz: np.ndarray) -> float:
    total = matriz.sum()
    if total == 0:
        return float("nan")
    observado = np.trace(matriz) / total
    esperado = float((matriz.sum(axis=0) * matriz.sum(axis=1)).sum()) / total**2
    if esperado == 1.0:
        return 1.0
    return float((observado - esperado) / (1.0 - esperado))


def comparar_con_experto(
    prediccion: Sequence[str],
    experto: Sequence[str],
    ventana: Ventana,
    clases: Sequence[str],
) -> Optional[Concordancia]:
    """Concordancia modelo-experto en la ventana analizada.

    Se excluyen las épocas que el experto no puntuó ("?") o puntuó con un
    estadio que el modelo no predice (p. ej. "M" si el modelo tiene 5 clases).
    """

    pred = np.asarray(prediccion, dtype=object)[ventana.inicio:ventana.fin]
    exp = np.asarray(experto, dtype=object)[ventana.inicio:ventana.fin]
    validas = np.isin(exp, list(clases))
    if not validas.any():
        return None
    pred, exp_validas = pred[validas], exp[validas]
    indice = {c: i for i, c in enumerate(clases)}
    matriz = np.zeros((len(clases), len(clases)), dtype=int)
    for e, p in zip(exp_validas, pred):
        if p in indice:
            matriz[indice[e], indice[p]] += 1
    return Concordancia(
        epocas_comparadas=int(validas.sum()),
        epocas_excluidas=int((~validas).sum()),
        exactitud=float(np.trace(matriz) / matriz.sum()) if matriz.sum() else float("nan"),
        kappa=kappa_cohen(matriz),
        clases=list(clases),
        matriz=matriz,
    )


def iterar_bloques(
    senales: Dict[str, np.ndarray],
    canales_usados: Sequence[str],
    canales_modelo: Sequence[str],
    bloques: Sequence[Tuple[int, int, int]],
    muestras_por_epoca: int,
    con_presencia: bool,
) -> Iterator[Tuple[Tuple[int, int, int], np.ndarray]]:
    for bloque in bloques:
        inicio, fin, _ = bloque
        yield bloque, construir_bloque(
            senales, canales_usados, canales_modelo, inicio, fin, muestras_por_epoca, con_presencia
        )
