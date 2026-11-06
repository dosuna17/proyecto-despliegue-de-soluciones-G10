"""Lectura de registros de sueño: archivos EDF subidos y estudios de demostración.

Ambos caminos producen el mismo objeto ``Registro``, para que el resto del
tablero no dependa de dónde vino la señal.
"""

from __future__ import annotations

import json
import tempfile
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Equivalencias de las anotaciones de Sleep-EDF (R&K) a la notación AASM,
# las mismas que usa src/data/segmentation.py para entrenar.
ANOTACIONES_A_ESTADIO = {
    "Sleep stage W": "W",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",
    "Sleep stage R": "REM",
    "Movement time": "M",
    "Sleep stage ?": "?",
}
VOLTIOS_A_MICROVOLTIOS = 1e6


class ErrorRegistro(ValueError):
    """Problema con el archivo que se puede explicar al usuario."""


@dataclass
class Registro:
    """Una noche de polisomnografía lista para clasificar.

    senales: nombre de canal (nombre del modelo) -> señal continua en µV a
        ``frecuencia_hz``.
    etiquetas_experto: estadio por época de 30 s según el hipnograma, o None
        si el usuario no lo subió.
    """

    nombre: str
    origen: str  # "demo" o "carga"
    frecuencia_hz: int
    senales: Dict[str, np.ndarray]
    inicio: Optional[datetime] = None
    etiquetas_experto: Optional[np.ndarray] = None
    canales_en_archivo: List[str] = field(default_factory=list)
    avisos: List[str] = field(default_factory=list)
    metadatos: Dict[str, str] = field(default_factory=dict)

    @property
    def duracion_s(self) -> float:
        if not self.senales:
            return 0.0
        return min(len(s) for s in self.senales.values()) / self.frecuencia_hz

    @property
    def canales_disponibles(self) -> List[str]:
        return list(self.senales)

    @property
    def tiene_hipnograma(self) -> bool:
        return self.etiquetas_experto is not None


def _normalizar(nombre: str) -> str:
    return "".join(nombre.lower().split())


def mapear_canales(nombres_archivo: List[str], alias: Dict[str, List[str]]) -> Dict[str, str]:
    """Relaciona cada canal conocido con su nombre en el archivo.

    Devuelve {nombre_modelo: nombre_en_archivo} solo para los encontrados.
    """

    por_normalizado = {_normalizar(n): n for n in nombres_archivo}
    encontrados: Dict[str, str] = {}
    for canonico, opciones in alias.items():
        for opcion in [canonico, *opciones]:
            nombre = por_normalizado.get(_normalizar(opcion))
            if nombre is not None:
                encontrados[canonico] = nombre
                break
    return encontrados


def _a_datetime(valor) -> Optional[datetime]:
    if valor is None:
        return None
    if isinstance(valor, tuple):
        valor = valor[0]
    if isinstance(valor, datetime):
        return valor if valor.tzinfo else valor.replace(tzinfo=timezone.utc)
    return None


def etiquetas_por_epoca(
    onsets: np.ndarray,
    duraciones: np.ndarray,
    descripciones: List[str],
    n_epocas: int,
    duracion_epoca_s: float,
) -> np.ndarray:
    """Convierte las anotaciones del hipnograma en un estadio por época.

    Las épocas sin anotación quedan como "?" (no puntuadas) y no se usan
    para comparar con el modelo.
    """

    etiquetas = np.full(n_epocas, "?", dtype=object)
    for inicio, duracion, descripcion in zip(onsets, duraciones, descripciones):
        estadio = ANOTACIONES_A_ESTADIO.get(str(descripcion).strip())
        if estadio is None:
            continue
        primera = int(round(inicio / duracion_epoca_s))
        cantidad = int(round(duracion / duracion_epoca_s))
        primera = max(primera, 0)
        ultima = min(primera + cantidad, n_epocas)
        if ultima > primera:
            etiquetas[primera:ultima] = estadio
    return etiquetas


def leer_edf(
    ruta_psg: str | Path,
    alias: Dict[str, List[str]],
    frecuencia_objetivo: int,
    duracion_epoca_s: float,
    ruta_hipnograma: str | Path | None = None,
    nombre: Optional[str] = None,
) -> Registro:
    """Lee un PSG en EDF (y opcionalmente su hipnograma) en un ``Registro``.

    - Solo carga las señales que el tablero conoce (``alias``), para ahorrar memoria.
    - Convierte voltios a microvoltios, igual que en el entrenamiento.
    - Si la frecuencia no es ``frecuencia_objetivo``, remuestrea y lo avisa.
    """

    import mne  # importación diferida: solo se necesita al subir archivos

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = mne.io.read_raw_edf(str(ruta_psg), preload=False, verbose="ERROR")
    except Exception as error:  # noqa: BLE001 - el formato puede fallar de muchas formas
        raise ErrorRegistro(
            "No fue posible leer el archivo de señales. Verifique que sea un archivo EDF válido."
        ) from error

    canales_archivo = list(raw.ch_names)
    encontrados = mapear_canales(canales_archivo, alias)
    if not encontrados:
        raise ErrorRegistro(
            "El archivo no contiene ninguna de las señales que el modelo reconoce "
            f"({', '.join(alias)}). Canales encontrados: {', '.join(canales_archivo)}."
        )

    avisos: List[str] = []
    senales: Dict[str, np.ndarray] = {}
    for canonico, nombre_archivo in encontrados.items():
        canal = raw.copy().pick([nombre_archivo]).load_data(verbose="ERROR")
        frecuencia = float(canal.info["sfreq"])
        if abs(frecuencia - frecuencia_objetivo) > 1e-6:
            canal.resample(frecuencia_objetivo, verbose="ERROR")
            avisos.append(
                f"{canonico} venía a {frecuencia:g} Hz y se ajustó a {frecuencia_objetivo} Hz, "
                "la frecuencia con la que se entrenó el modelo."
            )
        datos = canal.get_data()[0] * VOLTIOS_A_MICROVOLTIOS
        if not np.isfinite(datos).all():
            datos = np.nan_to_num(datos, nan=0.0, posinf=0.0, neginf=0.0)
            avisos.append(f"{canonico} tenía valores inválidos; se reemplazaron por cero.")
        if np.allclose(datos, 0.0):
            avisos.append(f"{canonico} está vacía (solo ceros); no se podrá usar.")
            continue
        senales[canonico] = datos.astype(np.float32)

    if not senales:
        raise ErrorRegistro("Las señales reconocidas del archivo están vacías.")

    longitud = min(len(s) for s in senales.values())
    senales = {k: v[:longitud] for k, v in senales.items()}
    n_epocas = int(longitud // (frecuencia_objetivo * duracion_epoca_s))

    etiquetas = None
    if ruta_hipnograma is not None:
        try:
            anotaciones = mne.read_annotations(str(ruta_hipnograma))
        except Exception as error:  # noqa: BLE001
            raise ErrorRegistro(
                "No fue posible leer el hipnograma. Verifique que sea el archivo "
                "*-Hypnogram.edf correspondiente a este registro."
            ) from error
        desplazamiento = 0.0
        inicio_psg = _a_datetime(raw.info.get("meas_date"))
        inicio_hipno = _a_datetime(anotaciones.orig_time)
        if inicio_psg and inicio_hipno:
            desplazamiento = (inicio_hipno - inicio_psg).total_seconds()
        etiquetas = etiquetas_por_epoca(
            anotaciones.onset + desplazamiento,
            anotaciones.duration,
            list(anotaciones.description),
            n_epocas,
            duracion_epoca_s,
        )
        if not np.isin(etiquetas, ["W", "N1", "N2", "N3", "REM", "M"]).any():
            raise ErrorRegistro(
                "El hipnograma no tiene estadios reconocibles (Sleep stage W/1/2/3/4/R). "
                "Puede continuar sin él."
            )

    return Registro(
        nombre=nombre or Path(ruta_psg).stem,
        origen="carga",
        frecuencia_hz=frecuencia_objetivo,
        senales=senales,
        inicio=_a_datetime(raw.info.get("meas_date")),
        etiquetas_experto=etiquetas,
        canales_en_archivo=canales_archivo,
        avisos=avisos,
    )


def leer_edf_subido(
    contenido_psg: bytes,
    contenido_hipnograma: Optional[bytes],
    nombre: str,
    alias: Dict[str, List[str]],
    frecuencia_objetivo: int,
    duracion_epoca_s: float,
) -> Registro:
    """Guarda los bytes subidos en archivos temporales y los lee con ``leer_edf``."""

    with tempfile.TemporaryDirectory(prefix="tablero_") as carpeta:
        ruta_psg = Path(carpeta) / "registro-PSG.edf"
        ruta_psg.write_bytes(contenido_psg)
        ruta_hipno = None
        if contenido_hipnograma:
            ruta_hipno = Path(carpeta) / "registro-Hypnogram.edf"
            ruta_hipno.write_bytes(contenido_hipnograma)
        registro = leer_edf(ruta_psg, alias, frecuencia_objetivo, duracion_epoca_s, ruta_hipno, nombre)
    return registro


# --------------------------------------------------------------------------
# Estudios de demostración (.npz generados por scripts/generar_estudios_demo.py)
# --------------------------------------------------------------------------

def listar_demos(carpeta: str | Path) -> List[Dict[str, str]]:
    """Lista los estudios de demostración disponibles con sus metadatos."""

    carpeta = Path(carpeta)
    if not carpeta.is_dir():
        return []
    demos = []
    for ruta in sorted(carpeta.glob("*.npz")):
        try:
            with np.load(ruta, allow_pickle=False) as datos:
                meta = json.loads(str(datos["metadatos"]))
        except Exception:  # noqa: BLE001 - un archivo dañado no debe tumbar el tablero
            continue
        meta["ruta"] = str(ruta)
        demos.append(meta)
    return demos


def leer_demo(ruta: str | Path) -> Registro:
    """Carga un estudio de demostración como ``Registro``."""

    with np.load(ruta, allow_pickle=False) as datos:
        meta = json.loads(str(datos["metadatos"]))
        senales = {
            canal: datos[f"senal_{i}"].astype(np.float32)
            for i, canal in enumerate(meta["canales"])
        }
        etiquetas = datos["etiquetas_experto"].astype(object) if "etiquetas_experto" in datos else None
    inicio = datetime.fromisoformat(meta["inicio"]) if meta.get("inicio") else None
    return Registro(
        nombre=meta["nombre"],
        origen="demo",
        frecuencia_hz=int(meta["frecuencia_hz"]),
        senales=senales,
        inicio=inicio,
        etiquetas_experto=etiquetas,
        canales_en_archivo=list(meta["canales"]),
        metadatos={k: str(v) for k, v in meta.items() if k in ("descripcion", "edad", "sexo", "noche")},
    )


def guardar_demo(registro: Registro, ruta: str | Path, descripcion: str, extra: Dict[str, str] | None = None) -> Path:
    """Guarda un ``Registro`` como estudio de demostración comprimido.

    Las señales se guardan en float16: para valores de hasta 256 µV el error
    de redondeo es menor a 0.13 µV y el archivo pesa la mitad.
    """

    ruta = Path(ruta)
    canales = registro.canales_disponibles
    meta = {
        "nombre": registro.nombre,
        "descripcion": descripcion,
        "frecuencia_hz": registro.frecuencia_hz,
        "canales": canales,
        "inicio": registro.inicio.isoformat() if registro.inicio else None,
        **(extra or {}),
    }
    arreglos = {f"senal_{i}": registro.senales[c].astype(np.float16) for i, c in enumerate(canales)}
    if registro.etiquetas_experto is not None:
        arreglos["etiquetas_experto"] = np.asarray(registro.etiquetas_experto, dtype="U3")
    np.savez_compressed(ruta, metadatos=np.array(json.dumps(meta, ensure_ascii=False)), **arreglos)
    return ruta
