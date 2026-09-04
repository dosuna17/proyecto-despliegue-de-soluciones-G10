"""
Ventaneo (30s) y etiquetado a partir de la señal cruda + anotaciones
crudas cargadas por loader.py.

Esquema de etiquetas (según especificación del proyecto):
    El hypnograma original usa el manual antiguo Rechtschaffen & Kales,
    con 6 clases principales {W, 1, 2, 3, 4, R}, más dos anotaciones
    adicionales que NO son estadíos de sueño: 'M' (movimiento/ruido,
    época no clasificable) y '?' (época no anotada).

    Este módulo:
      1. Descarta cualquier anotación que no sea una de las 6 clases
         R&K permitidas -> se eliminan 'M' y '?' (y cualquier otra
         etiqueta no reconocida).
      2. Fusiona los estadíos '3' y '4' en una sola clase "N3".
      3. Mapea las 5 clases resultantes a enteros:
             W   -> 0
             N1  -> 1   (antes '1')
             N2  -> 2   (antes '2')
             N3  -> 3   (antes '3' y '4' fusionados)
             REM -> 4   (antes 'R')

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import mne

EPOCH_LEN_SEC = 30.0

# MNE lee las señales EDF en voltios; el estándar de trabajo en EEG/PSG
# es microvoltios (µV), así que se reescala por defecto al extraer épocas.
VOLTS_TO_MICROVOLTS = 1e6

# Descripción cruda del hypnograma -> etiqueta canónica R&K
_RAW_TO_STAGE = {
    "Sleep stage W": "W",
    "Sleep stage 1": "1",
    "Sleep stage 2": "2",
    "Sleep stage 3": "3",
    "Sleep stage 4": "4",
    "Sleep stage R": "R",
}
# Todo lo demás ('Sleep stage ?', 'Movement time', 'M', etc.) se descarta
# automáticamente por no estar en el diccionario anterior.

# Etiqueta canónica -> clase final (3 y 4 se fusionan en N3)
_STAGE_TO_CLASS = {
    "W": "W",
    "1": "N1",
    "2": "N2",
    "3": "N3",
    "4": "N3",
    "R": "REM",
}

# Clase final -> entero usado por el modelo
CLASS_TO_INT: Dict[str, int] = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}
INT_TO_CLASS: Dict[int, str] = {v: k for k, v in CLASS_TO_INT.items()}


def parse_recording_id(recording_id: str) -> Tuple[str, str]:
    """
    Extrae (subject_code, night) a partir del ID de grabación de Sleep-EDF
    Sleep Cassette, con convención de nombre 'SC4ssNEc':
        ss = número de sujeto (persiste entre noches, p. ej. '00')
        N  = número de noche  ('1' o '2')

    Ej.: 'SC4001' -> subject_code='00', night='1'

    Esto es lo que se necesita para un split correcto por sujeto: dos
    noches del mismo sujeto DEBEN quedar del mismo lado del split
    (train/val/test) para evitar data leakage.

    Si el ID no sigue esta convención (p. ej. viene de otro dataset),
    se devuelve el ID completo como subject_code y night='0' en vez de
    fallar.
    """
    if len(recording_id) >= 6 and recording_id.startswith("SC4"):
        return recording_id[3:5], recording_id[5]
    return recording_id, "0"


@dataclass
class EpochDataset:
    """
    Épocas de longitud fija ya extraídas y etiquetadas para una grabación
    (un sujeto/noche). Cada época queda representada de forma consistente
    y autocontenida: señal + etiqueta de 5 clases + identificador de
    sujeto/noche (uno por época, no solo a nivel de objeto), de modo que
    al concatenar varias grabaciones (ver `concatenate_datasets`) no se
    pierda la trazabilidad de origen de cada época.
    """
    recording_id: str      # ID completo de la grabación, p. ej. 'SC4001'
    subject_code: str      # sujeto persistente entre noches, p. ej. '00'
    night: str             # número de noche, p. ej. '1'
    X: np.ndarray          # (n_epochs, n_channels, n_samples_por_epoca)
    y: np.ndarray          # (n_epochs,) enteros según CLASS_TO_INT
    recording_ids: np.ndarray   # (n_epochs,) str — recording_id repetido, uno por época
    subject_codes: np.ndarray   # (n_epochs,) str — subject_code repetido, uno por época
    channel_names: List[str]
    sfreq: float
    signal_unit: str        # "uV" o "V" — unidad en la que está escalado X
    dropped_epochs: int    # épocas descartadas (etiqueta inválida o cola incompleta)

    @property
    def subject_id(self) -> str:
        """Alias retrocompatible: antes el único identificador era el ID de grabación."""
        return self.recording_id


def _annotations_to_valid_intervals(
    annotations: mne.Annotations,
) -> List[Tuple[float, float, str]]:
    """
    Convierte las anotaciones crudas en intervalos (start, end, clase_final),
    quedándose solo con las 6 etiquetas R&K permitidas y aplicando la
    fusión 3+4 -> N3. Descarta '?' y 'M' (u otra etiqueta no reconocida).
    """
    intervals = []
    for onset, duration, desc in zip(
        annotations.onset, annotations.duration, annotations.description
    ):
        stage = _RAW_TO_STAGE.get(desc)
        if stage is None:
            continue  # descarta 'Sleep stage ?', 'Movement time', etc.
        intervals.append((onset, onset + duration, _STAGE_TO_CLASS[stage]))
    intervals.sort(key=lambda t: t[0])
    return intervals


def _label_for_epoch(
    epoch_start: float,
    epoch_end: float,
    intervals: Sequence[Tuple[float, float, str]],
) -> Optional[str]:
    """
    Devuelve la clase final cuyo intervalo contiene el punto medio de la
    época, o None si el punto medio cae en una región descartada/sin anotar.
    """
    midpoint = (epoch_start + epoch_end) / 2.0
    for start, end, class_name in intervals:
        if start <= midpoint < end:
            return class_name
    return None


def build_epoch_grid(
    total_duration_sec: float, epoch_len: float = EPOCH_LEN_SEC
) -> List[Tuple[float, float]]:
    """
    Construye una grilla fija de ventanas [start, end) sin solapamiento
    sobre toda la grabación, empezando en t=0. Cualquier cola final
    menor a epoch_len se descarta (nunca se rellena ni se fusiona con
    la ventana anterior).
    """
    n_full_epochs = int(total_duration_sec // epoch_len)
    return [(i * epoch_len, (i + 1) * epoch_len) for i in range(n_full_epochs)]


def extract_epochs(
    raw: mne.io.BaseRaw,
    annotations: mne.Annotations,
    epoch_len: float = EPOCH_LEN_SEC,
    recording_id: str = "unknown",
    channel_picks: Optional[Sequence[str]] = None,
    scale_to_uv: bool = True,
    dtype: np.dtype = np.float32,
) -> EpochDataset:
    """
    Segmenta la señal continua de una grabación (sujeto/noche) en épocas
    fijas de 30s y les asigna una etiqueta entera, descartando toda época
    cuya etiqueta no sea una de las 5 clases permitidas (W, N1, N2, N3, REM).

    channel_picks: lista opcional de nombres de canal a conservar
    (p. ej. ['EEG Fpz-Cz'] o ['EEG Fpz-Cz', 'EEG Pz-Oz']). Si es None,
    se conservan todos los canales del PSG. Totalmente parametrizable:
    cualquier subconjunto de `raw.ch_names` es válido.

    scale_to_uv: si True (default), la señal se reescala de voltios
    (unidad nativa del EDF vía MNE) a microvoltios. Si False, se deja
    en voltios tal cual la entrega MNE.

    dtype: tipo de dato de X (default float32). MNE entrega float64 por
    defecto, que duplica innecesariamente el uso de RAM para EEG/EOG/EMG
    -- float32 tiene precisión de sobra para estas señales y es clave
    para procesar el dataset completo (153 sujetos) sin quedarse sin
    memoria.
    """
    if channel_picks is not None:
        raw = raw.copy().pick(channel_picks)

    sfreq = raw.info["sfreq"]
    total_duration = raw.n_times / sfreq
    samples_per_epoch = int(round(epoch_len * sfreq))

    intervals = _annotations_to_valid_intervals(annotations)
    grid = build_epoch_grid(total_duration, epoch_len)

    raw.load_data(verbose=False)
    data = raw.get_data().astype(dtype)  # (n_channels, n_samples), en voltios

    if scale_to_uv:
        data = data * np.array(VOLTS_TO_MICROVOLTS, dtype=dtype)
    signal_unit = "uV" if scale_to_uv else "V"

    X_list, y_list = [], []
    dropped = 0

    for start, end in grid:
        class_name = _label_for_epoch(start, end, intervals)
        if class_name is None:
            dropped += 1
            continue

        start_sample = int(round(start * sfreq))
        end_sample = start_sample + samples_per_epoch
        if end_sample > data.shape[1]:
            dropped += 1
            continue

        X_list.append(data[:, start_sample:end_sample])
        y_list.append(CLASS_TO_INT[class_name])

    if not X_list:
        raise ValueError(f"No se extrajo ninguna época válida para la grabación {recording_id}")

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)

    subject_code, night = parse_recording_id(recording_id)
    n_epochs = len(y)

    return EpochDataset(
        recording_id=recording_id,
        subject_code=subject_code,
        night=night,
        X=X,
        y=y,
        recording_ids=np.full(n_epochs, recording_id, dtype=object),
        subject_codes=np.full(n_epochs, subject_code, dtype=object),
        channel_names=raw.ch_names,
        sfreq=sfreq,
        signal_unit=signal_unit,
        dropped_epochs=dropped,
    )


def concatenate_datasets(datasets: Sequence[EpochDataset]) -> EpochDataset:
    """
    Concatena varias EpochDataset (una por sujeto/noche) en una sola,
    preservando por época el identificador de sujeto/noche de origen.
    Útil para armar el dataset completo de entrenamiento y luego hacer
    un split GroupKFold/GroupShuffleSplit usando `subject_codes` como
    grupo (nunca `recording_ids`, para no separar las 2 noches de un
    mismo sujeto entre train y test).

    Requiere que todas las grabaciones compartan el mismo conjunto de
    canales y la misma frecuencia de muestreo.
    """
    if not datasets:
        raise ValueError("La lista de datasets está vacía")

    ref_channels = datasets[0].channel_names
    ref_sfreq = datasets[0].sfreq
    ref_unit = datasets[0].signal_unit
    for ds in datasets[1:]:
        if ds.channel_names != ref_channels:
            raise ValueError(
                f"Canales inconsistentes entre grabaciones: {ds.recording_id} "
                f"tiene {ds.channel_names}, se esperaba {ref_channels}"
            )
        if ds.sfreq != ref_sfreq:
            raise ValueError(
                f"sfreq inconsistente: {ds.recording_id} tiene {ds.sfreq}, "
                f"se esperaba {ref_sfreq}"
            )
        if ds.signal_unit != ref_unit:
            raise ValueError(
                f"Unidad de señal inconsistente: {ds.recording_id} tiene "
                f"'{ds.signal_unit}', se esperaba '{ref_unit}'"
            )

    return EpochDataset(
        recording_id="+".join(ds.recording_id for ds in datasets),
        subject_code="multi",
        night="multi",
        X=np.concatenate([ds.X for ds in datasets], axis=0),
        y=np.concatenate([ds.y for ds in datasets], axis=0),
        recording_ids=np.concatenate([ds.recording_ids for ds in datasets], axis=0),
        subject_codes=np.concatenate([ds.subject_codes for ds in datasets], axis=0),
        channel_names=ref_channels,
        sfreq=ref_sfreq,
        signal_unit=ref_unit,
        dropped_epochs=sum(ds.dropped_epochs for ds in datasets),
    )


def concatenate_datasets_by_split(
    datasets: Sequence[EpochDataset],
    subject_to_split: Dict[str, str],
    target_split: str,
) -> EpochDataset:
    """
    Filtra una lista de EpochDataset (uno por sujeto/noche) quedándose solo
    con los que pertenecen a `target_split` ("train" o "test"), y los
    concatena con concatenate_datasets.

    IMPORTANTE: el nombre de archivo / recording_id NO contiene esta
    información -- el split se decide aparte (ver split_subjects.ipynb,
    que genera SC-subjects-split.xlsx) y debe cruzarse explícitamente por
    subject_code. `subject_to_split` es ese mapeo, típicamente construido así:

        import pandas as pd
        split_df = pd.read_excel("SC-subjects-split.xlsx")
        split_df["subject_code"] = split_df["subject"].astype(str).str.zfill(2)
        subject_to_split = dict(zip(split_df["subject_code"], split_df["split"]))
    """
    selected = [ds for ds in datasets if subject_to_split.get(ds.subject_code) == target_split]
    if not selected:
        raise ValueError(
            f"Ningún dataset pertenece al split '{target_split}' según subject_to_split"
        )
    return concatenate_datasets(selected)


def trim_wake_edges(dataset: EpochDataset, buffer_epochs: int = 60) -> EpochDataset:
    """
    Recorta el exceso de vigilia (W) al inicio y al final de UNA grabación,
    dejando un buffer alrededor de la primera y última época de sueño real.

    Esto reduce el desbalance hacia W sin descartar información real: no
    toca los despertares (arousals) que ocurren DENTRO de la noche, solo el
    tramo de vigilia prolongada antes de dormirse y después de despertar
    (el sujeto acostado con el equipo puesto, sin actividad de sueño).

    buffer_epochs: cuántas épocas de W se conservan antes de la primera y
    después de la última época de sueño real (default 60 = 30 minutos,
    valor típico en literatura de sleep staging, p. ej. DeepSleepNet).

    Debe aplicarse por grabación individual, ANTES de concatenar varios
    sujetos -- los límites de recorte son específicos de cada noche.
    """
    wake_int = CLASS_TO_INT["W"]
    non_wake_idx = np.where(dataset.y != wake_int)[0]

    if len(non_wake_idx) == 0:
        raise ValueError(
            f"La grabación {dataset.recording_id} no tiene ninguna época de sueño real "
            f"(todo es W) -- no se puede recortar."
        )

    first_sleep = non_wake_idx.min()
    last_sleep = non_wake_idx.max()
    start = max(0, first_sleep - buffer_epochs)
    end = min(len(dataset.y), last_sleep + buffer_epochs + 1)

    n_removed = len(dataset.y) - (end - start)

    return EpochDataset(
        recording_id=dataset.recording_id,
        subject_code=dataset.subject_code,
        night=dataset.night,
        X=dataset.X[start:end],
        y=dataset.y[start:end],
        recording_ids=dataset.recording_ids[start:end],
        subject_codes=dataset.subject_codes[start:end],
        channel_names=dataset.channel_names,
        sfreq=dataset.sfreq,
        signal_unit=dataset.signal_unit,
        dropped_epochs=dataset.dropped_epochs + n_removed,
    )


def class_distribution(dataset: EpochDataset) -> Dict[str, int]:
    """Conteo de épocas por clase final, en formato legible."""
    counts = {name: 0 for name in CLASS_TO_INT}
    for label in dataset.y:
        counts[INT_TO_CLASS[int(label)]] += 1
    return counts


if __name__ == "__main__":
    # Prueba rápida manual encadenando con loader.py
    import sys
    from loader import load_subject

    psg_path = sys.argv[1] if len(sys.argv) > 1 else "SC4001E0-PSG.edf"
    hypno_path = sys.argv[2] if len(sys.argv) > 2 else "SC4001EC-Hypnogram.edf"

    subject = load_subject(psg_path, hypno_path)
    dataset = extract_epochs(subject.raw, subject.annotations, recording_id=subject.subject_id)

    print(f"Grabación: {dataset.recording_id} (sujeto={dataset.subject_code}, noche={dataset.night})")
    print(f"Canales: {dataset.channel_names}  | unidad: {dataset.signal_unit}")
    print(f"X shape: {dataset.X.shape}  (n_epochs, n_channels, n_samples)")
    print(f"y shape: {dataset.y.shape}")
    print(f"recording_ids[:3]: {dataset.recording_ids[:3]}")
    print(f"Épocas descartadas: {dataset.dropped_epochs}")
    print(f"Distribución de clases: {class_distribution(dataset)}")
