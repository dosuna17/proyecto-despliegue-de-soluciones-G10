"""
Ventaneo (30s) y etiquetado a partir de la señal cruda + anotaciones
crudas cargadas por loader.py.

Segmentación (igual que en 01_EDA):
    Los intervalos del hypnograma agrupan épocas consecutivas con la misma
    etiqueta y siempre duran un múltiplo de 30 s. Cada intervalo se parte
    en duracion // 30 épocas de 30 s que conservan la etiqueta del
    intervalo, y se recorta la señal en [inicio, inicio + 30 s).

Esquema de etiquetas (según especificación del proyecto):
    El hypnograma original usa el manual antiguo Rechtschaffen & Kales,
    con 6 clases principales {W, 1, 2, 3, 4, R}, más dos anotaciones
    adicionales que NO son estadíos de sueño: 'M' (movimiento/ruido,
    época no clasificable) y '?' (época no anotada).

    Este módulo:
      1. Fusiona los estadíos '3' y '4' en una sola clase "N3".
      2. Conserva solo las clases pedidas en `classes` (ver
         extract_epochs). Por defecto son las 5 clases de sueño
         W, N1, N2, N3, REM, es decir, se eliminan 'M' y '?'; se pueden
         conservar pasándolas explícitamente. Cualquier otra etiqueta no
         reconocida se descarta siempre.

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import mne

EPOCH_LEN_SEC = 30.0

_RAW_TO_CLASS = {
    "Sleep stage W": "W",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",
    "Sleep stage R": "REM",
    "Movement time": "M",
    "Sleep stage ?": "?",
}

CLASS_NAMES = ["W", "N1", "N2", "N3", "REM"]
ALL_CLASS_NAMES = CLASS_NAMES + ["M", "?"]
SLEEP_CLASS_NAMES = ["N1", "N2", "N3", "REM"]
CLASS_TO_INT: Dict[str, int] = {name: idx for idx, name in enumerate(ALL_CLASS_NAMES)}
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
    y: np.ndarray          # (n_epochs,) nombres de clase (las pedidas en `classes`)
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


def _validate_classes(classes: Sequence[str]) -> List[str]:
    """Verifica que `classes` solo tenga clases conocidas y la devuelve como lista."""
    unknown = [c for c in classes if c not in ALL_CLASS_NAMES]
    if unknown:
        raise ValueError(f"Clases desconocidas: {unknown}. Opciones válidas: {ALL_CLASS_NAMES}")
    if not classes:
        raise ValueError("`classes` no puede estar vacío")
    return list(classes)


def annotations_to_epochs(
    annotations: mne.Annotations,
    epoch_len: float = EPOCH_LEN_SEC,
    classes: Sequence[str] = CLASS_NAMES,
) -> List[Tuple[float, Optional[str]]]:
    """
    Divide cada intervalo del hypnograma en épocas consecutivas de
    `epoch_len` segundos, igual que en 01_EDA: un intervalo de 300 s con
    'Sleep stage 2' se convierte en 10 épocas N2 que empiezan en
    onset, onset + 30, ..., onset + 270.

    classes: clases a conservar, cualquier subconjunto de ALL_CLASS_NAMES
    (default CLASS_NAMES = W, N1, N2, N3, REM). Por ejemplo, para
    conservar también las épocas de movimiento: CLASS_NAMES + ["M"].

    Los intervalos del hypnograma son múltiplos de 30 s (comprobado en
    01_EDA); si alguno de las clases conservadas no lo es se lanza un
    error en vez de truncarlo.

    Devuelve una lista de (inicio_seg, clase_final) ordenada por inicio.
    clase_final es None para las épocas de clases no pedidas en `classes`
    (o de descripciones no reconocidas), para que el llamador pueda
    contarlas como descartadas.
    """
    keep = set(_validate_classes(classes))

    epochs = []
    for onset, duration, desc in zip(
        annotations.onset, annotations.duration, annotations.description
    ):
        class_name = _RAW_TO_CLASS.get(desc)
        if class_name not in keep:
            class_name = None

        n_epochs = duration / epoch_len
        if class_name is not None and not np.isclose(n_epochs, round(n_epochs)):
            raise ValueError(
                f"Intervalo '{desc}' en onset={onset}s dura {duration}s, "
                f"que no es múltiplo de {epoch_len}s"
            )

        for k in range(int(round(n_epochs))):
            epochs.append((onset + k * epoch_len, class_name))

    epochs.sort(key=lambda t: t[0])
    return epochs


def extract_epochs(
    raw: mne.io.BaseRaw,
    annotations: mne.Annotations,
    epoch_len: float = EPOCH_LEN_SEC,
    recording_id: str = "unknown",
    channel_picks: Optional[Sequence[str]] = None,
    scale_to_uv: bool = True,
    dtype: np.dtype = np.float32,
    classes: Sequence[str] = CLASS_NAMES,
) -> EpochDataset:
    """
    Segmenta la señal continua de una grabación (sujeto/noche) en épocas
    de 30s a partir de los intervalos del hypnograma (ver
    `annotations_to_epochs`): cada intervalo se parte en épocas de 30s que
    heredan su etiqueta. Se descarta toda época cuya etiqueta no esté en
    `classes` o que se salga de la señal.

    classes: clases a conservar, cualquier subconjunto de ALL_CLASS_NAMES
    = [W, N1, N2, N3, REM, M, ?]. Por defecto CLASS_NAMES (las 5 clases
    de sueño); p. ej. classes=CLASS_NAMES + ["M"] conserva también las
    épocas de movimiento.

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

    epochs = annotations_to_epochs(annotations, epoch_len, classes)

    raw.load_data(verbose=False)
    signal_unit = "uV" if scale_to_uv else "V"
    data = raw.get_data(units=signal_unit).astype(dtype)  # (n_channels, n_samples)

    X_list, y_list = [], []
    dropped = 0

    for start, class_name in epochs:
        if start >= total_duration:
            continue
        if class_name is None:
            dropped += 1
            continue

        start_sample = int(round(start * sfreq))
        end_sample = start_sample + samples_per_epoch
        if end_sample > data.shape[1]:
            dropped += 1
            continue

        X_list.append(data[:, start_sample:end_sample])
        y_list.append(class_name)

    if not X_list:
        raise ValueError(f"No se extrajo ninguna época válida para la grabación {recording_id}")

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=str)

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
    labels = np.array([label_to_class_name(label) for label in dataset.y])
    sleep_idx = np.where(np.isin(labels, SLEEP_CLASS_NAMES))[0]

    if len(sleep_idx) == 0:
        raise ValueError(
            f"La grabación {dataset.recording_id} no tiene ninguna época de sueño real "
            f"(N1/N2/N3/REM) -- no se puede recortar."
        )

    first_sleep = sleep_idx.min()
    last_sleep = sleep_idx.max()
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


def label_to_class_name(label: Union[str, int, np.integer]) -> str:
    """Convierte una etiqueta nueva o antigua al nombre canonico de clase."""

    if isinstance(label, str):
        if label not in ALL_CLASS_NAMES:
            raise ValueError(f"Etiqueta desconocida: {label}")
        return label

    class_name = INT_TO_CLASS.get(int(label))
    if class_name is None:
        raise ValueError(f"Etiqueta numerica desconocida: {label}")
    return class_name


def class_distribution(dataset: EpochDataset) -> Dict[str, int]:
    """
    Conteo de épocas por clase final, en formato legible. Siempre incluye
    las 5 clases de CLASS_NAMES; M y ? aparecen solo si hay épocas de ellas.
    """
    counts = {name: 0 for name in CLASS_NAMES}
    for label in dataset.y:
        name = label_to_class_name(label)
        counts[name] = counts.get(name, 0) + 1
    return counts


if __name__ == "__main__":
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
