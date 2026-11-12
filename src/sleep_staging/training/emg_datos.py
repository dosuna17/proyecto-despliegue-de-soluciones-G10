"""
Carga de grabaciones con el EMG submental como cuarta posicion.

El EMG viene a 1 Hz (30 valores por epoca de 30 s) y se guarda aparte, en una
carpeta con un .npz por grabacion (X de forma (n_epocas, 1, 30)), generada con
scripts/generar_emg.py. Aqui se lee junto a los .npz de 3 canales y se anade
como canal 4, repetido 100 veces por valor (30 x 100 = 3000 muestras); la rama
EMG del modelo recupera los 30 valores originales con [::100].

Las ventanas de epocas por grabacion (max_epochs_per_recording) se eligen igual
que en load_npz_recordings de data/dataset.py (misma semilla y mismo orden), asi
que los pliegues y las secuencias coinciden con los del modelo de 3 senales.

Carpeta del EMG: variable de entorno SLEEP_EMG_DIR (por defecto
data/interim/epochs_emg_recortado).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))  # carpeta src/
from data import dataset as ds_mod  # noqa: E402
from data.dataset import CLASS_TO_INT, SubjectRecording  # noqa: E402

REPETIR = 100   # debe coincidir con REPETIR_EMG de canales_flexibles y REPETIR del modelo
EMG_DIR = Path(os.environ.get("SLEEP_EMG_DIR", "data/interim/epochs_emg_recortado"))


def cargar_con_emg(
    npz_dir: Optional[Path] = None,
    subject_codes: Optional[Sequence[str]] = None,
    max_epochs_per_recording: Optional[int] = None,
    seed: int = 42,
    emg_dir: Optional[Path] = None,
) -> List[SubjectRecording]:
    """Igual que load_npz_recordings, pero cada grabacion trae 4 canales: Fpz-Cz, Pz-Oz, EOG y EMG."""
    npz_dir = Path(npz_dir) if npz_dir is not None else ds_mod.NPZ_DIR
    emg_dir = Path(emg_dir) if emg_dir is not None else EMG_DIR
    archivos = sorted(npz_dir.glob("*_epochs.npz"))
    if not archivos:
        raise FileNotFoundError(f"No se encontraron .npz en {npz_dir}.")
    if not emg_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta del EMG: {emg_dir}. Genera los datos con "
                                f"scripts/generar_emg.py o define SLEEP_EMG_DIR.")
    codigos = set(subject_codes) if subject_codes is not None else None
    rng = np.random.default_rng(seed)

    recordings: List[SubjectRecording] = []
    for ruta in archivos:
        d = np.load(ruta, allow_pickle=True)
        codigo = str(d["subject_code"])
        if codigos is not None and codigo not in codigos:
            continue
        X, y = d["X"], d["y"]
        ruta_emg = emg_dir / ruta.name
        if not ruta_emg.exists():
            raise FileNotFoundError(f"Falta el EMG de {ruta.name} en {emg_dir}.")
        e = np.load(ruta_emg, allow_pickle=True)
        E, ye = e["X"], e["y"]
        if len(y) != len(ye) or not (y == ye).all():
            raise ValueError(f"{ruta.name}: las etiquetas del EMG no coinciden con las de 3 canales.")
        if E.ndim != 3 or E.shape[1] != 1 or E.shape[2] != 30:
            raise ValueError(f"{ruta.name}: el EMG debe tener forma (n_epocas, 1, 30); llego {E.shape}.")
        if y.dtype.kind in ("U", "S", "O"):
            y = np.array([CLASS_TO_INT[str(c)] for c in y], dtype=np.int64)
        else:
            y = y.astype(np.int64)
        if max_epochs_per_recording is not None and X.shape[0] > max_epochs_per_recording:
            inicio = int(rng.integers(0, X.shape[0] - max_epochs_per_recording + 1))
            sl = slice(inicio, inicio + max_epochs_per_recording)
            X, y, E = X[sl], y[sl], E[sl]
        E3000 = np.repeat(E, REPETIR, axis=2).astype(np.float32)           # (n, 1, 3000)
        recordings.append(SubjectRecording(
            recording_id=str(d["recording_id"]), subject_code=codigo,
            X=np.concatenate([X, E3000], axis=1), y=y,
        ))
    return recordings
