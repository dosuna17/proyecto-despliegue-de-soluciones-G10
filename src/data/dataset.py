"""
Dataset y DataLoader de PyTorch para el modelo avanzado (TinySleepNet).

DECISIÓN DE DISEÑO (documentar en el informe): el paper original
(Supratak & Guo, 2020, Sección II.C) propone dos técnicas de data
augmentation durante el entrenamiento -- "sequence augmentation"
(offset aleatorio al cortar secuencias) y "signal augmentation"
(desplazamiento temporal de la señal dentro de cada época). Ambas
técnicas se implementaron y se validaron funcionalmente, pero se
decidió NO activarlas en el pipeline final de entrenamiento, para:
  - mantener el procesamiento de datos simple y consistente con el
    preprocesamiento determinístico usado en el modelo baseline
    (Santiago), facilitando la comparabilidad y la reproducibilidad
    entre ambos modelos;
  - reducir la complejidad y el riesgo de errores dado el tiempo
    disponible para el proyecto.
Este módulo usa, por tanto, un corte FIJO y DETERMINÍSTICO de
secuencias, tanto para entrenamiento como para validación/test.

Hiperparámetros de la Sección III.B del paper que SÍ se adoptan
(no están relacionados con augmentation): seq_length=15, batch_size=20.

Carácter temporal de split_utils: src/utils/splits.py todavía no existe.
load_split() lee el Excel directamente; aislado en su propia función
para que el swap a splits.py oficial sea trivial cuando exista.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from torch.utils.data import Dataset, DataLoader

NPZ_DIR = Path("data/interim/epochs_por_sujeto_recortado")  # antes: epochs_por_sujeto
SPLIT_XLSX_PATH = Path("data/raw/SC-subjects-split.xlsx")

N_FOLDS_DEFAULT = 5           # decisión propia (cross-validation, como Santiago)
DEFAULT_SEQ_LENGTH = 15       # paper: seq_length=15
DEFAULT_BATCH_SIZE = 20       # paper: batch_size=20


# ---------------------------------------------------------------------
# 1. Carga del split (temporal -- ver docstring del módulo)
# ---------------------------------------------------------------------
def load_split(xlsx_path: Path = SPLIT_XLSX_PATH) -> pd.DataFrame:
    """
    Lee SC-subjects-split.xlsx -> DataFrame con columnas:
        subject_code : str, padding a 2 dígitos (ej. "01")
        split        : "train" | "test"
    """
    df = pd.read_excel(xlsx_path)
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})

    if "subject" not in df.columns or "split" not in df.columns:
        raise ValueError(
            f"Se esperaban columnas 'subject' y 'split' en {xlsx_path}, "
            f"se encontraron: {list(df.columns)}"
        )

    df["subject_code"] = df["subject"].astype(int).astype(str).str.zfill(2)
    df["split"] = df["split"].str.strip().str.lower()

    return df[["subject_code", "split"]].drop_duplicates()


def get_train_test_subject_codes(
    xlsx_path: Path = SPLIT_XLSX_PATH,
) -> Tuple[List[str], List[str]]:
    """Devuelve (subject_codes_train, subject_codes_test)."""
    split_df = load_split(xlsx_path)
    train_codes = split_df.loc[split_df["split"] == "train", "subject_code"].tolist()
    test_codes = split_df.loc[split_df["split"] == "test", "subject_code"].tolist()
    return sorted(train_codes), sorted(test_codes)


# ---------------------------------------------------------------------
# 2. Carga de los .npz y filtrado por sujeto
# ---------------------------------------------------------------------
@dataclass
class SubjectRecording:
    """Una grabación (sujeto/noche) cargada desde su .npz."""
    recording_id: str
    subject_code: str
    X: np.ndarray  # (n_epochs, n_channels, n_samples)
    y: np.ndarray  # (n_epochs,)


def load_npz_recordings(
    npz_dir: Path = NPZ_DIR,
    subject_codes: Optional[Sequence[str]] = None,
    max_epochs_per_recording: Optional[int] = None,
    seed: int = 42,
) -> List[SubjectRecording]:
    """
    Carga todos los .npz en npz_dir. Si se pasa `subject_codes`, filtra
    solo las grabaciones cuyo subject_code esté en esa lista.

    max_epochs_per_recording: si se especifica, cada grabación se recorta
    a una VENTANA CONTINUA ALEATORIA de a lo sumo ese número de épocas
    (en vez de la noche completa). Decisión de diseño por restricciones
    de memoria/tiempo de cómputo (documentar en el informe): cargar las
    ~153 grabaciones completas puede exceder la RAM disponible en un
    portátil típico. Se usa una ventana continua (no un muestreo disperso)
    para preservar la coherencia temporal que necesita la LSTM, y
    aleatoria (no siempre desde el inicio) para no sesgar hacia la
    vigilia larga del comienzo/final de cada noche.
    """
    npz_files = sorted(npz_dir.glob("*_epochs.npz"))
    if not npz_files:
        raise FileNotFoundError(
            f"No se encontraron .npz en {npz_dir}. ¿Ya corriste generar_npz.py?"
        )

    subject_codes_set = set(subject_codes) if subject_codes is not None else None
    rng = np.random.default_rng(seed)

    recordings = []
    for path in npz_files:
        data = np.load(path, allow_pickle=True)
        subject_code = str(data["subject_code"])

        if subject_codes_set is not None and subject_code not in subject_codes_set:
            continue

        X, y = data["X"], data["y"]
        if max_epochs_per_recording is not None and X.shape[0] > max_epochs_per_recording:
            max_start = X.shape[0] - max_epochs_per_recording
            start = int(rng.integers(0, max_start + 1))
            X = X[start:start + max_epochs_per_recording]
            y = y[start:start + max_epochs_per_recording]

        recordings.append(
            SubjectRecording(
                recording_id=str(data["recording_id"]),
                subject_code=subject_code,
                X=X,
                y=y,
            )
        )
    return recordings


# ---------------------------------------------------------------------
# 3. K-fold por sujeto (cross-validation, siguiendo el enfoque de Santiago)
# ---------------------------------------------------------------------
def make_subject_kfold_splits(
    train_subject_codes: Sequence[str],
    n_folds: int = N_FOLDS_DEFAULT,
) -> List[Tuple[List[str], List[str]]]:
    """
    Divide los sujetos de train en n_folds, a nivel de SUJETO. Devuelve
    una lista de (fold_train_codes, fold_val_codes).
    """
    codes = np.array(sorted(train_subject_codes))
    gkf = GroupKFold(n_splits=n_folds)
    dummy_X = np.zeros(len(codes))
    splits = []
    for train_idx, val_idx in gkf.split(dummy_X, groups=codes):
        splits.append((codes[train_idx].tolist(), codes[val_idx].tolist()))
    return splits


# ---------------------------------------------------------------------
# 4. Dataset de PyTorch: corte fijo y determinístico (sin augmentation)
# ---------------------------------------------------------------------
class SleepSequenceDataset(Dataset):
    """
    Cada elemento es una secuencia de `seq_length` épocas CONSECUTIVAS
    de una misma grabación (nunca se mezclan sujetos/noches dentro de
    una secuencia). Corte fijo desde la época 0, en bloques no
    solapados; el resto final (< seq_length) se descarta. Igual para
    train, val y test -- sin aleatoriedad, totalmente reproducible.
    """

    def __init__(self, recordings: List[SubjectRecording], seq_length: int = DEFAULT_SEQ_LENGTH):
        self.seq_length = seq_length
        self.sequences: List[Tuple[np.ndarray, np.ndarray]] = []

        for rec in recordings:
            n_epochs = rec.X.shape[0]
            n_full = n_epochs // seq_length
            for i in range(n_full):
                s, e = i * seq_length, (i + 1) * seq_length
                self.sequences.append((rec.X[s:e], rec.y[s:e]))

        if not self.sequences:
            raise ValueError(
                "No se generó ninguna secuencia -- revisa que seq_length "
                "no sea mayor que el número de épocas de tus grabaciones."
            )

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        X_seq, y_seq = self.sequences[idx]
        return torch.from_numpy(X_seq).float(), torch.from_numpy(y_seq).long()


def build_dataloader(
    recordings: List[SubjectRecording],
    seq_length: int = DEFAULT_SEQ_LENGTH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    shuffle: bool = True,
) -> DataLoader:
    """Envoltorio de conveniencia: recordings -> DataLoader listo para entrenar."""
    dataset = SleepSequenceDataset(recordings, seq_length=seq_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


# ---------------------------------------------------------------------
# 5. Class weights (SOLO desde el train de cada fold)
# ---------------------------------------------------------------------
def compute_class_weights(
    recordings: List[SubjectRecording], n_classes: int = 5, smoothing: str = "sqrt_inverse"
) -> List[float]:
    """
    Pesos por clase desde la distribución de un conjunto de grabaciones
    (debe ser SOLO el train del fold). smoothing="sqrt_inverse":
    peso = 1/sqrt(frecuencia). Nota: el paper usa un esquema mucho más
    simple (solo N1=1.5, el resto=1); este esquema suavizado sobre las
    5 clases es una decisión propia, documentada como tal en el informe.
    """
    all_y = np.concatenate([rec.y for rec in recordings])
    counts = np.bincount(all_y, minlength=n_classes).astype(np.float64)
    freqs = counts / counts.sum()

    if smoothing == "sqrt_inverse":
        weights = 1.0 / np.sqrt(freqs + 1e-8)
    elif smoothing == "inverse":
        weights = 1.0 / (freqs + 1e-8)
    else:
        raise ValueError(f"smoothing desconocido: {smoothing}")

    weights = weights / weights.mean()
    return weights.tolist()


# ---------------------------------------------------------------------
# Prueba rápida manual
# ---------------------------------------------------------------------
if __name__ == "__main__":
    train_codes, test_codes = get_train_test_subject_codes()
    print(f"Sujetos train: {len(train_codes)}  |  Sujetos test: {len(test_codes)}")

    folds = make_subject_kfold_splits(train_codes, n_folds=N_FOLDS_DEFAULT)
    print(f"Folds generados: {len(folds)}")
    for i, (fold_train, fold_val) in enumerate(folds):
        overlap = set(fold_train) & set(fold_val)
        print(f"  Fold {i}: {len(fold_train)} train / {len(fold_val)} val (overlap={len(overlap)})")

    fold_train_codes, fold_val_codes = folds[0]
    train_recs = load_npz_recordings(subject_codes=fold_train_codes)
    val_recs = load_npz_recordings(subject_codes=fold_val_codes)
    print(f"\nFold 0 -> {len(train_recs)} grabaciones train, {len(val_recs)} grabaciones val")

    weights = compute_class_weights(train_recs)
    print(f"Class weights (fold 0, train): {[f'{w:.3f}' for w in weights]}")

    train_loader = build_dataloader(train_recs, seq_length=DEFAULT_SEQ_LENGTH, batch_size=DEFAULT_BATCH_SIZE)
    X_batch, y_batch = next(iter(train_loader))
    print(f"\nBatch de ejemplo -> X: {X_batch.shape}, y: {y_batch.shape}")
    print("Esperado: X=(batch=20, seq_length=15, 1, 3000), y=(20, 15)")
