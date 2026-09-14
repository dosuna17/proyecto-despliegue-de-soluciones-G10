"""
Aplica trim_wake_edges() a los .npz YA GENERADOS por generar_npz.py,
sin necesidad de releer los .edf crudos (rápido).

Contexto (9/4/2026): Santiago confirmó que su pipeline de baseline usa
trim_wake_edges(buffer_epochs=60) por defecto (ver utils.py, parámetro
use_trim_wake_edges=True). Para que el modelo avanzado sea comparable,
se aplica la MISMA transformación sobre los .npz ya existentes.

Uso:
    python src\\data\\aplicar_recorte_vigilia.py

Genera una copia nueva en data/interim/epochs_por_sujeto_recortado/,
dejando intactos los .npz originales (sin recorte) por si se necesitan
después para comparar con/sin recorte.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))

# Reimplementación local de trim_wake_edges, idéntica a la de Santiago
# (segmentation.py), para no depender de tener su módulo completo en
# esta rama -- opera directamente sobre los arrays X/y de un .npz.
CLASS_TO_INT = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}
WAKE_INT = CLASS_TO_INT["W"]
WAKE_LABEL = "W"


def trim_wake_edges_arrays(X: np.ndarray, y: np.ndarray, buffer_epochs: int = 60):
    """
    Misma lógica que trim_wake_edges() de segmentation.py, operando
    directamente sobre arrays X/y en vez de un objeto EpochDataset.
    """
    if y.dtype.kind in ("U", "S", "O"):
        is_wake = (y == WAKE_LABEL)
    else:
        is_wake = (y == WAKE_INT)
    non_wake_idx = np.where(~is_wake)[0]

    if len(non_wake_idx) == 0:
        raise ValueError("La grabación no tiene ninguna época de sueño real (todo es W)")

    first_sleep = non_wake_idx.min()
    last_sleep = non_wake_idx.max()
    start = max(0, first_sleep - buffer_epochs)
    end = min(len(y), last_sleep + buffer_epochs + 1)

    return X[start:end], y[start:end]


def main(
    src_dir: str = "data/interim/epochs_por_sujeto",
    dst_dir: str = "data/interim/epochs_por_sujeto_recortado",
    buffer_epochs: int = 60,
):
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)
    dst_path.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(src_path.glob("*_epochs.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No se encontraron .npz en {src_path}")

    print(f"Se encontraron {len(npz_files)} archivos. Aplicando recorte (buffer={buffer_epochs} épocas)...")

    total_before, total_after = 0, 0
    failed = []

    for i, path in enumerate(npz_files, start=1):
        data = np.load(path, allow_pickle=True)
        X, y = data["X"], data["y"]
        n_before = X.shape[0]

        try:
            X_trimmed, y_trimmed = trim_wake_edges_arrays(X, y, buffer_epochs=buffer_epochs)
        except ValueError as e:
            print(f"[{i}/{len(npz_files)}] SALTADO {path.name}: {e}")
            failed.append(path.name)
            continue

        n_after = X_trimmed.shape[0]
        total_before += n_before
        total_after += n_after

        out_path = dst_path / path.name
        np.savez_compressed(
            out_path,
            X=X_trimmed,
            y=y_trimmed,
            recording_ids=data["recording_ids"][:n_after] if len(data["recording_ids"]) >= n_after else np.full(n_after, str(data["recording_id"]), dtype=object),
            subject_codes=np.full(n_after, str(data["subject_code"]), dtype=object),
            channel_names=data["channel_names"],
            sfreq=data["sfreq"],
            signal_unit=data["signal_unit"],
            dropped_epochs=int(data["dropped_epochs"]) if "dropped_epochs" in data else 0,
            recording_id=str(data["recording_id"]),
            subject_code=str(data["subject_code"]),
            night=str(data["night"]) if "night" in data else "0",
        )
        print(f"[{i}/{len(npz_files)}] OK {path.name}: {n_before} -> {n_after} épocas "
              f"({n_before - n_after} recortadas)")

    print("\n" + "=" * 60)
    print(f"Total épocas antes:   {total_before:,}")
    print(f"Total épocas después: {total_after:,}")
    print(f"Reducción: {100*(1 - total_after/total_before):.1f}%")
    if failed:
        print(f"\nGrabaciones saltadas (sin sueño real, revisar manualmente): {failed}")
    print(f"\nArchivos recortados guardados en: {dst_path.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
