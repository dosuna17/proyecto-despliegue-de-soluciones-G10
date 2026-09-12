"""Utilidades para preparar archivos intermedios del proyecto."""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Sequence

import numpy as np

from src.data.loader import find_subject_pairs, load_subject
from src.data.segmentation import extract_epochs, trim_wake_edges


def generate_epoch_npz_files(
    data_dir: str | Path,
    out_dir: str | Path,
    channel_picks: Sequence[str] | None = ("EEG Fpz-Cz",),
    use_trim_wake_edges: bool = True,
    trim_buffer_epochs: int = 60,
    overwrite: bool = False,
) -> None:
    """Genera archivos .npz por grabacion a partir de pares PSG/Hypnogram.

    Parameters
    ----------
    data_dir:
        Carpeta donde estan los archivos `*-PSG.edf` y `*-Hypnogram.edf`.
    out_dir:
        Carpeta donde se guardan los archivos `{recording_id}_epochs.npz`.
    channel_picks:
        Canales a extraer. Usa `None` para conservar todos los canales.
    use_trim_wake_edges:
        Si es True, aplica `trim_wake_edges` a cada grabacion antes de guardar.
    trim_buffer_epochs:
        Buffer de epocas W que se conserva antes/despues del sueño real.
    overwrite:
        Si es True, reprocesa aunque el archivo .npz ya exista.
    """

    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = find_subject_pairs(str(data_dir))
    print(f"{len(pairs)} par(es) encontrados")

    for psg_path, hypno_path in pairs:
        recording_id = os.path.basename(psg_path).split("-")[0]
        out_path = out_dir / f"{recording_id}_epochs.npz"
        subj = None
        ds = None

        if out_path.exists() and not overwrite:
            print(f"[skip] {recording_id} ya procesado")
            continue

        try:
            subj = load_subject(psg_path, hypno_path)
            ds = extract_epochs(
                subj.raw,
                subj.annotations,
                recording_id=recording_id,
                channel_picks=channel_picks,
            )

            if use_trim_wake_edges:
                ds = trim_wake_edges(ds, buffer_epochs=trim_buffer_epochs)

            np.savez_compressed(
                out_path,
                X=ds.X,
                y=ds.y,
                recording_ids=ds.recording_ids.astype(str),
                subject_codes=ds.subject_codes.astype(str),
                channel_names=np.array(ds.channel_names),
                sfreq=ds.sfreq,
                signal_unit=ds.signal_unit,
            )
            print(f"[ok] {recording_id}: {ds.X.shape}")

        except Exception as exc:
            print(f"[error] {recording_id}: {exc}")

        finally:
            del subj, ds
            gc.collect()
