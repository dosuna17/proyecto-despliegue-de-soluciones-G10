"""
Extracción de features espectrales por época a partir de un EpochDataset
(la señal cruda producida por segmentation.py).

Objetivo del proyecto: comparar modelos tabulares (Random Forest, XGBoost,
SVM, etc.) contra redes neuronales que reciben la señal cruda. Este módulo
alimenta la rama "tabular" -- la rama "red neuronal" sigue usando
directamente `EpochDataset.X` de segmentation.py, sin pasar por aquí.

Ambas ramas comparten el mismo `y`, `recording_ids` y `subject_codes`,
para que el split por sujeto sea idéntico y la comparación sea justa.

Features calculadas por canal, usando el método de Welch para estimar
la densidad espectral de potencia (PSD) de cada época de 30s:
    - Potencia absoluta y relativa en 5 bandas clásicas de EEG:
      delta, theta, alpha, sigma (spindle), beta.
    - Razones entre bandas (delta/alpha, theta/alpha).
    - Spectral Edge Frequency al 95% (SEF95).
    - Entropía espectral (Shannon, normalizada).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.signal import welch
from pathlib import Path
import sys

project_root = Path.cwd().parent
sys.path.insert(0, str(project_root))

from src.data.segmentation import EpochDataset

# numpy >= 2.0 renombró trapz -> trapezoid; se soporta ambos casos.
_trapz = getattr(np, "trapezoid", None) or np.trapz

# Bandas clásicas de EEG (Hz): nombre -> (f_min, f_max)
FREQ_BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "sigma": (12.0, 16.0),   # banda de husos de sueño (spindles), se solapa con alpha/beta a propósito
    "beta": (13.0, 30.0),
}

# Rango total sobre el que se calcula la potencia de referencia (relativa, SEF, entropía)
TOTAL_BAND: Tuple[float, float] = (0.5, 30.0)


@dataclass
class FeatureDataset:
    """
    Features espectrales por época, en formato tabular. Comparte el mismo
    esquema de metadata que EpochDataset (recording_ids, subject_codes)
    para que el split por sujeto sea reproducible entre la rama tabular
    y la rama de red neuronal.
    """
    recording_id: str
    subject_code: str
    X: np.ndarray               # (n_epochs, n_features)
    y: np.ndarray                # (n_epochs,) — idéntico al del EpochDataset de origen
    feature_names: List[str]
    recording_ids: np.ndarray    # (n_epochs,) str
    subject_codes: np.ndarray    # (n_epochs,) str


def _band_power(freqs: np.ndarray, psd: np.ndarray, band: Tuple[float, float]) -> float:
    """Integra la PSD (trapecio) dentro de un rango de frecuencias [f_min, f_max]."""
    f_min, f_max = band
    mask = (freqs >= f_min) & (freqs < f_max)
    if not np.any(mask):
        return 0.0
    return float(_trapz(psd[mask], freqs[mask]))


def _spectral_edge_freq(freqs: np.ndarray, psd: np.ndarray, edge: float = 0.95) -> float:
    """Frecuencia bajo la cual cae `edge` (p. ej. 95%) de la potencia total."""
    cumulative = np.cumsum(psd)
    total = cumulative[-1]
    if total <= 0:
        return 0.0
    idx = np.searchsorted(cumulative, edge * total)
    idx = min(idx, len(freqs) - 1)
    return float(freqs[idx])


def _spectral_entropy(psd: np.ndarray) -> float:
    """Entropía de Shannon normalizada [0, 1] de la distribución de potencia espectral."""
    psd_sum = psd.sum()
    if psd_sum <= 0:
        return 0.0
    p = psd / psd_sum
    p = p[p > 0]  # evitar log(0)
    entropy = -np.sum(p * np.log2(p))
    max_entropy = np.log2(len(psd))
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


def _epoch_channel_features(signal: np.ndarray, sfreq: float) -> Dict[str, float]:
    """
    Calcula todas las features espectrales para una sola señal 1D
    (un canal, una época). Devuelve un diccionario {nombre_feature: valor}.
    """
    freqs, psd = welch(signal, fs=sfreq, nperseg=min(len(signal), int(sfreq * 4)))

    feats: Dict[str, float] = {}

    total_power = _band_power(freqs, psd, TOTAL_BAND)
    for band_name, band_range in FREQ_BANDS.items():
        abs_power = _band_power(freqs, psd, band_range)
        feats[f"{band_name}_power_abs"] = abs_power
        feats[f"{band_name}_power_rel"] = abs_power / total_power if total_power > 0 else 0.0

    feats["delta_alpha_ratio"] = (
        feats["delta_power_abs"] / feats["alpha_power_abs"] if feats["alpha_power_abs"] > 0 else 0.0
    )
    feats["theta_alpha_ratio"] = (
        feats["theta_power_abs"] / feats["alpha_power_abs"] if feats["alpha_power_abs"] > 0 else 0.0
    )
    feats["sef95"] = _spectral_edge_freq(freqs, psd, edge=0.95)
    feats["spectral_entropy"] = _spectral_entropy(psd)

    return feats


def extract_features(dataset: EpochDataset) -> FeatureDataset:
    """
    Calcula las features espectrales de cada época de un EpochDataset,
    para cada canal presente. Si hay más de un canal, los nombres de
    feature se prefijan con el nombre del canal (p. ej.
    'EEG_Fpz-Cz__delta_power_rel') para no mezclarlos.
    """
    n_epochs, n_channels, _ = dataset.X.shape
    feature_rows: List[List[float]] = []
    feature_names: List[str] = []

    for epoch_idx in range(n_epochs):
        row: List[float] = []
        row_names: List[str] = []
        for ch_idx in range(n_channels):
            ch_name = dataset.channel_names[ch_idx].replace(" ", "_")
            signal = dataset.X[epoch_idx, ch_idx, :]
            feats = _epoch_channel_features(signal, dataset.sfreq)
            for feat_name, value in feats.items():
                row.append(value)
                if epoch_idx == 0:  # los nombres son iguales para todas las épocas
                    row_names.append(f"{ch_name}__{feat_name}")
        feature_rows.append(row)
        if epoch_idx == 0:
            feature_names = row_names

    X_features = np.array(feature_rows, dtype=np.float32)

    return FeatureDataset(
        recording_id=dataset.recording_id,
        subject_code=dataset.subject_code,
        X=X_features,
        y=dataset.y.copy(),
        feature_names=feature_names,
        recording_ids=dataset.recording_ids.copy(),
        subject_codes=dataset.subject_codes.copy(),
    )


def to_dataframe(features: FeatureDataset):
    """
    Conveniencia: convierte un FeatureDataset a un pandas.DataFrame
    (una fila por época, columnas = features + y + metadata). Útil para
    EDA rápido o para pasarlo directo a un modelo tabular tipo sklearn.
    Import de pandas local para no forzar la dependencia si no se usa.
    """
    import pandas as pd

    df = pd.DataFrame(features.X, columns=features.feature_names)
    df["y"] = features.y
    df["recording_id"] = features.recording_ids
    df["subject_code"] = features.subject_codes
    return df


if __name__ == "__main__":
    import sys
    from loader import load_subject
    from segmentation import extract_epochs

    psg_path = sys.argv[1] if len(sys.argv) > 1 else "SC4001E0-PSG.edf"
    hypno_path = sys.argv[2] if len(sys.argv) > 2 else "SC4001EC-Hypnogram.edf"

    subject = load_subject(psg_path, hypno_path)
    epoch_ds = extract_epochs(
        subject.raw, subject.annotations,
        recording_id=subject.subject_id,
        channel_picks=["EEG Fpz-Cz"],
    )
    feat_ds = extract_features(epoch_ds)

    print(f"X_features shape: {feat_ds.X.shape}  (n_epochs, n_features)")
    print(f"N° features: {len(feat_ds.feature_names)}")
    print(f"Features: {feat_ds.feature_names}")
    print(f"Ejemplo (época 1050): {feat_ds.X[1050]}")
