"""
Extracción de features espectrales por época a partir de un EpochDataset
(la señal cruda producida por segmentation.py).

Features calculadas por canal, usando el método de Welch para estimar
la densidad espectral de potencia (PSD) de cada época de 30s:
    - Potencia absoluta y relativa en 5 bandas clásicas de EEG:
      delta, theta, alpha, sigma, beta.
    - Razones entre bandas (delta/alpha, theta/alpha).
    - Spectral Edge Frequency al 95% (SEF95).
    - Entropía espectral (Shannon, normalizada).

En los canales EOG se agregan además features en el dominio del tiempo:
    - Amplitud pico a pico robusta (percentil 99 - percentil 1).
    - Movilidad de Hjorth.
    - Conteo de sacadas (picos de velocidad sobre un umbral fijo).
    - Curtosis.

Los canales EMG no admiten análisis por bandas (ver EMG_NATIVE_SFREQ), así
que en lugar de las features espectrales se calculan:
    - Media y desviación estándar de la época.
    - Percentiles 10 y 90 de la época.
    - Pendiente de la recta ajustada dentro de la época (µV/s).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.signal import find_peaks, welch
from scipy.stats import kurtosis
from pathlib import Path
import sys

project_root = Path.cwd().parent
sys.path.insert(0, str(project_root))

from src.data.segmentation import EpochDataset

_trapz = getattr(np, "trapezoid", None) or np.trapz

FREQ_BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "sigma": (12.0, 16.0), 
    "beta": (16.0, 30.0),
}


TOTAL_BAND: Tuple[float, float] = (0.5, 30.0)
EOG_CHANNEL_PREFIX = "EOG"
SACCADE_VELOCITY_UV_S = 750.0
SACCADE_MIN_INTERVAL_SEC = 0.3
SACCADE_SMOOTH_SEC = 0.1
EMG_CHANNEL_PREFIX = "EMG"
EMG_NATIVE_SFREQ = 1.0


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
    X: np.ndarray
    y: np.ndarray
    feature_names: List[str]
    recording_ids: np.ndarray
    subject_codes: np.ndarray


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
    
    band_mask = (freqs >= TOTAL_BAND[0]) & (freqs < TOTAL_BAND[1])
    feats["sef95"] = _spectral_edge_freq(freqs[band_mask], psd[band_mask], edge=0.95)
    feats["spectral_entropy"] = _spectral_entropy(psd[band_mask])

    return feats


def eog_p2p_robust(signal: np.ndarray) -> float:
    """Amplitud pico a pico sin los extremos: percentil 99 - percentil 1."""
    p1, p99 = np.percentile(signal, [1, 99])
    return float(p99 - p1)


def eog_hjorth_mobility(signal: np.ndarray, sfreq: float) -> float:
    """Movilidad de Hjorth, sqrt(Var(x') / Var(x)), con x' en unidades/s."""
    var_x = np.var(signal)
    if var_x <= 0:
        return np.nan
    derivative = np.diff(signal) * sfreq
    return float(np.sqrt(np.var(derivative) / var_x))


def eog_saccade_count(
    signal: np.ndarray,
    sfreq: float,
    threshold: float = SACCADE_VELOCITY_UV_S,
    smooth_sec: float = SACCADE_SMOOTH_SEC,
) -> float:
    """Cantidad de picos de velocidad (|x'|, µV/s) que superan `threshold`."""
    smooth_len = max(round(smooth_sec * sfreq), 1)
    if smooth_len > 1:
        signal = np.convolve(signal, np.ones(smooth_len) / smooth_len, mode="valid")
    velocity = np.abs(np.diff(signal)) * sfreq
    distance = max(round(SACCADE_MIN_INTERVAL_SEC * sfreq), 1)
    peaks, _ = find_peaks(velocity, height=threshold, distance=distance)
    return float(len(peaks))


def eog_kurtosis(signal: np.ndarray) -> float:
    """Curtosis de Fisher (0 para una señal gaussiana)."""
    if np.std(signal) <= 0:
        return np.nan
    return float(kurtosis(signal, fisher=True))


def _epoch_eog_features(signal: np.ndarray, sfreq: float) -> Dict[str, float]:
    """Features espectrales de un canal EOG más sus features en el dominio del tiempo."""
    feats = _epoch_channel_features(signal, sfreq)
    feats["p2p_robust"] = eog_p2p_robust(signal)
    feats["hjorth_mobility"] = eog_hjorth_mobility(signal, sfreq)
    feats["saccade_count"] = eog_saccade_count(signal, sfreq)
    feats["kurtosis"] = eog_kurtosis(signal)
    return feats


def _epoch_emg_features(signal: np.ndarray, sfreq: float) -> Dict[str, float]:
    """
    Calcula las features de EMG para una sola señal 1D (un canal, una época),
    usando solo las muestras originales a EMG_NATIVE_SFREQ. Devuelve un
    diccionario {nombre_feature: valor}.
    """
    step = max(int(round(sfreq / EMG_NATIVE_SFREQ)), 1)
    samples = signal[::step]
    time_sec = np.arange(len(samples)) * step / sfreq
    p10, p90 = np.percentile(samples, [10, 90])

    return {
        "mean": float(np.mean(samples)),
        "std": float(np.std(samples)),
        "p10": float(p10),
        "p90": float(p90),
        "slope": float(np.polyfit(time_sec, samples, 1)[0]),
    }


def extract_features(dataset: EpochDataset) -> FeatureDataset:
    """
    Calcula las features espectrales de cada época de un EpochDataset,
    para cada canal presente: espectrales en EEG, espectrales y de tiempo
    en EOG, y estadísticas en EMG. Si hay más de un canal, los nombres de
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
            if ch_name.startswith(EMG_CHANNEL_PREFIX):
                feats = _epoch_emg_features(signal, dataset.sfreq)
            elif ch_name.startswith(EOG_CHANNEL_PREFIX):
                feats = _epoch_eog_features(signal, dataset.sfreq)
            else:
                feats = _epoch_channel_features(signal, dataset.sfreq)
            for feat_name, value in feats.items():
                row.append(value)
                if epoch_idx == 0:
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
