"""API simulada con el mismo contrato que app/api/main.py, para desarrollar y
probar el tablero sin MLflow ni modelos entrenados.

NO usa ningún modelo: "clasifica" por la frecuencia dominante del EEG, lo que
solo funciona con los registros de tests/datos_sinteticos.py. Nunca debe
desplegarse.

Uso (desde la raíz del repositorio):
    uvicorn app.dashboard.tests.api_simulada:app --port 8000
    # con GET /models:  API_SIMULADA_CON_MODELS=1 uvicorn ...
"""

from __future__ import annotations

import os
from itertools import product
from typing import Any, Dict, List

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

CANALES = ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal", "EMG submental"]
CLASES = ["W", "N1", "N2", "N3", "REM"]
EEG_OPCIONES = [[CANALES[0]], [CANALES[1]], CANALES[:2]]
EXTRAS_OPCIONES = [[], [CANALES[2]], [CANALES[3]], CANALES[2:]]

app = FastAPI(title="API simulada (solo pruebas del tablero)")


class Solicitud(BaseModel):
    channel_names: List[str]
    sampling_rate_hz: int = 100
    signal_unit: str = "uV"
    epochs: List[Any]


def _clasificar(epoca: np.ndarray, eog: np.ndarray | None, rng: np.random.Generator) -> Dict[str, float]:
    espectro = np.abs(np.fft.rfft(epoca - epoca.mean())) ** 2
    f = np.fft.rfftfreq(epoca.size, 1 / 100)
    banda = lambda a, b: espectro[(f >= a) & (f < b)].sum() + 1e-9  # noqa: E731
    total = banda(0.5, 30)
    delta, theta, alfa, sigma = banda(0.5, 2.5) / total, banda(4.5, 7) / total, banda(8, 11) / total, banda(12, 14.5) / total
    rem = 0.0 if eog is None else float(np.mean(np.abs(np.diff(eog)) > 30))
    puntajes = np.array([alfa * 3, theta * 2 * (rem < 0.002), sigma * 4, delta * 2, theta * 1.5 + rem * 40 + 0.1 * (theta > 0.3)])
    puntajes = puntajes + rng.normal(0, 0.15, 5)
    p = np.exp(puntajes * 4)
    return dict(zip(CLASES, (p / p.sum()).tolist()))


def _respuesta(epocas: np.ndarray, indices_eeg: List[int], indice_eog: int | None) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(int(abs(epocas[:, indices_eeg[0], :50].sum()) * 1000) % 2**32)
    salida = []
    for i, epoca in enumerate(epocas):
        eeg = np.mean([epoca[k] for k in indices_eeg], axis=0)
        probs = _clasificar(eeg, None if indice_eog is None else epoca[indice_eog], rng)
        estadio = max(probs, key=probs.get)
        salida.append({"epoch_index": i, "predicted_stage": estadio, "confidence": probs[estadio], "probabilities": probs})
    return salida


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "healthy", "modo": "simulado"}


if os.getenv("API_SIMULADA_CON_MODELS") == "1":
    @app.get("/models")
    def models() -> Dict[str, Any]:
        return {
            "modelo": "TinySleepNet (simulado)",
            "endpoint": "/predict-tinysleepnet",
            "epocas_por_llamada": 15,
            "canales_modelo": CANALES,
            "clases": CLASES,
            "combinaciones": [eeg + extras for eeg, extras in product(EEG_OPCIONES, EXTRAS_OPCIONES)],
        }


@app.post("/predict-tinysleepnet")
def predict_tinysleepnet(solicitud: Solicitud) -> Dict[str, Any]:
    esperado = CANALES + [f"presence {c}" for c in CANALES]
    if solicitud.channel_names != esperado:
        raise HTTPException(422, {"message": "channel_names debe respetar el orden fijo.", "expected": esperado})
    x = np.asarray(solicitud.epochs, dtype=np.float32)
    if x.ndim == 3:
        x = x[None, ...]
    if x.ndim != 4 or not 1 <= x.shape[0] <= 4 or x.shape[1:] != (15, 8, 3000):
        raise HTTPException(422, f"Forma inválida {x.shape}; se espera (lote de 1-4, 15, 8, 3000).")
    presencia = x[:, :, 4:8, 0]
    if not np.isin(presencia, (0, 1)).all() or not np.all(presencia[:, :, :2].any(axis=(1, 2))):
        raise HTTPException(422, "Debe llegar al menos un EEG con marca de presencia.")
    if not np.all(x[:, :, 4:8, :] == x[:, :, 4:8, :1]):
        raise HTTPException(422, "Cada marca de presencia debe mantenerse constante en la secuencia.")
    for k in range(4):
        faltante = presencia[:, :, k] == 0
        if np.any(faltante) and np.any(x[:, :, k, :][faltante] != 0):
            raise HTTPException(422, "Una señal marcada como ausente debe enviarse en ceros.")
    predicciones = []
    for indice_lote, secuencia in enumerate(x):
        presencia_secuencia = presencia[indice_lote, 0]
        eeg = [k for k in (0, 1) if presencia_secuencia[k] == 1]
        eog = 2 if presencia_secuencia[2] == 1 else None
        predicciones.extend(_respuesta(secuencia, eeg, eog))
    for indice, prediccion in enumerate(predicciones):
        prediccion["epoch_index"] = indice
    return {"model_name": "simulado", "channel": "TinySleepNet-8-canales", "sampling_rate_hz": 100,
            "predictions": predicciones}


