"""API HTTP para inferencia con el checkpoint final de TinySleepNet."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Literal

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from src.data.segmentation import CLASS_NAMES
from src.models.cnn import TinySleepNet, TinySleepNetConfig

SAMPLE_RATE_HZ = 100
EPOCH_SECONDS = 30
SAMPLES_PER_EPOCH = SAMPLE_RATE_HZ * EPOCH_SECONDS
SEQUENCE_LENGTH = 15
#DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "tinysleepnet_final.pth"
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://32.194.151.154:5000",
)

MLFLOW_RUN_ID = os.getenv("MLFLOW_RUN_ID")

MLFLOW_ARTIFACT_PATH = os.getenv(
    "MLFLOW_ARTIFACT_PATH",
    "tinysleepnet_final.pth",
)

LOCAL_MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        str(
            Path(__file__).resolve().parents[2]
            / "models"
            / "tinysleepnet_final.pth"
        ),
    )
)
app = FastAPI(title="Sleep Stage Classification API", version="0.2.0")

class PredictionRequest(BaseModel):
    """Quince épocas contiguas de EEG Fpz-Cz expresadas en microvoltios."""

    channel: Literal["EEG Fpz-Cz"] = "EEG Fpz-Cz"
    sampling_rate_hz: Literal[100] = SAMPLE_RATE_HZ
    signal_unit: Literal["uV"] = "uV"
    epochs: List[List[float]] = Field(
        ...,
        description=(
            "15 épocas de EEG con 3.000 muestras cada una "
            "(30 segundos a 100 Hz), expresadas en microvoltios."
        ),
    )

    @field_validator("epochs")
    @classmethod
    def validate_epochs(cls, epochs: List[List[float]]) -> List[List[float]]:
        if len(epochs) != SEQUENCE_LENGTH:
            raise ValueError(f"Se requieren exactamente {SEQUENCE_LENGTH} épocas.")
        for index, epoch in enumerate(epochs):
            if len(epoch) != SAMPLES_PER_EPOCH:
                raise ValueError(
                    f"La época {index} debe contener exactamente "
                    f"{SAMPLES_PER_EPOCH} muestras."
                )
            if not np.isfinite(epoch).all():
                raise ValueError(f"La época {index} contiene muestras no finitas.")
        return epochs

class EpochPrediction(BaseModel):
    epoch_index: int
    predicted_stage: str
    confidence: float
    probabilities: Dict[str, float]

class PredictionResponse(BaseModel):
    model_name: str = "tinysleepnet_final"
    channel: str
    sampling_rate_hz: int
    predictions: List[EpochPrediction]

#def model_path() -> Path:
#    return Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))).expanduser()
def model_path() -> Path:
    if not MLFLOW_RUN_ID:
        raise RuntimeError(
            "No se ha definido MLFLOW_RUN_ID. "
            "Configure la variable de entorno con el Run ID de MLflow."
        )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    downloaded_path = mlflow.artifacts.download_artifacts(
        run_id=MLFLOW_RUN_ID,
        artifact_path=MLFLOW_ARTIFACT_PATH,
    )

    return Path(downloaded_path)

@lru_cache(maxsize=1)
def load_model(checkpoint_path: str) -> TinySleepNet:
    """Descarga/carga una única vez el checkpoint de TinySleepNet."""

    path = Path(checkpoint_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"No se encontró el checkpoint de TinySleepNet en {path}"
        )

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "El checkpoint debe ser un state_dict de TinySleepNet."
        )

    config = TinySleepNetConfig(
        sampling_rate=SAMPLE_RATE_HZ,
        input_size=SAMPLES_PER_EPOCH,
        n_classes=len(CLASS_NAMES),
        seq_length=SEQUENCE_LENGTH,
        use_rnn=True,
    )

    model = TinySleepNet(config)
    model.load_state_dict(checkpoint, strict=True)
    model.eval()

    return model
def model_unavailable(detail: Exception) -> HTTPException:
    return HTTPException(
        status_code=503,
        detail={
            "code": "model_unavailable",
            "message": "No fue posible cargar TinySleepNet para realizar inferencia.",
            "reason": str(detail),
            "expected_model_path": str(model_path()),
        },
    )

@app.get("/health")
def health() -> dict:
    """Informa el estado del servicio y la configuración del modelo."""

    return {
        "status": "healthy",
        "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
        "mlflow_run_id": MLFLOW_RUN_ID,
        "mlflow_artifact_path": MLFLOW_ARTIFACT_PATH,
        "model_path": str(LOCAL_MODEL_PATH),
        "sequence_length": SEQUENCE_LENGTH,
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Clasifica las 15 épocas recibidas con el modelo final TinySleepNet."""
    try:
        model = load_model(str(model_path()))
    except (FileNotFoundError, OSError, RuntimeError, TypeError) as error:
        raise model_unavailable(error) from error

    # (épocas, muestras) -> (lote=1, secuencia=15, canal=1, muestras=3000)
    signal = torch.tensor(request.epochs, dtype=torch.float32).unsqueeze(0).unsqueeze(2)
    with torch.inference_mode():
        result, _ = model.predict_proba(signal)

    probabilities = result["probabilities"].squeeze(0).cpu().tolist()
    predicted_classes = result["pred_class"].squeeze(0).cpu().tolist()
    confidences = result["confidence"].squeeze(0).cpu().tolist()
    predictions = [
        EpochPrediction(
            epoch_index=index,
            predicted_stage=CLASS_NAMES[predicted_class],
            confidence=confidence,
            probabilities=dict(zip(CLASS_NAMES, epoch_probabilities)),
        )
        for index, (predicted_class, confidence, epoch_probabilities) in enumerate(
            zip(predicted_classes, confidences, probabilities)
        )
    ]
    return PredictionResponse(
        channel=request.channel,
        sampling_rate_hz=request.sampling_rate_hz,
        predictions=predictions,
    )
