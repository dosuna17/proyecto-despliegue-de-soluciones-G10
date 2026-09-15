"""API HTTP para inferencia con checkpoints de TinySleepNet registrados en MLflow."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Literal

import mlflow
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field, field_validator

from src.data.segmentation import CLASS_NAMES
from src.models.cnn import TinySleepNet, TinySleepNetConfig

SAMPLE_RATE_HZ = 100
EPOCH_SECONDS = 30
SAMPLES_PER_EPOCH = SAMPLE_RATE_HZ * EPOCH_SECONDS
SEQUENCE_LENGTH = 15
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://32.194.151.154:5000")
MLFLOW_ARTIFACT_PATH = os.getenv("MLFLOW_ARTIFACT_PATH", "tinysleepnet_final.pth")

app = FastAPI(title="Sleep Stage Classification API", version="0.3.0")


class PredictionRequest(BaseModel):
    """Quince épocas contiguas de EEG Fpz-Cz expresadas en microvoltios."""

    channel: Literal["EEG Fpz-Cz"] = "EEG Fpz-Cz"
    sampling_rate_hz: Literal[100] = SAMPLE_RATE_HZ
    signal_unit: Literal["uV"] = "uV"
    model_artifact: str = Field(
        default=MLFLOW_ARTIFACT_PATH,
        description=(
            "Nombre o ruta relativa exacta del checkpoint en MLflow. La API usa "
            "el run terminado más reciente que lo contiene."
        ),
    )
    epochs: List[List[float]] = Field(
        ...,
        description=(
            "15 épocas de EEG con 3.000 muestras cada una "
            "(30 segundos a 100 Hz), expresadas en microvoltios."
        ),
    )

    @field_validator("model_artifact")
    @classmethod
    def validate_model_artifact(cls, artifact: str) -> str:
        path = Path(artifact)
        if not artifact or path.is_absolute() or ".." in path.parts:
            raise ValueError("model_artifact debe ser una ruta relativa no vacía.")
        return path.as_posix()

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
    model_name: str
    channel: str
    sampling_rate_hz: int
    predictions: List[EpochPrediction]


def configured_local_model_path() -> Path | None:
    """Obtiene un checkpoint local solo si MODEL_PATH se configuró explícitamente."""

    value = os.getenv("MODEL_PATH")
    return Path(value).expanduser() if value else None


def find_run_id_for_artifact(client: MlflowClient, artifact_path: str) -> str:
    """Devuelve el run terminado más reciente que contiene el artefacto pedido."""

    parent = Path(artifact_path).parent.as_posix()
    parent = "" if parent == "." else parent
    for experiment in client.search_experiments(view_type=ViewType.ACTIVE_ONLY):
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="attributes.status = 'FINISHED'",
            order_by=["attributes.start_time DESC"],
            max_results=50_000,
        )
        for run in runs:
            artifacts = client.list_artifacts(run.info.run_id, path=parent)
            if any(not item.is_dir and item.path == artifact_path for item in artifacts):
                return run.info.run_id

    raise FileNotFoundError(
        f"No se encontró el artefacto '{artifact_path}' en ningún run terminado de MLflow."
    )


def model_path(artifact_path: str) -> Path:
    """Descarga un artefacto y resuelve su run por nombre."""

    local_path = configured_local_model_path()
    if local_path is not None and artifact_path == MLFLOW_ARTIFACT_PATH:
        return local_path

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    run_id = find_run_id_for_artifact(MlflowClient(), artifact_path)
    downloaded_path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path=artifact_path,
    )
    return Path(downloaded_path)


@lru_cache(maxsize=8)
def load_model(checkpoint_path: str) -> TinySleepNet:
    """Carga y conserva en caché los checkpoints usados recientemente."""

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"No se encontró el checkpoint de TinySleepNet en {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError("El checkpoint debe ser un state_dict de TinySleepNet.")

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


def model_unavailable(detail: Exception, artifact_path: str) -> HTTPException:
    return HTTPException(
        status_code=503,
        detail={
            "code": "model_unavailable",
            "message": "No fue posible cargar TinySleepNet para realizar inferencia.",
            "reason": str(detail),
            "model_artifact": artifact_path,
        },
    )


@app.get("/health")
def health() -> dict:
    """Informa la configuración usada para resolver modelos."""

    return {
        "status": "healthy",
        "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
        "default_model_artifact": MLFLOW_ARTIFACT_PATH,
        "local_model_path": str(configured_local_model_path() or ""),
        "sequence_length": SEQUENCE_LENGTH,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Clasifica las 15 épocas recibidas con el checkpoint seleccionado."""

    try:
        model = load_model(str(model_path(request.model_artifact)))
    except Exception as error:
        raise model_unavailable(error, request.model_artifact) from error

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
        model_name=request.model_artifact,
        channel=request.channel,
        sampling_rate_hz=request.sampling_rate_hz,
        predictions=predictions,
    )