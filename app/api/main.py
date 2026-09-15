"""API HTTP para inferencia con checkpoints de TinySleepNet registrados en MLflow."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Literal

import mlflow
import joblib
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field, field_validator

from src.data.segmentation import CLASS_NAMES, EpochDataset
from src.preprocessing.feature_engineering import extract_features
from src.models.cnn import TinySleepNet, TinySleepNetConfig

SAMPLE_RATE_HZ = 100
EPOCH_SECONDS = 30
SAMPLES_PER_EPOCH = SAMPLE_RATE_HZ * EPOCH_SECONDS
SEQUENCE_LENGTH = 15
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://32.194.151.154:5000")
MLFLOW_ARTIFACT_PATH = os.getenv("MLFLOW_ARTIFACT_PATH", "tinysleepnet_final.pth")
XGBOOST_ARTIFACT_PATH = os.getenv(
    "XGBOOST_ARTIFACT_PATH", "best-xgboost-multicanal.joblib"
)

app = FastAPI(title="Sleep Stage Classification API", version="0.3.0")


class PredictionRequest(BaseModel):
    """Quince épocas contiguas de EEG Fpz-Cz expresadas en microvoltios."""

    channel: Literal["EEG Fpz-Cz"] = "EEG Fpz-Cz"
    channel_names: List[str] = Field(default_factory=lambda: ["EEG Fpz-Cz"])
    sampling_rate_hz: Literal[100] = SAMPLE_RATE_HZ
    signal_unit: Literal["uV"] = "uV"
    model_artifact: str = Field(
        default=MLFLOW_ARTIFACT_PATH,
        description=(
            "Nombre o ruta relativa exacta del checkpoint en MLflow. La API usa "
            "el run terminado más reciente que lo contiene."
        ),
    )
    epochs: List[Any] = Field(
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
    def validate_epochs(cls, epochs: List[Any]) -> List[Any]:
        if len(epochs) != SEQUENCE_LENGTH:
            raise ValueError(f"Se requieren exactamente {SEQUENCE_LENGTH} épocas.")
        first_epoch = np.asarray(epochs[0])
        if first_epoch.ndim == 1:
            if len(epochs) != SEQUENCE_LENGTH:
                raise ValueError(f"Se requieren exactamente {SEQUENCE_LENGTH} épocas.")
            if len(epochs[0]) != SAMPLES_PER_EPOCH:
                raise ValueError(f"Cada época debe contener {SAMPLES_PER_EPOCH} muestras.")
            n_channels = 1
        elif first_epoch.ndim == 2:
            n_channels = first_epoch.shape[0]
            if n_channels == 0 or first_epoch.shape[1] != SAMPLES_PER_EPOCH:
                raise ValueError(f"Cada canal debe contener {SAMPLES_PER_EPOCH} muestras.")
        else:
            raise ValueError("epochs debe tener forma (épocas, muestras) o (épocas, canales, muestras).")
        for index, epoch in enumerate(epochs):
            array = np.asarray(epoch)
            if array.shape != first_epoch.shape or not np.isfinite(array).all():
                raise ValueError(f"La época {index} tiene una forma inválida o muestras no finitas.")
        return epochs


class XGBoostPredictionRequest(BaseModel):
    """Épocas crudas multicanal para extraer features y usar XGBoost."""

    channel_names: List[str] = Field(..., min_length=1)
    sampling_rate_hz: Literal[100] = SAMPLE_RATE_HZ
    signal_unit: Literal["uV"] = "uV"
    model_artifact: str = XGBOOST_ARTIFACT_PATH
    epochs: List[List[List[float]]] = Field(
        ...,
        description="Lista de épocas con forma (épocas, canales, 3000 muestras).",
    )

    @field_validator("epochs")
    @classmethod
    def validate_multichannel_epochs(
        cls, epochs: List[List[List[float]]]
    ) -> List[List[List[float]]]:
        if not epochs:
            raise ValueError("Se requiere al menos una época.")
        n_channels = len(epochs[0])
        if n_channels == 0:
            raise ValueError("Se requiere al menos un canal por época.")
        for index, epoch in enumerate(epochs):
            if len(epoch) != n_channels:
                raise ValueError(f"La época {index} tiene un número inconsistente de canales.")
            for channel in epoch:
                if len(channel) != SAMPLES_PER_EPOCH:
                    raise ValueError(
                        f"Cada canal debe contener exactamente {SAMPLES_PER_EPOCH} muestras."
                    )
                if not np.isfinite(channel).all():
                    raise ValueError("Las señales contienen muestras no finitas.")
        return epochs


class TabularPrediction(BaseModel):
    epoch_index: int
    predicted_stage: str
    confidence: float
    probabilities: Dict[str, float]


class XGBoostPredictionResponse(BaseModel):
    model_name: str
    channel_names: List[str]
    sampling_rate_hz: int
    predictions: List[TabularPrediction]


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
def load_model(checkpoint_path: str, input_channels: int = 1) -> TinySleepNet:
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
        input_channels=input_channels,
        n_classes=len(CLASS_NAMES),
        seq_length=SEQUENCE_LENGTH,
        use_rnn=True,
    )
    model = TinySleepNet(config)
    model.load_state_dict(checkpoint, strict=True)
    model.eval()
    return model


@lru_cache(maxsize=4)
def load_xgboost_artifact(artifact_path: str) -> dict:
    """Descarga el joblib de XGBoost y devuelve su modelo y metadatos."""

    path = model_path(artifact_path)
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or not hasattr(artifact.get("model"), "predict"):
        raise TypeError("El artefacto XGBoost debe contener un diccionario con 'model'.")
    if not artifact.get("feature_columns"):
        raise ValueError("El artefacto XGBoost no contiene 'feature_columns'.")
    return artifact


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

    # (épocas, muestras) -> (lote=1, secuencia=15, canal=1, muestras=3000)
    signal_array = np.asarray(request.epochs, dtype=np.float32)
    input_channels = 1 if signal_array.ndim == 2 else signal_array.shape[1]
    if len(request.channel_names) != input_channels:
        raise HTTPException(status_code=422, detail="channel_names no coincide con los canales de epochs.")
    try:
        model = load_model(str(model_path(request.model_artifact)), input_channels)
    except Exception as error:
        raise model_unavailable(error, request.model_artifact) from error
    signal = torch.tensor(signal_array, dtype=torch.float32).unsqueeze(0)
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


@app.post("/predict-xgboost", response_model=XGBoostPredictionResponse)
def predict_xgboost(request: XGBoostPredictionRequest) -> XGBoostPredictionResponse:
    """Extrae features espectrales de señales crudas y clasifica con XGBoost."""

    if len(request.channel_names) != len(request.epochs[0]):
        raise HTTPException(
            status_code=422,
            detail="channel_names debe tener la misma cantidad de canales que epochs.",
        )

    try:
        artifact = load_xgboost_artifact(request.model_artifact)
        raw_epochs = np.asarray(request.epochs, dtype=np.float32)
        n_epochs, n_channels, _ = raw_epochs.shape
        dataset = EpochDataset(
            recording_id="api-request",
            subject_code="api-request",
            night="0",
            X=raw_epochs,
            y=np.full(n_epochs, "W", dtype=str),
            recording_ids=np.full(n_epochs, "api-request", dtype=object),
            subject_codes=np.full(n_epochs, "api-request", dtype=object),
            channel_names=request.channel_names,
            sfreq=request.sampling_rate_hz,
            signal_unit=request.signal_unit,
            dropped_epochs=0,
        )
        feature_dataset = extract_features(dataset)
        feature_frame = pd.DataFrame(
            feature_dataset.X,
            columns=feature_dataset.feature_names,
        )
        channel_flags = {
            "channel_count": n_channels,
            "has_eeg_pz_oz": int(any("pz-oz" in channel.lower() for channel in request.channel_names)),
            "has_eog_horizontal": int(any("eog" in channel.lower() for channel in request.channel_names)),
            "channel_combo_eeg_fpz_cz": int(any("fpz-cz" in channel.lower() for channel in request.channel_names)),
            "channel_combo_eeg_fpz_cz__eeg_pz_oz": int(
                any("fpz-cz" in channel.lower() for channel in request.channel_names)
                and any("pz-oz" in channel.lower() for channel in request.channel_names)
            ),
            "channel_combo_eeg_fpz_cz__eeg_pz_oz__eog_horizontal": int(
                any("fpz-cz" in channel.lower() for channel in request.channel_names)
                and any("pz-oz" in channel.lower() for channel in request.channel_names)
                and any("eog" in channel.lower() for channel in request.channel_names)
            ),
            "channel_combo_eeg_fpz_cz__eog_horizontal": int(
                any("fpz-cz" in channel.lower() for channel in request.channel_names)
                and any("eog" in channel.lower() for channel in request.channel_names)
            ),
        }
        for column, value in channel_flags.items():
            feature_frame[column] = value
        feature_columns = artifact["feature_columns"]
        feature_frame = feature_frame.reindex(columns=feature_columns, fill_value=0.0)
        model = artifact["model"]
        encoded = model.predict(feature_frame)
        probabilities = model.predict_proba(feature_frame)
        label_encoder = artifact.get("label_encoder")
        labels = (
            label_encoder.inverse_transform(encoded.astype(int))
            if label_encoder is not None
            else np.asarray(encoded).astype(str)
        )
        class_labels = artifact.get("class_labels", CLASS_NAMES)
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "xgboost_model_unavailable",
                "message": "No fue posible cargar o ejecutar el modelo XGBoost.",
                "reason": str(error),
                "model_artifact": request.model_artifact,
            },
        ) from error

    predictions = [
        TabularPrediction(
            epoch_index=index,
            predicted_stage=str(labels[index]),
            confidence=float(np.max(probabilities[index])),
            probabilities={
                str(class_name): float(probabilities[index, class_index])
                for class_index, class_name in enumerate(class_labels)
            },
        )
        for index in range(len(encoded))
    ]
    return XGBoostPredictionResponse(
        model_name=request.model_artifact,
        channel_names=request.channel_names,
        sampling_rate_hz=request.sampling_rate_hz,
        predictions=predictions,
    )
