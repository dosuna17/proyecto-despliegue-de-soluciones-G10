"""API HTTP para inferencia con checkpoints de TinySleepNet registrados en MLflow."""

from __future__ import annotations

import os
import json
import logging
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Literal

import mlflow
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field, field_validator

from src.data.segmentation import CLASS_NAMES
from src.sleep_staging.models.tinysleepnet import TinySleepNet, TinySleepNetConfig
from src.config import MLFLOW_TRACKING_URI
from src.utils.mlflow_runs import load_model_registry, resolve_latest_model_artifact

SAMPLE_RATE_HZ = 100
EPOCH_SECONDS = 30
SAMPLES_PER_EPOCH = SAMPLE_RATE_HZ * EPOCH_SECONDS
SEQUENCE_LENGTH = 15
DL_SIGNAL_CHANNELS = (
    "EEG Fpz-Cz",
    "EEG Pz-Oz",
    "EOG horizontal",
    "EMG submental",
)
DL_PRESENCE_CHANNELS = tuple(f"presence {name}" for name in DL_SIGNAL_CHANNELS)
DL_CHANNEL_NAMES = DL_SIGNAL_CHANNELS + DL_PRESENCE_CHANNELS
DL_INPUT_CHANNELS = len(DL_CHANNEL_NAMES)
MLFLOW_ARTIFACT_PATH = os.getenv(
    "MLFLOW_ARTIFACT_PATH", "modelo_final/flexible_final.pth"
)
MODEL_REGISTRY = load_model_registry()
MODEL_RUN_ID_CHECK_INTERVAL_SECONDS = max(
    60, int(os.getenv("MODEL_RUN_ID_CHECK_INTERVAL_SECONDS", "86400"))
)
logger = logging.getLogger(__name__)
_model_run_check_lock = threading.Lock()
_model_run_check_last_at = 0.0
_model_run_check_status: dict[str, Any] = {
    "status": "not_checked",
    "local_run_id": None,
    "latest_run_id": None,
    "checked_at": None,
    "message": None,
}

app = FastAPI(title="Sleep Stage Classification API", version="0.3.0")


class PredictionRequest(BaseModel):
    """Quince épocas contiguas de EEG Fpz-Cz expresadas en microvoltios."""

    channel_names: List[str] = Field(
        default_factory=lambda: list(DL_CHANNEL_NAMES),
        description="Canales en el orden fijo de TinySleepNet.",
    )
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
            raise ValueError(
                f"TinySleepNet requiere epochs con forma (15, {DL_INPUT_CHANNELS}, 3000)."
            )
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
        if first_epoch.ndim == 2 and first_epoch.shape[0] != DL_INPUT_CHANNELS:
            raise ValueError(f"TinySleepNet requiere exactamente {DL_INPUT_CHANNELS} canales.")
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


def configured_model_reference(client: MlflowClient, artifact_path: str) -> dict[str, str]:
    """Resuelve el run configurado para una ruta de artefacto de modelo."""

    for model_config in MODEL_REGISTRY["models"].values():
        if model_config.get("artifact_name") == artifact_path:
            return resolve_latest_model_artifact(client, model_config)

    raise KeyError(
        f"El artefacto '{artifact_path}' no está configurado en el registro de modelos."
    )


def model_path(artifact_path: str) -> Path:
    """Resuelve el modelo local o descarga el artefacto desde MLflow."""

    local_path = configured_local_model_path()
    if local_path is not None and artifact_path == MLFLOW_ARTIFACT_PATH:
        validate_local_model_run_id(local_path, artifact_path)
        return local_path

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    reference = configured_model_reference(MlflowClient(), artifact_path)
    downloaded_path = mlflow.artifacts.download_artifacts(
        run_id=reference["run_id"],
        artifact_path=reference["artifact_path"],
    )
    return Path(downloaded_path)


def validate_local_model_run_id(local_path: Path, artifact_path: str) -> None:
    """Compara periódicamente el run local registrado con el último run de MLflow.

    La comprobación no bloquea inferencias si MLflow no está disponible y nunca
    reemplaza el checkpoint local. El run_id local se guarda en un JSON hermano
    del checkpoint (por ejemplo, flexible_final.pth.metadata.json).
    """
    global _model_run_check_last_at

    now = time.monotonic()
    if now - _model_run_check_last_at < MODEL_RUN_ID_CHECK_INTERVAL_SECONDS:
        return
    if not _model_run_check_lock.acquire(blocking=False):
        return
    try:
        now = time.monotonic()
        if now - _model_run_check_last_at < MODEL_RUN_ID_CHECK_INTERVAL_SECONDS:
            return

        metadata_path = Path(f"{local_path}.metadata.json")
        local_run_id = None
        if metadata_path.is_file():
            with metadata_path.open(encoding="utf-8") as metadata_file:
                local_run_id = json.load(metadata_file).get("run_id")

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        latest_run_id = configured_model_reference(
            MlflowClient(), artifact_path
        )["run_id"]
        if not local_run_id:
            status = "local_run_id_missing"
            message = f"Falta registrar el run_id local en {metadata_path}."
        elif local_run_id == latest_run_id:
            status = "match"
            message = None
        else:
            status = "mismatch"
            message = "El checkpoint local no corresponde al último run publicado en MLflow."

        _model_run_check_status.update(
            status=status,
            local_run_id=local_run_id,
            latest_run_id=latest_run_id,
            checked_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            message=message,
        )
        if status != "match":
            logger.warning("Verificación de run_id del modelo: %s", message)
    except Exception as error:
        _model_run_check_status.update(
            status="check_failed",
            checked_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            message=str(error),
        )
        logger.warning("No se pudo verificar el run_id del modelo local: %s", error)
    finally:
        _model_run_check_last_at = time.monotonic()
        _model_run_check_lock.release()


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
        "model_run_id_check_interval_seconds": MODEL_RUN_ID_CHECK_INTERVAL_SECONDS,
        "model_run_id_validation": dict(_model_run_check_status),
        "sequence_length": SEQUENCE_LENGTH,
    }


@app.post("/predict-tinysleepnet", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Clasifica las 15 épocas recibidas con el checkpoint seleccionado."""

    # (épocas, señales + máscaras, muestras) -> entrada TinySleepNet de 8 canales.
    if request.channel_names != list(DL_CHANNEL_NAMES):
        raise HTTPException(
            status_code=422,
            detail={"message": "channel_names debe respetar el orden fijo de TinySleepNet.",
                    "expected": list(DL_CHANNEL_NAMES)},
        )
    signal_array = np.asarray(request.epochs, dtype=np.float32)
    input_channels = signal_array.shape[1]
    if len(request.channel_names) != input_channels:
        raise HTTPException(status_code=422, detail="channel_names no coincide con los canales de epochs.")
    n_signal_channels = len(DL_SIGNAL_CHANNELS)
    presence = signal_array[:, n_signal_channels:, :]
    if not np.all(np.isin(presence, (0.0, 1.0))):
        raise HTTPException(status_code=422, detail="Las marcas de presencia deben ser 0 o 1.")
    if not np.all(presence == presence[:, :, :1]):
        raise HTTPException(
            status_code=422,
            detail="Cada marca de presencia debe mantenerse constante en la secuencia.",
        )
    for signal_index in range(n_signal_channels):
        missing = presence[:, signal_index, 0] == 0.0
        if np.any(missing) and not np.all(signal_array[missing, signal_index, :] == 0.0):
            raise HTTPException(
                status_code=422,
                detail="Una señal marcada como ausente debe enviarse completamente en ceros.",
            )
    if not np.any(presence[:, :2, 0] == 1.0):
        raise HTTPException(status_code=422, detail="Debe llegar al menos un EEG.")
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
        channel="TinySleepNet-8-canales",
        sampling_rate_hz=request.sampling_rate_hz,
        predictions=predictions,
    )

