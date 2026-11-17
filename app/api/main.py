"""API HTTP para inferencia con checkpoints de TinySleepNet registrados en MLflow."""

from __future__ import annotations

import os
import json
import logging
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path
from threading import Lock
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
from src.sleep_staging.models.tinysleepnet_emg import TinySleepNetEMG
from src.config import MLFLOW_TRACKING_URI
from src.utils.mlflow_runs import load_model_registry, resolve_latest_model_artifact

SAMPLE_RATE_HZ = 100
EPOCH_SECONDS = 30
SAMPLES_PER_EPOCH = SAMPLE_RATE_HZ * EPOCH_SECONDS
SEQUENCE_LENGTH = 15
MAX_SEQUENCES_PER_REQUEST = 4
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
    "MLFLOW_ARTIFACT_PATH", "modelo_final/flexible_emg_mlp_final.pth"
)
MODEL_REGISTRY = load_model_registry()
MODEL_USE_MLFLOW = os.getenv("MODEL_USE_MLFLOW", "false").strip().lower() in {
    "1", "true", "yes", "on"
}
logger = logging.getLogger(__name__)
_cached_model_run_id: str | None = None
_model_sync_lock = Lock()

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
            "Una secuencia de 15 épocas o un lote de hasta "
            f"{MAX_SEQUENCES_PER_REQUEST} secuencias; cada época tiene 3.000 muestras por canal."
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
        try:
            array = np.asarray(epochs)
        except (TypeError, ValueError) as error:
            raise ValueError("epochs debe ser una secuencia o lote rectangular de épocas.") from error

        if array.ndim == 3:
            forma = f"(15, {DL_INPUT_CHANNELS}, {SAMPLES_PER_EPOCH})"
            if array.shape[0] != SEQUENCE_LENGTH:
                raise ValueError(f"Se requieren exactamente {SEQUENCE_LENGTH} épocas por secuencia.")
        elif array.ndim == 4:
            forma = f"(lote, 15, {DL_INPUT_CHANNELS}, {SAMPLES_PER_EPOCH})"
            if not 1 <= array.shape[0] <= MAX_SEQUENCES_PER_REQUEST:
                raise ValueError(
                    f"El lote debe contener entre 1 y {MAX_SEQUENCES_PER_REQUEST} secuencias."
                )
            if array.shape[1] != SEQUENCE_LENGTH:
                raise ValueError(f"Cada secuencia debe contener exactamente {SEQUENCE_LENGTH} épocas.")
        else:
            raise ValueError(
                "epochs debe tener forma (15, 8, 3000) o "
                f"(lote, 15, 8, 3000); se recibió {array.shape}."
            )

        if array.shape[-2:] != (DL_INPUT_CHANNELS, SAMPLES_PER_EPOCH):
            raise ValueError(f"TinySleepNet requiere epochs con forma {forma}.")
        if not np.isfinite(array).all():
            raise ValueError("epochs contiene muestras no finitas.")
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


def _guardar_metadata_checkpoint(local_path: Path, reference: dict[str, str]) -> None:
    """Guarda la procedencia del checkpoint local mediante reemplazo atomico."""

    metadata_path = Path(f"{local_path}.metadata.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "run_id": reference["run_id"],
        "run_name": reference.get("run_name", ""),
        "artifact_path": reference["artifact_path"],
    }
    temporal = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=metadata_path.parent,
            prefix=f".{metadata_path.name}.", suffix=".tmp", delete=False,
        ) as archivo:
            temporal = Path(archivo.name)
            json.dump(metadata, archivo, ensure_ascii=False, indent=2)
            archivo.write("\n")
        os.chmod(temporal, 0o644)
        os.replace(temporal, metadata_path)
    finally:
        if temporal is not None:
            temporal.unlink(missing_ok=True)


def _actualizar_checkpoint_local(
    downloaded_path: Path, local_path: Path, reference: dict[str, str]
) -> None:
    """Copia el artefacto de MLflow al volumen local sin dejar un archivo parcial."""

    if not downloaded_path.is_file():
        raise FileNotFoundError(f"MLflow no descargo un archivo de checkpoint: {downloaded_path}")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    temporal = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=local_path.parent,
            prefix=f".{local_path.name}.", suffix=".tmp", delete=False,
        ) as archivo:
            temporal = Path(archivo.name)
            with downloaded_path.open("rb") as origen:
                shutil.copyfileobj(origen, archivo)
            archivo.flush()
            os.fsync(archivo.fileno())
        os.chmod(temporal, 0o644)
        os.replace(temporal, local_path)
    finally:
        if temporal is not None:
            temporal.unlink(missing_ok=True)
    _guardar_metadata_checkpoint(local_path, reference)


def _sincronizar_champion_local(artifact_path: str) -> tuple[Path, dict[str, str]]:
    """Serializa la sincronización para evitar descargas duplicadas concurrentes."""

    with _model_sync_lock:
        return _sincronizar_champion_local_locked(artifact_path)


def _sincronizar_champion_local_locked(artifact_path: str) -> tuple[Path, dict[str, str]]:
    """Resuelve y sincroniza el champion de MLflow en MODEL_PATH."""

    global _cached_model_run_id
    local_path = configured_local_model_path()
    if local_path is None:
        raise FileNotFoundError(
            "MODEL_USE_MLFLOW=true requiere MODEL_PATH para guardar el checkpoint descargado."
        )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    reference = configured_model_reference(MlflowClient(), artifact_path)
    metadata_path = Path(f"{local_path}.metadata.json")
    metadata_run_id = None
    metadata_artifact_path = None
    if metadata_path.is_file():
        try:
            with metadata_path.open(encoding="utf-8") as metadata_file:
                metadata = json.load(metadata_file)
                metadata_run_id = metadata.get("run_id")
                metadata_artifact_path = metadata.get("artifact_path")
        except (OSError, json.JSONDecodeError):
            logger.warning(
                "No se pudo leer la metadata de %s; se actualizara el checkpoint.",
                metadata_path,
            )

    if (
        not local_path.is_file()
        or metadata_run_id != reference["run_id"]
        or metadata_artifact_path != reference["artifact_path"]
    ):
        downloaded_path = Path(mlflow.artifacts.download_artifacts(
            run_id=reference["run_id"], artifact_path=reference["artifact_path"]
        ))
        _actualizar_checkpoint_local(downloaded_path, local_path, reference)
        logger.info(
            "Checkpoint local actualizado desde MLflow: run_id=%s, path=%s",
            reference["run_id"], local_path,
        )

    if _cached_model_run_id != reference["run_id"]:
        load_model.cache_clear()
        _cached_model_run_id = reference["run_id"]
    return local_path, reference


def model_path(artifact_path: str) -> Path:
    """Usa el checkpoint local o sincroniza MODEL_PATH con el champion de MLflow."""

    local_path = configured_local_model_path()
    if not MODEL_USE_MLFLOW:
        if local_path is None:
            raise FileNotFoundError(
                "MODEL_USE_MLFLOW=false requiere configurar MODEL_PATH al checkpoint descargado."
            )
        if artifact_path != MLFLOW_ARTIFACT_PATH:
            raise ValueError("El modo local solo admite el artefacto de modelo predeterminado.")
        return local_path

    synced_path, _ = _sincronizar_champion_local(artifact_path)
    return synced_path


@lru_cache(maxsize=8)
def load_model(checkpoint_path: str, input_channels: int = DL_INPUT_CHANNELS) -> torch.nn.Module:
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
    emg_weights = checkpoint.get("emg_branch.red.0.weight")
    if emg_weights is not None:
        rama_emg = "conv" if emg_weights.ndim == 3 else "mlp"
        model = TinySleepNetEMG(config, rama=rama_emg)
        model.uses_emg = True
    else:
        conv_weight = checkpoint.get("cnn.block1.conv.weight")
        if conv_weight is None:
            raise ValueError("El checkpoint no contiene cnn.block1.conv.weight.")
        config.input_channels = int(conv_weight.shape[1])
        model = TinySleepNet(config)
        model.uses_emg = False
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

    run_id = None
    run_name = None
    if MODEL_USE_MLFLOW:
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            reference = configured_model_reference(MlflowClient(), MLFLOW_ARTIFACT_PATH)
            run_id = reference["run_id"]
            run_name = reference.get("run_name") or None
        except Exception as error:
            logger.warning("No se pudo resolver el run activo para /health: %s", error)
    else:
        local_path = configured_local_model_path()
        metadata_path = Path(f"{local_path}.metadata.json") if local_path else None
        if metadata_path and metadata_path.is_file():
            try:
                with metadata_path.open(encoding="utf-8") as metadata_file:
                    metadata = json.load(metadata_file)
                run_id = metadata.get("run_id")
                run_name = metadata.get("run_name")
            except (OSError, json.JSONDecodeError):
                logger.warning("No se pudo leer metadata del checkpoint local: %s", metadata_path)

    return {
        "status": "healthy",
        "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
        "default_model_artifact": MLFLOW_ARTIFACT_PATH,
        "local_model_path": str(configured_local_model_path() or ""),
        "model_use_mlflow": MODEL_USE_MLFLOW,
        "model_run_id": run_id,
        "model_run_name": run_name,
        "sequence_length": SEQUENCE_LENGTH,
    }


@app.post("/predict-tinysleepnet", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Clasifica una o varias secuencias de 15 épocas con el checkpoint seleccionado."""

    # (épocas, señales + máscaras, muestras) -> entrada TinySleepNet de 8 canales.
    if request.channel_names != list(DL_CHANNEL_NAMES):
        raise HTTPException(
            status_code=422,
            detail={"message": "channel_names debe respetar el orden fijo de TinySleepNet.",
                    "expected": list(DL_CHANNEL_NAMES)},
        )
    signal_array = np.asarray(request.epochs, dtype=np.float32)
    if signal_array.ndim == 3:
        signal_array = signal_array[None, ...]
    _batch, _sequence_length, input_channels, _samples = signal_array.shape
    if len(request.channel_names) != input_channels:
        raise HTTPException(status_code=422, detail="channel_names no coincide con los canales de epochs.")
    n_signal_channels = len(DL_SIGNAL_CHANNELS)
    presence = signal_array[:, :, n_signal_channels:, :]
    if not np.all(np.isin(presence, (0.0, 1.0))):
        raise HTTPException(status_code=422, detail="Las marcas de presencia deben ser 0 o 1.")
    if not np.all(presence == presence[:, :, :, :1]):
        raise HTTPException(
            status_code=422,
            detail="Cada marca de presencia debe mantenerse constante en la secuencia.",
        )
    for signal_index in range(n_signal_channels):
        missing = presence[:, :, signal_index, 0] == 0.0
        signal_values = signal_array[:, :, signal_index, :]
        if np.any(missing) and not np.all(signal_values[missing] == 0.0):
            raise HTTPException(
                status_code=422,
                detail="Una señal marcada como ausente debe enviarse completamente en ceros.",
            )
    eeg_present = np.any(presence[:, :, :2, 0] == 1.0, axis=(1, 2))
    if not np.all(eeg_present):
        raise HTTPException(status_code=422, detail="Debe llegar al menos un EEG.")
    try:
        model = load_model(str(model_path(request.model_artifact)), input_channels)
    except Exception as error:
        raise model_unavailable(error, request.model_artifact) from error
    emg_present = np.any(presence[:, :, 3, 0] == 1.0)
    if emg_present and not getattr(model, "uses_emg", False):
        raise HTTPException(
            status_code=422,
            detail={
                "code": "unsupported_signal",
                "message": "El modelo desplegado no usa EMG; envíe el canal EMG como ausente.",
                "channel": "EMG submental",
            },
        )
    signal = torch.tensor(signal_array, dtype=torch.float32)
    if not getattr(model, "uses_emg", False):
        # Checkpoints flexibles anteriores usan EEG/EOG y sus marcas (6 canales).
        n_canales_cnn = model.cnn.block1.conv.in_channels
        if n_canales_cnn == 6 and input_channels == DL_INPUT_CHANNELS:
            signal = signal[:, :, [0, 1, 2, 4, 5, 6], :]
        elif n_canales_cnn != input_channels:
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "model_unavailable",
                    "message": "La cantidad de canales del checkpoint no es compatible con la entrada.",
                    "checkpoint_channels": n_canales_cnn,
                    "input_channels": input_channels,
                },
            )
    with torch.inference_mode():
        result, _ = model.predict_proba(signal)

    probabilities = result["probabilities"].reshape(-1, len(CLASS_NAMES)).cpu().tolist()
    predicted_classes = result["pred_class"].reshape(-1).cpu().tolist()
    confidences = result["confidence"].reshape(-1).cpu().tolist()
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
        channel=("TinySleepNetEMG-8-canales" if getattr(model, "uses_emg", False)
                 else f"TinySleepNet-{model.cnn.block1.conv.in_channels}-canales"),
        sampling_rate_hz=request.sampling_rate_hz,
        predictions=predictions,
    )

