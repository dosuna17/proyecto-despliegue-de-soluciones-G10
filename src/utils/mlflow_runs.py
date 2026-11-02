"""Resolución de artefactos de modelos publicados en MLflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_REGISTRY_PATH = PROJECT_ROOT / "configs" / "models" / "model_registry.yaml"


def load_model_registry(
    registry_path: str | Path = DEFAULT_MODEL_REGISTRY_PATH,
) -> dict[str, Any]:
    """Carga la configuración de modelos y valida su estructura básica."""

    path = Path(registry_path)
    with path.open(encoding="utf-8") as registry_file:
        registry = yaml.safe_load(registry_file) or {}

    models = registry.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError(f"El registro de modelos no contiene 'models': {path}")
    return registry


def list_artifacts_recursive(
    client: MlflowClient,
    run_id: str,
    path: str = "",
) -> list[str]:
    """Devuelve las rutas de todos los artefactos de un run, incluidas subcarpetas."""

    artifact_paths: list[str] = []
    for item in client.list_artifacts(run_id, path):
        if item.is_dir:
            artifact_paths.extend(list_artifacts_recursive(client, run_id, item.path))
        else:
            artifact_paths.append(item.path)
    return artifact_paths


def resolve_latest_model_artifact(
    client: MlflowClient,
    model_config: Mapping[str, str],
) -> dict[str, str]:
    """Resuelve el último run FINISHED con el artefacto configurado.

    La búsqueda queda limitada al experiment y al nombre de run configurados.
    El resultado contiene el run_id y la ruta exacta del artefacto.
    """

    experiment_name = model_config["experiment_name"]
    run_name = model_config["run_name"]
    artifact_name = model_config["artifact_name"]

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise FileNotFoundError(
            f"No existe el experiment de MLflow '{experiment_name}'."
        )

    # Filtrar por nombre en MLflow evita descargar y recorrer todos los runs del
    # experimento. La búsqueda anterior podía hacer cientos de llamadas a
    # list_artifacts y dejaba la petición de inferencia esperando varios minutos.
    escaped_run_name = run_name.replace("'", "\\'")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=(
            "attributes.status = 'FINISHED' "
            f"and tags.mlflow.runName = '{escaped_run_name}'"
        ),
        run_view_type=ViewType.ALL,
        order_by=["attributes.start_time DESC"],
        max_results=100,
    )
    runs = sorted(
        runs,
        key=lambda run: run.info.start_time or 0,
        reverse=True,
    )

    for run in runs:
        # El artefacto se conoce por ruta exacta. Listar solo su carpeta evita
        # la exploración recursiva de todo el árbol de artefactos del run.
        parent = Path(artifact_name).parent.as_posix()
        parent = "" if parent == "." else parent
        run_id = run.info.run_id
        artifact_paths = client.list_artifacts(run_id, parent)
        if any(not item.is_dir and item.path == artifact_name for item in artifact_paths):
            return {
                "run_id": run_id,
                "artifact_path": artifact_name,
            }

    raise FileNotFoundError(
        "No se encontró un run FINISHED con el artefacto "
        f"'{artifact_name}' para el run '{run_name}' en '{experiment_name}'."
    )


def resolve_latest_model_artifact_by_name(
    client: MlflowClient,
    model_name: str,
    registry: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Resuelve un modelo por su clave del registro de configuración."""

    registry = registry or load_model_registry()
    models = registry["models"]
    if model_name not in models:
        raise KeyError(f"Modelo no configurado: '{model_name}'.")
    return resolve_latest_model_artifact(client, models[model_name])
