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
    model_config: Mapping[str, Any],
) -> dict[str, str]:
    """Resuelve el último run FINISHED que cumple los criterios configurados.

    La búsqueda queda limitada al experiment, nombre de run y tags requeridos.
    El resultado contiene el run_id y la ruta exacta del artefacto.
    """

    experiment_name = model_config["experiment_name"]
    run_names = model_config.get("run_names") or (
        [model_config["run_name"]] if model_config.get("run_name") else [None]
    )
    artifact_name = model_config["artifact_name"]

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise FileNotFoundError(
            f"No existe el experiment de MLflow '{experiment_name}'."
        )

    # Filtrar en MLflow evita recorrer todos los runs y artefactos del experimento.
    runs_by_id = {}
    for run_name in run_names:
        filters = ["attributes.status = 'FINISHED'"]
        if run_name is not None:
            escaped_run_name = run_name.replace("'", "\\'")
            filters.append(f"tags.mlflow.runName = '{escaped_run_name}'")
        for tag_name, tag_value in model_config.get("required_tags", {}).items():
            escaped_tag_value = str(tag_value).replace("'", "\\'")
            filters.append(f"tags.{tag_name} = '{escaped_tag_value}'")
        for run in client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=" and ".join(filters),
            run_view_type=ViewType.ACTIVE_ONLY,
            order_by=["attributes.start_time DESC"],
            max_results=100,
        ):
            runs_by_id[run.info.run_id] = run
    runs = sorted(
        runs_by_id.values(),
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
                "run_name": run.data.tags.get("mlflow.runName", ""),
                "artifact_path": artifact_name,
            }

    raise FileNotFoundError(
        "No se encontró un run FINISHED con el artefacto "
        f"'{artifact_name}' para los runs {run_names} en '{experiment_name}'."
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
