"""Descarga TinySleepNet desde MLflow y registra el run_id junto al checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

# Permite ejecutar el script directamente con
# `python scripts/descargar_modelo_local.py` desde la raíz del repositorio.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlflow
from mlflow.tracking import MlflowClient

from src.utils.mlflow_runs import load_model_registry, resolve_latest_model_artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--salida",
        type=Path,
        default=Path(os.getenv("MODEL_PATH", "modelos/flexible_final.pth")),
        help="Ruta local del checkpoint (por defecto MODEL_PATH o modelos/flexible_final.pth).",
    )
    parser.add_argument(
        "--artefacto",
        default=os.getenv("MLFLOW_ARTIFACT_PATH", "modelo_final/flexible_final.pth"),
    )
    args = parser.parse_args()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise SystemExit("Define MLFLOW_TRACKING_URI y las credenciales de MLflow en el entorno.")

    registry = load_model_registry()
    model_config = next(
        (
            config
            for config in registry["models"].values()
            if config.get("artifact_name") == args.artefacto
        ),
        None,
    )
    if model_config is None:
        raise SystemExit(f"El artefacto '{args.artefacto}' no está en el registro de modelos.")

    mlflow.set_tracking_uri(tracking_uri)
    reference = resolve_latest_model_artifact(MlflowClient(), model_config)
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = Path(f"{args.salida}.metadata.json")

    with tempfile.TemporaryDirectory(dir=args.salida.parent) as temp_dir:
        downloaded = Path(
            mlflow.artifacts.download_artifacts(
                run_id=reference["run_id"], artifact_path=reference["artifact_path"]
            )
        )
        staged_model = Path(temp_dir) / args.salida.name
        shutil.copy2(downloaded, staged_model)
        staged_metadata = Path(temp_dir) / metadata_path.name
        staged_metadata.write_text(
            json.dumps(
                {
                    "run_id": reference["run_id"],
                    "artifact_path": reference["artifact_path"],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        staged_model.replace(args.salida)
        staged_metadata.replace(metadata_path)

    print(f"Checkpoint: {args.salida}")
    print(f"Run ID: {reference['run_id']}")
    print(f"Metadatos: {metadata_path}")


if __name__ == "__main__":
    main()
