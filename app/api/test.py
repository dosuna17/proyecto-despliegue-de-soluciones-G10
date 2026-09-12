import os
import mlflow

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://32.194.151.154:5000",
)

MLFLOW_RUN_ID = os.getenv(
    "MLFLOW_RUN_ID",
    "d3b246605ec84428b5422a76425789ff",
)

MLFLOW_ARTIFACT_PATH = os.getenv(
    "MLFLOW_ARTIFACT_PATH",
    "tinysleepnet_final.pth.zip",
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

print("MLflow:", MLFLOW_TRACKING_URI)
print("Run ID:", MLFLOW_RUN_ID)
print("Artifact:", MLFLOW_ARTIFACT_PATH)

path = mlflow.artifacts.download_artifacts(
    run_id=MLFLOW_RUN_ID,
    artifact_path=MLFLOW_ARTIFACT_PATH,
)

print("Artifact descargado:")
print(path)