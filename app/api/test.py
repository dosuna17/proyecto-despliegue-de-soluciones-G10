
#descarga de artefactos
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

import torch

path = r"C:\Users\danos\Documents\tinysleepnet_final.pth"

checkpoint = torch.load(
    r"C:\Users\danos\AppData\Local\Temp\tmpb7m16i3_\tinysleepnet_final.pth.zip",
    map_location="cpu"
)
print(type(checkpoint))
print(checkpoint.keys() if isinstance(checkpoint, dict) else checkpoint)
state_dict = torch.load(
    r"C:\Users\danos\AppData\Local\Temp\tmpb7m16i3_\tinysleepnet_final.pth.zip",
    map_location="cpu"
)

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.models.cnn import TinySleepNet, TinySleepNetConfig
import torch

config = TinySleepNetConfig()

model = TinySleepNet(config)
model.load_state_dict(checkpoint)
model.eval()
print("Modelo cargado correctamente")
print("CNN output dim:", model.cnn_out_dim)
x = torch.randn(1, 15, 1, 3000)

result, hidden = model.predict_proba(x)

print("Clase:", result["pred_class"])
print("Confianza:", result["confidence"])
print("Probabilidades:", result["probabilities"])
