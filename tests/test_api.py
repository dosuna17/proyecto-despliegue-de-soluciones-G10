from types import SimpleNamespace

import numpy as np
import torch
from fastapi.testclient import TestClient

from app.api.main import app, find_run_id_for_artifact, load_model
from src.models.cnn import TinySleepNet, TinySleepNetConfig

from sklearn.preprocessing import LabelEncoder
from src.data.segmentation import CLASS_NAMES

def _payload() -> dict:
    return {
        "channel": "EEG Fpz-Cz",
        "sampling_rate_hz": 100,
        "signal_unit": "uV",
        "epochs": np.zeros((15, 3000), dtype=float).tolist(),
    }


def test_predict_returns_a_prediction_for_each_epoch(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "tinysleepnet_final.pth"
    model = TinySleepNet(TinySleepNetConfig())
    torch.save(model.state_dict(), checkpoint)
    monkeypatch.setenv("MODEL_PATH", str(checkpoint))
    load_model.cache_clear()

    response = TestClient(app).post("/predict", json=_payload())

    assert response.status_code == 200
    body = response.json()
    assert len(body["predictions"]) == 15
    assert set(body["predictions"][0]["probabilities"]) == {"W", "N1", "N2", "N3", "REM"}
    assert abs(sum(body["predictions"][0]["probabilities"].values()) - 1) < 1e-5


def test_predict_rejects_an_invalid_epoch_count() -> None:
    payload = _payload()
    payload["epochs"] = payload["epochs"][:14]

    response = TestClient(app).post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_rejects_an_invalid_model_artifact_path() -> None:
    payload = _payload()
    payload["model_artifact"] = "../other-model.pth"

    response = TestClient(app).post("/predict", json=payload)

    assert response.status_code == 422


def test_find_run_id_for_artifact_returns_latest_matching_run() -> None:
    matching_artifact = SimpleNamespace(
        is_dir=False,
        path="tinysleepnet_final.pth",
    )
    client = SimpleNamespace(
        search_experiments=lambda view_type: [
            SimpleNamespace(
                experiment_id="1",
                name="sleep-staging-advanced",
            )
        ],
        search_runs=lambda **kwargs: [
            SimpleNamespace(info=SimpleNamespace(run_id="newest-run", start_time=2)),
            SimpleNamespace(info=SimpleNamespace(run_id="older-run", start_time=1)),
        ],
        list_artifacts=lambda run_id, path: (
            [matching_artifact] if run_id == "newest-run" else []
        ),
    )

    result = find_run_id_for_artifact(client, "tinysleepnet_final.pth")

    assert result == "newest-run"


def test_predict_reports_a_missing_model(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_PATH", "/does/not/exist.pth")
    load_model.cache_clear()

    response = TestClient(app).post("/predict", json=_payload())

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "model_unavailable"

# Genera un payload de prueba con señal sintetica para el endpoint de predicción con XGBoost. Se puede personalizar el número de canales y la cantidad de épocas.
def _xgboost_payload(channel_names=("EEG Fpz-Cz", "EOG horizontal"), n_epochs=5) -> dict:
    rng = np.random.default_rng(42)
    return {
        "channel_names": list(channel_names),
        "sampling_rate_hz": 100,
        "signal_unit": "uV",
        "model_artifact": "best-xgboost-multicanal.joblib",
        "epochs": rng.normal(0, 10, size=(n_epochs, len(channel_names), 3000)).tolist(),
    }
# Clase de modelo XGBoost falso para pruebas, que devuelve predicciones y probabilidades constantes.
class _FakeXGBModel:

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        n_classes = len(CLASS_NAMES)
        return np.full((len(X), n_classes), 1 / n_classes)

# Función auxiliar para generar un artefacto de modelo XGBoost falso para pruebas, incluyendo un codificador de etiquetas y nombres de clases.
def _fake_xgboost_artifact() -> dict:
    label_encoder = LabelEncoder()
    label_encoder.fit(CLASS_NAMES)
    return {
        "model": _FakeXGBModel(),
        "feature_columns": ["dummy_feature_1", "dummy_feature_2"],
        "label_encoder": label_encoder,
        "class_labels": CLASS_NAMES,
    }


def test_predict_xgboost_returns_a_prediction_for_each_epoch(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.api.main.load_xgboost_artifact",
        lambda artifact_path: _fake_xgboost_artifact(),
    )

    response = TestClient(app).post("/predict-xgboost", json=_xgboost_payload())

    assert response.status_code == 200
    body = response.json()
    assert len(body["predictions"]) == 5
    assert set(body["predictions"][0]["probabilities"]) == set(CLASS_NAMES)
    assert abs(sum(body["predictions"][0]["probabilities"].values()) - 1) < 1e-5


def test_predict_xgboost_rejects_mismatched_channel_names() -> None:
    payload = _xgboost_payload(channel_names=("EEG Fpz-Cz", "EOG horizontal"))
    payload["channel_names"] = ["EEG Fpz-Cz"]  # las épocas siguen teniendo 2 canales

    response = TestClient(app).post("/predict-xgboost", json=payload)

    assert response.status_code == 422


def test_predict_xgboost_rejects_inconsistent_channels_across_epochs() -> None:
    payload = _xgboost_payload(n_epochs=2)
    payload["epochs"][1] = payload["epochs"][1][:1]  # una época pierde un canal

    response = TestClient(app).post("/predict-xgboost", json=payload)

    assert response.status_code == 422

def test_predict_xgboost_reports_a_missing_model(monkeypatch) -> None:
    def _raise_not_found(artifact_path):
        raise FileNotFoundError(f"artefacto no encontrado: {artifact_path}")

    monkeypatch.setattr("app.api.main.load_xgboost_artifact", _raise_not_found)

    response = TestClient(app).post("/predict-xgboost", json=_xgboost_payload())

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "xgboost_model_unavailable"