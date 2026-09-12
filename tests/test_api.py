import numpy as np
import torch
from fastapi.testclient import TestClient

from app.api.main import app, load_model
from src.models.cnn import TinySleepNet, TinySleepNetConfig


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


def test_predict_reports_a_missing_model(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_PATH", "/does/not/exist.pth")
    load_model.cache_clear()

    response = TestClient(app).post("/predict", json=_payload())

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "model_unavailable"
