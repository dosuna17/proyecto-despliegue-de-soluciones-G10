import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from mlflow.entities import ViewType

import app.api.main as api_main
from app.api.main import (
    app,
    find_run_id_for_artifact,
    load_model,
)
from src.utils.mlflow_runs import resolve_latest_model_artifact
from src.sleep_staging.models.tinysleepnet import TinySleepNet, TinySleepNetConfig

from src.data.segmentation import CLASS_NAMES

def _payload() -> dict:
    epochs = np.zeros((15, 8, 3000), dtype=np.float32)
    epochs[:, 4, :] = 1.0
    epochs[:, 5, :] = 1.0
    return {
        "channel_names": api_main.DL_CHANNEL_NAMES,
        "sampling_rate_hz": 100,
        "signal_unit": "uV",
        "epochs": epochs.tolist(),
    }


def test_predict_returns_a_prediction_for_each_epoch(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "tinysleepnet_final_eeg_eog.pth.pth"
    model = TinySleepNet(TinySleepNetConfig(input_channels=6))
    torch.save(model.state_dict(), checkpoint)
    monkeypatch.setenv("MODEL_PATH", str(checkpoint))
    load_model.cache_clear()

    response = TestClient(app).post("/predict-tinysleepnet", json=_payload())

    assert response.status_code == 200
    body = response.json()
    assert len(body["predictions"]) == 15
    assert set(body["predictions"][0]["probabilities"]) == {"W", "N1", "N2", "N3", "REM"}
    assert abs(sum(body["predictions"][0]["probabilities"].values()) - 1) < 1e-5


def test_predict_rejects_emg_when_checkpoint_does_not_use_it(monkeypatch) -> None:
    class ModeloSinEMG:
        uses_emg = False
        cnn = SimpleNamespace(block1=SimpleNamespace(conv=SimpleNamespace(in_channels=6)))

    payload = _payload()
    epochs = np.asarray(payload["epochs"], dtype=np.float32)
    epochs[:, 3, :] = 1.0
    epochs[:, 7, :] = 1.0
    payload["epochs"] = epochs.tolist()
    monkeypatch.setattr(api_main, "model_path", lambda artifact: "fake-checkpoint.pth")
    monkeypatch.setattr(api_main, "load_model", lambda *args: ModeloSinEMG())

    response = TestClient(app).post("/predict-tinysleepnet", json=payload)

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "unsupported_signal"


def test_predict_rejects_an_invalid_epoch_count() -> None:
    payload = _payload()
    payload["epochs"] = payload["epochs"][:14]

    response = TestClient(app).post("/predict-tinysleepnet", json=payload)

    assert response.status_code == 422


def test_predict_processes_multiple_sequences_in_one_request(monkeypatch) -> None:
    class ModeloSimulado:
        uses_emg = True

        @staticmethod
        def predict_proba(signal):
            batch, sequence, _channels, _samples = signal.shape
            probabilities = torch.full((batch, sequence, len(CLASS_NAMES)), 1 / len(CLASS_NAMES))
            predictions = torch.zeros((batch, sequence), dtype=torch.long)
            confidences = torch.full((batch, sequence), 1 / len(CLASS_NAMES))
            return {
                "probabilities": probabilities,
                "pred_class": predictions,
                "confidence": confidences,
            }, None

    payload = _payload()
    one_sequence = np.asarray(payload["epochs"], dtype=np.float32)
    payload["epochs"] = np.stack([one_sequence, one_sequence]).tolist()
    monkeypatch.setattr(api_main, "model_path", lambda artifact: "fake-checkpoint.pth")
    monkeypatch.setattr(api_main, "load_model", lambda *args: ModeloSimulado())

    response = TestClient(app).post("/predict-tinysleepnet", json=payload)

    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert len(predictions) == 30
    assert [prediction["epoch_index"] for prediction in predictions] == list(range(30))


def test_predict_rejects_an_invalid_model_artifact_path() -> None:
    payload = _payload()
    payload["model_artifact"] = "../other-model.pth"

    response = TestClient(app).post("/predict-tinysleepnet", json=payload)

    assert response.status_code == 422


def test_find_run_id_for_artifact_returns_latest_matching_run() -> None:
    matching_artifact = SimpleNamespace(
        is_dir=False,
        path="tinysleepnet_final_eeg_eog.pth.pth",
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

    result = find_run_id_for_artifact(client, "tinysleepnet_final_eeg_eog.pth.pth")

    assert result == "newest-run"


def test_resolve_latest_model_artifact_uses_run_name_and_artifact() -> None:
    model_artifact = SimpleNamespace(
        is_dir=False,
        path="models/model.pth",
    )
    newest_without_artifact = SimpleNamespace(
        info=SimpleNamespace(run_id="newest-without-artifact", start_time=3),
        data=SimpleNamespace(tags={"mlflow.runName": "model-run"}),
    )
    older_with_artifact = SimpleNamespace(
        info=SimpleNamespace(run_id="older-with-artifact", start_time=2),
        data=SimpleNamespace(tags={"mlflow.runName": "model-run"}),
    )
    search_kwargs = {}

    def search_runs(**kwargs):
        search_kwargs.update(kwargs)
        return [newest_without_artifact, older_with_artifact]

    client = SimpleNamespace(
        get_experiment_by_name=lambda name: SimpleNamespace(experiment_id="7"),
        search_runs=search_runs,
        list_artifacts=lambda run_id, path="": (
            [model_artifact] if run_id == "older-with-artifact" else []
        ),
    )

    result = resolve_latest_model_artifact(
        client,
        {
            "experiment_name": "sleep-staging-advanced",
            "run_name": "model-run",
            "artifact_name": "models/model.pth",
        },
    )

    assert result == {
        "run_id": "older-with-artifact",
        "run_name": "model-run",
        "artifact_path": "models/model.pth",
    }
    assert search_kwargs["run_view_type"] == ViewType.ACTIVE_ONLY


def test_predict_reports_a_missing_model(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_PATH", "/does/not/exist.pth")
    monkeypatch.setattr(api_main, "MODEL_USE_MLFLOW", False)
    load_model.cache_clear()

    response = TestClient(app).post("/predict-tinysleepnet", json=_payload())

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "model_unavailable"

