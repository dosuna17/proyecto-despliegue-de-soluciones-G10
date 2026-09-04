"""Entrena el mejor XGBoost y registra metricas en MLflow."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mlflow
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.data.segmentation import CLASS_NAMES, INT_TO_CLASS
from src.evaluation.metrics import evaluate_classification


MLFLOW_TRACKING_URI = "http://32.194.151.154:5000"
MLFLOW_EXPERIMENT_NAME = "baseline"

TARGET_COLUMN = "y"
ID_COLUMNS = ["recording_id", "subject_code"]

BEST_PARAMS = {
    "subsample": 0.9,
    "reg_lambda": 1.0,
    "reg_alpha": 0.1,
    "n_estimators": 300,
    "min_child_weight": 1,
    "max_depth": 10,
    "learning_rate": 0.08,
    "gamma": 1.0,
    "colsample_bytree": 0.7,
}

FIXED_PARAMS = {
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
}


def main() -> None:
    data_dir = PROJECT_ROOT / "data" / "processed"
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    feature_columns = [
        column
        for column in train_df.columns
        if column not in [TARGET_COLUMN, *ID_COLUMNS]
    ]

    X_train = train_df[feature_columns]
    X_test = test_df[feature_columns]
    y_train_raw = train_df[TARGET_COLUMN]
    y_test_raw = test_df[TARGET_COLUMN]

    y_train_names = y_train_raw.map(INT_TO_CLASS).fillna(y_train_raw).astype(str)
    y_test_names = y_test_raw.map(INT_TO_CLASS).fillna(y_test_raw).astype(str)

    label_encoder = LabelEncoder()
    label_encoder.fit(CLASS_NAMES)
    y_train = label_encoder.transform(y_train_names)
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    fixed_params = {
        **FIXED_PARAMS,
        "num_class": len(CLASS_NAMES),
    }

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment = mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    print(f"Tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"Experiment: {experiment.name}")
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    print(f"Classes: {CLASS_NAMES}")

    with mlflow.mlflow.start_run(experiment_id=experiment.experiment_id, run_name="xgboost_baseline"):
        mlflow.set_tag("model_type", "baseline")
        mlflow.set_tag("signals_used", "EEG Fpz-Cz")
        mlflow.set_tag("stage", "final_model")
        model = XGBClassifier(**fixed_params, **BEST_PARAMS)
        model.fit(X_train, y_train, sample_weight=sample_weights)

        y_pred_encoded = model.predict(X_test)
        y_pred = label_encoder.inverse_transform(y_pred_encoded)
        metrics = evaluate_classification(
            y_true=y_test_names,
            y_pred=y_pred,
            class_labels=CLASS_NAMES,
        )

        mlflow.log_params(BEST_PARAMS)
        mlflow.log_param("num_class", len(CLASS_NAMES))
        mlflow.log_param("n_features", len(feature_columns))
        mlflow.log_param("excluded_id_columns", ",".join(ID_COLUMNS))

        for metric_name, metric_value in metrics.summary.items():
            mlflow.log_metric(metric_name, metric_value)

        mlflow.xgboost.log_model(model, name="best-xgboost-model")

        print("Metricas:")
        for metric_name, metric_value in metrics.summary.items():
            print(f"{metric_name}: {metric_value:.4f}")

        print("\nMetricas por clase:")
        print(metrics.per_class.to_string(index=False))

        print("\nMatriz de confusion:")
        print(metrics.confusion_matrix.to_string())


if __name__ == "__main__":
    main()
