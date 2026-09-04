"""Metricas comunes para evaluar modelos multiclase."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


@dataclass(frozen=True)
class ClassificationMetrics:
    """Contenedor con metricas agregadas, por clase y matriz de confusion."""

    summary: dict[str, float]
    per_class: pd.DataFrame
    confusion_matrix: pd.DataFrame


def predictions_to_labels(y_pred: np.ndarray | pd.Series | list[Any]) -> np.ndarray:
    """Convierte predicciones de clase, probabilidades o logits a etiquetas."""

    y_pred_array = np.asarray(y_pred)

    if y_pred_array.ndim == 1:
        return y_pred_array

    if y_pred_array.ndim == 2:
        return np.argmax(y_pred_array, axis=1)

    raise ValueError(
        "y_pred debe ser un vector de etiquetas o una matriz de probabilidades/logits."
    )


def evaluate_classification(
    y_true: np.ndarray | pd.Series | list[Any],
    y_pred: np.ndarray | pd.Series | list[Any],
    class_labels: list[Any] | np.ndarray | None = None,
) -> ClassificationMetrics:
    """Calcula metricas comparables para modelos de clasificacion multiclase.

    La metrica principal del proyecto es `macro_f1`.
    """

    y_true_array = np.asarray(y_true)
    y_pred_labels = predictions_to_labels(y_pred)

    if class_labels is None:
        class_labels = np.unique(np.concatenate([y_true_array, y_pred_labels]))
    class_labels = np.asarray(class_labels)

    accuracy = accuracy_score(y_true_array, y_pred_labels)
    macro_f1 = f1_score(
        y_true_array,
        y_pred_labels,
        labels=class_labels,
        average="macro",
        zero_division=0,
    )
    balanced_accuracy = balanced_accuracy_score(y_true_array, y_pred_labels)
    cohens_kappa = cohen_kappa_score(y_true_array, y_pred_labels, labels=class_labels)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_array,
        y_pred_labels,
        labels=class_labels,
        zero_division=0,
    )

    per_class = pd.DataFrame(
        {
            "class": class_labels,
            "precision_per_class": precision,
            "recall_per_class": recall,
            "f1_per_class": f1,
            "support": support,
        }
    )

    conf_matrix = confusion_matrix(
        y_true_array,
        y_pred_labels,
        labels=class_labels,
    )
    conf_matrix_df = pd.DataFrame(
        conf_matrix,
        index=class_labels,
        columns=class_labels,
    )

    summary = {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "balanced_accuracy": float(balanced_accuracy),
        "cohens_kappa": float(cohens_kappa),
    }

    return ClassificationMetrics(
        summary=summary,
        per_class=per_class,
        confusion_matrix=conf_matrix_df,
    )


def metrics_to_json_dict(metrics: ClassificationMetrics) -> dict[str, Any]:
    """Convierte las metricas a una estructura serializable en JSON."""

    return {
        "summary": metrics.summary,
        "per_class": metrics.per_class.to_dict(orient="records"),
        "confusion_matrix": {
            "labels": metrics.confusion_matrix.index.tolist(),
            "values": metrics.confusion_matrix.values.tolist(),
        },
    }
