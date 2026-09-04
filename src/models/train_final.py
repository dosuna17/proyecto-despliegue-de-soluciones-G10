"""
Entrena el modelo FINAL de TinySleepNet: una sola corrida sobre TODOS
los sujetos de train (sin folds de cross-validation), con un pequeño
conjunto de validación interno (tomado de train, NUNCA de test) usado
solo para decidir el mejor checkpoint vía early stopping.

Este protocolo replica el que usa el propio paper para el modelo final
de cada fold (Sección III.B: "In each fold, we further split 10% of
the training set into a validation set for evaluating the training
model. The model that achieved the best overall accuracy was kept for
evaluation with the test set.") -- aquí se aplica una sola vez, sobre
todo train, para producir el modelo final del proyecto.

El conjunto de TEST se evalúa UNA SOLA VEZ, al final, con el mejor
checkpoint -- nunca se usa para ninguna decisión de entrenamiento.

Requiere MLFLOW_TRACKING_URI configurada (igual que train.py), ej.:
    $env:MLFLOW_TRACKING_URI = "http://<IP-EC2>:5000"
    python src\\models\\train_final.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # sin ventana gráfica, solo guardar a archivo
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit

sys.path.append(str(Path(__file__).resolve().parent.parent))

from data.dataset import (  # noqa: E402
    build_dataloader,
    compute_class_weights,
    get_train_test_subject_codes,
    load_npz_recordings,
    DEFAULT_BATCH_SIZE,
    DEFAULT_SEQ_LENGTH,
)
from models.cnn import TinySleepNet, TinySleepNetConfig, build_class_weighted_loss  # noqa: E402
from models.train import (  # noqa: E402 -- reutiliza lo ya construido y probado
    CLASS_NAMES,
    LEARNING_RATE,
    ADAM_BETAS,
    GRAD_CLIP_NORM,
    DEVICE,
    compute_metrics,
    run_epoch,
)

# --- Presupuesto de entrenamiento del modelo final (una sola corrida,
#     por eso se permite un poco más de margen que en cada fold de CV) ---
MAX_EPOCHS_FINAL = 30
PATIENCE_FINAL = 7
VAL_FRACTION = 0.15  # % de los sujetos de train reservado para early stopping

# Mismo límite de memoria que train.py (ver comentario ahí). Se aplica
# SOLO a fit/val (donde está el volumen grande de datos); el TEST se
# carga completo, sin recortar, porque son los números finales que van
# a la tabla comparativa con el baseline -- ahí sí se quiere el
# conjunto completo, no una aproximación.
MAX_EPOCHS_PER_RECORDING = 500


def make_internal_validation_split(train_codes, val_fraction: float = VAL_FRACTION, seed: int = 42):
    """
    Separa una porción de los sujetos de TRAIN (nunca de test) para
    usar como validación interna, a nivel de sujeto (GroupShuffleSplit).
    """
    codes = np.array(sorted(train_codes))
    gss = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    dummy_X = np.zeros(len(codes))
    fit_idx, val_idx = next(gss.split(dummy_X, groups=codes))
    return codes[fit_idx].tolist(), codes[val_idx].tolist()


def save_confusion_matrix_plot(cm: list, class_names: list, out_path: str) -> None:
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusión -- Modelo avanzado (test)")
    thresh = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise RuntimeError(
            "MLFLOW_TRACKING_URI no está configurada. Corre primero:\n"
            '  $env:MLFLOW_TRACKING_URI = "http://<EC2_PUBLIC_IP>:5000"   (PowerShell)'
        )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("sleep-staging-advanced")

    train_codes, test_codes = get_train_test_subject_codes()
    fit_codes, val_codes = make_internal_validation_split(train_codes)

    print(f"Sujetos: fit={len(fit_codes)} | val (early stopping)={len(val_codes)} | "
          f"test (evaluación final, una sola vez)={len(test_codes)}")

    fit_recs = load_npz_recordings(subject_codes=fit_codes, max_epochs_per_recording=MAX_EPOCHS_PER_RECORDING)
    val_recs = load_npz_recordings(subject_codes=val_codes, max_epochs_per_recording=MAX_EPOCHS_PER_RECORDING)
    test_recs = load_npz_recordings(subject_codes=test_codes)  # completo, sin recortar

    class_weights = compute_class_weights(fit_recs)  # SOLO desde fit, nunca val/test

    fit_loader = build_dataloader(fit_recs, seq_length=DEFAULT_SEQ_LENGTH,
                                   batch_size=DEFAULT_BATCH_SIZE, shuffle=True)
    val_loader = build_dataloader(val_recs, seq_length=DEFAULT_SEQ_LENGTH,
                                   batch_size=DEFAULT_BATCH_SIZE, shuffle=False)
    test_loader = build_dataloader(test_recs, seq_length=DEFAULT_SEQ_LENGTH,
                                    batch_size=DEFAULT_BATCH_SIZE, shuffle=False)

    config = TinySleepNetConfig(use_rnn=True, seq_length=DEFAULT_SEQ_LENGTH, class_weights=class_weights)
    model = TinySleepNet(config).to(DEVICE)
    criterion = build_class_weighted_loss(class_weights, device=DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS)

    with mlflow.start_run(run_name="tinysleepnet_final"):
        mlflow.set_tag("model_type", "advanced")
        mlflow.set_tag("signals_used", "EEG Fpz-Cz")
        mlflow.set_tag("stage", "final_model")
        mlflow.log_params(
            {
                "architecture": "TinySleepNet (CNN1D + LSTM)",
                "seq_length": DEFAULT_SEQ_LENGTH,
                "batch_size": DEFAULT_BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "adam_betas": str(ADAM_BETAS),
                "grad_clip_norm": GRAD_CLIP_NORM,
                "max_epochs": MAX_EPOCHS_FINAL,
                "early_stopping_patience": PATIENCE_FINAL,
                "val_fraction_internal": VAL_FRACTION,
                "class_weight_smoothing": "sqrt_inverse",
                "data_augmentation": "none (decisión propia, ver informe)",
                "n_subjects_fit": len(fit_codes),
                "n_subjects_val": len(val_codes),
                "n_subjects_test": len(test_codes),
                "device": DEVICE,
            }
        )

        best_val_macro_f1 = -1.0
        best_state = None
        epochs_without_improvement = 0

        for epoch in range(1, MAX_EPOCHS_FINAL + 1):
            train_loss, y_true_tr, y_pred_tr = run_epoch(model, fit_loader, criterion, optimizer)
            val_loss, y_true_val, y_pred_val = run_epoch(model, val_loader, criterion, optimizer=None)

            val_metrics = compute_metrics(y_true_val, y_pred_val)
            train_macro_f1 = f1_score(y_true_tr, y_pred_tr, average="macro", zero_division=0)

            print(
                f"epoch {epoch}/{MAX_EPOCHS_FINAL} | "
                f"train_loss={train_loss:.4f} train_macroF1={train_macro_f1:.4f} | "
                f"val_loss={val_loss:.4f} val_macroF1={val_metrics['macro_f1']:.4f}"
            )
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_macro_f1": train_macro_f1,
                    "val_loss": val_loss,
                    "val_macro_f1": val_metrics["macro_f1"],
                },
                step=epoch,
            )

            if val_metrics["macro_f1"] > best_val_macro_f1:
                best_val_macro_f1 = val_metrics["macro_f1"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= PATIENCE_FINAL:
                    print(f"Early stopping en epoch {epoch} (sin mejora en {PATIENCE_FINAL} epochs)")
                    break

        model.load_state_dict(best_state)

        # --- Evaluación FINAL sobre test -- UNA SOLA VEZ ---
        test_loss, y_true_test, y_pred_test = run_epoch(model, test_loader, criterion, optimizer=None)
        test_metrics = compute_metrics(y_true_test, y_pred_test)

        print("\n=== EVALUACIÓN FINAL SOBRE TEST (una sola vez, nunca antes tocado) ===")
        for k in ["accuracy", "macro_f1", "balanced_accuracy", "cohens_kappa"]:
            print(f"  {k}: {test_metrics[k]:.4f}")
        print(f"  matriz de confusión: {test_metrics['confusion_matrix']}")

        mlflow.log_metrics(
            {
                "test_accuracy": test_metrics["accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
                "test_balanced_accuracy": test_metrics["balanced_accuracy"],
                "test_cohens_kappa": test_metrics["cohens_kappa"],
            }
        )
        for cls in CLASS_NAMES:
            mlflow.log_metric(f"test_precision_{cls}", test_metrics["precision_per_class"][cls])
            mlflow.log_metric(f"test_recall_{cls}", test_metrics["recall_per_class"][cls])
            mlflow.log_metric(f"test_f1_{cls}", test_metrics["f1_per_class"][cls])

        cm_path = "confusion_matrix_test.png"
        save_confusion_matrix_plot(test_metrics["confusion_matrix"], CLASS_NAMES, cm_path)
        mlflow.log_artifact(cm_path)

        # mlflow.pytorch.log_model exige un input_example en esta versión
        # de MLflow. Se fuerza serialization_format="pickle" (en vez del
        # nuevo formato "pt2" que intenta por defecto) porque el modelo
        # devuelve una tupla (logits, hidden) -- el formato pt2 solo
        # soporta modelos que devuelven un único tensor de salida.
        example_X, _ = next(iter(test_loader))
        input_example = example_X[:1].cpu().numpy()  # (1, seq_length, 1, 3000)
        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            input_example=input_example,
            serialization_format="pickle",
        )

    print("\nListo. Revisa el run 'tinysleepnet_final' en MLflow para armar la "
          "tabla comparativa contra el baseline de Santiago.")


if __name__ == "__main__":
    main()
