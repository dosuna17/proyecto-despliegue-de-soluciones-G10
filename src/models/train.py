"""
Training loop del modelo avanzado (TinySleepNet) con cross-validation
por sujeto y logging a MLflow.

Requiere la variable de entorno MLFLOW_TRACKING_URI configurada antes
de correr este script, por ejemplo (PowerShell):
    $env:MLFLOW_TRACKING_URI = "http://<EC2_PUBLIC_IP>:5000"
    python src\\models\\train.py

Hiperparámetros de entrenamiento tomados del paper (Sección III.B,
Sleep-EDF): Adam(lr=1e-4, betas=(0.9,0.999)), L2 weight decay=1e-3
(aplicado dentro de TinySleepNet.l2_regularization_loss, ver cnn.py),
gradient clipping=5.0.

Adaptación por restricciones de tiempo/cómputo (decisión propia,
documentar en el informe): el paper entrena 200 épocas por fold; aquí
se usa un número mucho menor con early stopping sobre macro F1 de
validación, deteniendo el entrenamiento si no mejora tras
`patience` épocas -- el propio paper ya usa un criterio similar
("the model that achieved the best overall accuracy was kept").

NOTA: src/evaluation/metrics.py todavía no existe (a la fecha de
escribir esto). Se usa una función de métricas temporal (ver
`compute_metrics` más abajo) con exactamente las llaves acordadas
con Santiago por chat, para que el swap sea directo cuando él la
publique.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import mlflow
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

sys.path.append(str(Path(__file__).resolve().parent.parent))  # para importar src/data, src/models

from data.dataset import (  # noqa: E402
    build_dataloader,
    compute_class_weights,
    get_train_test_subject_codes,
    load_npz_recordings,
    make_subject_kfold_splits,
    DEFAULT_BATCH_SIZE,
    DEFAULT_SEQ_LENGTH,
)
from models.cnn import TinySleepNet, TinySleepNetConfig, build_class_weighted_loss  # noqa: E402

CLASS_NAMES = ["W", "N1", "N2", "N3", "REM"]

# --- Hiperparámetros de entrenamiento (paper, Sección III.B) ---
LEARNING_RATE = 1e-4
ADAM_BETAS = (0.9, 0.999)
GRAD_CLIP_NORM = 5.0

# --- Adaptación propia por restricciones de tiempo (ver docstring) ---
MAX_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5

# --- Adaptación propia por restricciones de MEMORIA (ver comentario de
#     load_npz_recordings en dataset.py): cargar las grabaciones
#     completas de un fold puede exceder la RAM disponible y causar
#     memory thrashing. Se recorta cada grabación a una ventana continua
#     aleatoria de a lo sumo este número de épocas. Ajustar hacia arriba
#     si la máquina tiene más RAM libre disponible. ~500 épocas ~= 4.2
#     horas de sueño -- una porción representativa, no la noche completa.
MAX_EPOCHS_PER_RECORDING = 500

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------
# 1. Métricas (temporal -- ver docstring del módulo)
# ---------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 5) -> Dict:
    """
    Calcula exactamente las llaves acordadas con Santiago por chat:
    accuracy, macro_f1, balanced_accuracy, cohens_kappa,
    confusion_matrix, precision_per_class, recall_per_class, f1_per_class.
    """
    labels = list(range(n_classes))
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "cohens_kappa": cohen_kappa_score(y_true, y_pred, labels=labels),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "precision_per_class": {
            CLASS_NAMES[i]: v for i, v in enumerate(
                precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
            )
        },
        "recall_per_class": {
            CLASS_NAMES[i]: v for i, v in enumerate(
                recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
            )
        },
        "f1_per_class": {
            CLASS_NAMES[i]: v for i, v in enumerate(
                f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
            )
        },
    }


# ---------------------------------------------------------------------
# 2. Un epoch de entrenamiento / evaluación
# ---------------------------------------------------------------------
def run_epoch(
    model: TinySleepNet,
    loader,
    criterion: nn.Module,
    optimizer=None,  # None -> modo evaluación (val/test), no hace backward
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Corre un epoch completo. Si optimizer no es None, entrena
    (backward + step); si es None, solo evalúa (torch.no_grad()).

    Devuelve (loss_promedio, y_true_flat, y_pred_flat).
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, n_batches = 0.0, 0
    all_y_true: List[np.ndarray] = []
    all_y_pred: List[np.ndarray] = []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)  # (batch, seq_len, 1, 3000)
            y_batch = y_batch.to(DEVICE)  # (batch, seq_len)

            logits, _ = model(X_batch)  # (batch, seq_len, n_classes)

            # CrossEntropyLoss espera (N, C, d) con target (N, d) para
            # inputs multi-dimensionales -> permutamos de (batch, seq, C)
            # a (batch, C, seq).
            loss = criterion(logits.permute(0, 2, 1), y_batch)
            loss = loss + model.l2_regularization_loss()

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            preds = torch.argmax(logits, dim=-1)  # (batch, seq_len)
            all_y_true.append(y_batch.detach().cpu().numpy().reshape(-1))
            all_y_pred.append(preds.detach().cpu().numpy().reshape(-1))

    y_true_flat = np.concatenate(all_y_true)
    y_pred_flat = np.concatenate(all_y_pred)
    return total_loss / max(n_batches, 1), y_true_flat, y_pred_flat


# ---------------------------------------------------------------------
# 3. Entrenamiento de un fold, con early stopping
# ---------------------------------------------------------------------
def train_one_fold(
    fold_idx: int,
    train_codes: List[str],
    val_codes: List[str],
    max_epochs: int = MAX_EPOCHS,
    patience: int = EARLY_STOPPING_PATIENCE,
) -> Dict:
    train_recs = load_npz_recordings(subject_codes=train_codes, max_epochs_per_recording=MAX_EPOCHS_PER_RECORDING)
    val_recs = load_npz_recordings(subject_codes=val_codes, max_epochs_per_recording=MAX_EPOCHS_PER_RECORDING)

    class_weights = compute_class_weights(train_recs)  # SOLO desde train del fold

    train_loader = build_dataloader(
        train_recs, seq_length=DEFAULT_SEQ_LENGTH, batch_size=DEFAULT_BATCH_SIZE, shuffle=True
    )
    val_loader = build_dataloader(
        val_recs, seq_length=DEFAULT_SEQ_LENGTH, batch_size=DEFAULT_BATCH_SIZE, shuffle=False
    )

    config = TinySleepNetConfig(use_rnn=True, seq_length=DEFAULT_SEQ_LENGTH, class_weights=class_weights)
    model = TinySleepNet(config).to(DEVICE)
    criterion = build_class_weighted_loss(class_weights, device=DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS)

    best_macro_f1 = -1.0
    best_state = None
    best_val_metrics = None
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        train_loss, y_true_tr, y_pred_tr = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, y_true_val, y_pred_val = run_epoch(model, val_loader, criterion, optimizer=None)

        val_metrics = compute_metrics(y_true_val, y_pred_val)
        train_macro_f1 = f1_score(y_true_tr, y_pred_tr, average="macro", zero_division=0)

        print(
            f"  [fold {fold_idx}] epoch {epoch}/{max_epochs} | "
            f"train_loss={train_loss:.4f} train_macroF1={train_macro_f1:.4f} | "
            f"val_loss={val_loss:.4f} val_macroF1={val_metrics['macro_f1']:.4f}"
        )

        mlflow.log_metrics(
            {
                f"fold{fold_idx}_train_loss": train_loss,
                f"fold{fold_idx}_train_macro_f1": train_macro_f1,
                f"fold{fold_idx}_val_loss": val_loss,
                f"fold{fold_idx}_val_macro_f1": val_metrics["macro_f1"],
            },
            step=epoch,
        )

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_val_metrics = val_metrics
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"  [fold {fold_idx}] early stopping en epoch {epoch} "
                      f"(sin mejora en {patience} epochs)")
                break

    model.load_state_dict(best_state)
    return {"model": model, "metrics": best_val_metrics, "class_weights": class_weights}


# ---------------------------------------------------------------------
# 4. Cross-validation completa + logging a MLflow
# ---------------------------------------------------------------------
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
    folds = make_subject_kfold_splits(train_codes)

    with mlflow.start_run(run_name="tinysleepnet_cv"):
        mlflow.set_tag("model_type", "advanced")
        mlflow.set_tag("signals_used", "EEG Fpz-Cz")
        mlflow.log_params(
            {
                "architecture": "TinySleepNet (CNN1D + LSTM)",
                "seq_length": DEFAULT_SEQ_LENGTH,
                "batch_size": DEFAULT_BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "adam_betas": str(ADAM_BETAS),
                "grad_clip_norm": GRAD_CLIP_NORM,
                "max_epochs": MAX_EPOCHS,
                "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                "n_folds": len(folds),
                "class_weight_smoothing": "sqrt_inverse",
                "data_augmentation": "none (decisión propia, ver informe)",
                "device": DEVICE,
            }
        )

        fold_macro_f1s = []
        for fold_idx, (fold_train_codes, fold_val_codes) in enumerate(folds):
            print(f"\n=== FOLD {fold_idx} ===")
            result = train_one_fold(fold_idx, fold_train_codes, fold_val_codes)
            fold_macro_f1s.append(result["metrics"]["macro_f1"])

        mean_cv_macro_f1 = float(np.mean(fold_macro_f1s))
        std_cv_macro_f1 = float(np.std(fold_macro_f1s))
        mlflow.log_metrics(
            {"cv_mean_macro_f1": mean_cv_macro_f1, "cv_std_macro_f1": std_cv_macro_f1}
        )
        print(f"\n=== CV completa: macro_f1 = {mean_cv_macro_f1:.4f} +/- {std_cv_macro_f1:.4f} ===")
        print("Siguiente paso: entrenar el modelo final sobre TODO el train "
              "(sin fold), y evaluar UNA sola vez sobre el test reservado.")


if __name__ == "__main__":
    main()
