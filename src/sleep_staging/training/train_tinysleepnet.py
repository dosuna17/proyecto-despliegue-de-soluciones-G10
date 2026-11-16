"""
Training loop del modelo avanzado (TinySleepNet) con cross-validation
por sujeto y logging a MLflow.

Requiere la variable de entorno MLFLOW_TRACKING_URI configurada antes
de correr este script, por ejemplo (PowerShell):
    $env:MLFLOW_TRACKING_URI = "http://<EC2_PUBLIC_IP>:5000"
    python src\\models\\train.py

MODO OFFLINE (recomendado dado lo inestable que ha sido la EC2 del
equipo): si NO configuras MLFLOW_TRACKING_URI, el script cae
automáticamente a un tracking LOCAL (sqlite, cero red) -- así puedes
entrenar sin depender de que la EC2 esté prendida. Al terminar, corre
sync_to_mlflow.py (una sola vez, con la EC2 ya prendida) para subir
todo de golpe (params, tags, y el HISTORIAL completo de métricas por
época) al MLflow real del equipo.

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

Métricas: usa src/evaluation/metrics.py real de Santiago
(evaluate_classification()), integrado el 9/4 tras confirmación de
María José en el chat del equipo. compute_metrics() más abajo es un
envoltorio delgado que mapea enteros->nombres de clase y reconstruye
el formato de dict plano que ya usaba el resto del pipeline.
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import mlflow
import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score  # se sigue usando para train_macro_f1 (liviano, cada epoch)
import matplotlib
matplotlib.use("Agg")  # sin ventana gráfica, solo guardar a archivo
import matplotlib.pyplot as plt
import mlflow.pytorch
from sklearn.model_selection import GroupShuffleSplit

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))  # para importar src/data, src/models
sys.path.append(str(Path(__file__).resolve().parents[3]))
from src.config import MLFLOW_TRACKING_URI  # noqa: E402

from data.dataset import (  # noqa: E402
    build_dataloader,
    compute_class_weights,
    get_train_test_subject_codes,
    load_npz_recordings,
    make_subject_kfold_splits,
    DEFAULT_BATCH_SIZE,
    DEFAULT_SEQ_LENGTH,
)
from data.segmentation import INT_TO_CLASS  # noqa: E402 -- mapeo entero->nombre de clase
from evaluation.metrics import evaluate_classification  # noqa: E402 -- metrics.py real de Santiago
from sleep_staging.models.tinysleepnet import TinySleepNet, TinySleepNetConfig, build_class_weighted_loss  # noqa: E402

# Ruta del archivo de hiperparametros (se ubica desde la raiz del repositorio).
CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "models" / "tinysleepnet.yaml"


def load_config(path=CONFIG_PATH):
    """Lee los hiperparametros desde configs/models/tinysleepnet.yaml."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """Fija las semillas de Python, NumPy y PyTorch para que los resultados se repitan."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

CONFIG = load_config()

CLASS_NAMES = ["W", "N1", "N2", "N3", "REM"]

# --- Hiperparámetros de entrenamiento (paper, Sección III.B) ---
LEARNING_RATE = CONFIG["training"]["learning_rate"]
ADAM_BETAS = tuple(CONFIG["training"]["adam_betas"])
GRAD_CLIP_NORM = CONFIG["training"]["grad_clip_norm"]

# --- Adaptación propia por restricciones de tiempo (ver docstring) ---
MAX_EPOCHS = CONFIG["training"]["cv"]["max_epochs"]
EARLY_STOPPING_PATIENCE = CONFIG["training"]["cv"]["early_stopping_patience"]

# --- Adaptación propia por restricciones de MEMORIA (ver comentario de
#     load_npz_recordings en dataset.py): cargar las grabaciones
#     completas de un fold puede exceder la RAM disponible y causar
#     memory thrashing. Se recorta cada grabación a una ventana continua
#     aleatoria de a lo sumo este número de épocas. Ajustar hacia arriba
#     si la máquina tiene más RAM libre disponible. ~500 épocas ~= 4.2
#     horas de sueño -- una porción representativa, no la noche completa.
MAX_EPOCHS_PER_RECORDING = CONFIG["data"]["max_epochs_per_recording"]

# etiqueta legible para el tag signals_used de MLflow segun
# el numero de canales real de los .npz (no asumir siempre EEG-solo).
SIGNALS_USED_BY_N_CHANNELS = {1: "EEG Fpz-Cz", 2: "EEG+EOG", 3: "EEG+EOG+EMG"}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Tracking local (modo offline): usado automáticamente si no hay
# MLFLOW_TRACKING_URI configurada. Ver sync_to_mlflow.py para subir
# esto al MLflow real del equipo cuando la EC2 esté disponible.
LOCAL_TRACKING_DB = "mlflow_local_cv.db"


# ---------------------------------------------------------------------
# 1. Métricas -- usa evaluation/metrics.py real de Santiago (ya
#    integrado). Mapea las etiquetas enteras (0-4, ver segmentation.py)
#    a nombres de clase antes de llamar evaluate_classification(), que
#    espera class_labels como nombres (ver metrics.py de Santiago).
#    Reconstruye el mismo dict plano que usaba la versión temporal, para
#    que ningún otro código de este archivo (ni train_final.py, que
#    importa esta función) necesite cambiar.
# ---------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 5) -> Dict:
    """
    Calcula las mismas llaves de siempre (accuracy, macro_f1,
    balanced_accuracy, cohens_kappa, confusion_matrix,
    precision_per_class, recall_per_class, f1_per_class), ahora
    usando evaluate_classification() real de Santiago por debajo.
    """
    y_true_named = np.array([INT_TO_CLASS[int(i)] for i in y_true])
    y_pred_named = np.array([INT_TO_CLASS[int(i)] for i in y_pred])

    metrics_obj = evaluate_classification(y_true_named, y_pred_named, class_labels=CLASS_NAMES)

    per_class = metrics_obj.per_class.set_index("class")
    return {
        "accuracy": metrics_obj.summary["accuracy"],
        "macro_f1": metrics_obj.summary["macro_f1"],
        "balanced_accuracy": metrics_obj.summary["balanced_accuracy"],
        "cohens_kappa": metrics_obj.summary["cohens_kappa"],
        "confusion_matrix": metrics_obj.confusion_matrix.values.tolist(),
        "precision_per_class": per_class["precision_per_class"].to_dict(),
        "recall_per_class": per_class["recall_per_class"].to_dict(),
        "f1_per_class": per_class["f1_per_class"].to_dict(),
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

    # numero de canales tomado de los datos reales (1=EEG, 2=EEG+EOG, ...),
    # NO asumido fijo, para que este mismo script sirva para cualquier variante multicanal.
    n_input_channels = train_recs[0].X.shape[1]
    assert all(rec.X.shape[1] == n_input_channels for rec in train_recs + val_recs), (
        "Grabaciones con distinto numero de canales dentro del mismo fold -- revisar channel_picks"
    )
    config = TinySleepNetConfig(
        use_rnn=True, seq_length=DEFAULT_SEQ_LENGTH, class_weights=class_weights,
        input_channels=n_input_channels,
    )
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
def run_cv():
    set_seed(CONFIG["seed"])
    tracking_uri = MLFLOW_TRACKING_URI
    if not tracking_uri:
        tracking_uri = f"sqlite:///{LOCAL_TRACKING_DB}"
        print(f"[MODO OFFLINE] MLFLOW_TRACKING_URI no configurada -- "
              f"logueando localmente en {LOCAL_TRACKING_DB}")
        print(f"Cuando termine, corre sync_to_mlflow.py (con la EC2 "
              f"prendida) para subir todo al MLflow del equipo.\n")
    else:
        print(f"[MODO REMOTO] Logueando en vivo a {tracking_uri}\n")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("sleep-staging-advanced")

    train_codes, test_codes = get_train_test_subject_codes()
    folds = make_subject_kfold_splits(train_codes)

    # probe liviano (1 sujeto) solo para saber cuantos canales
    # trae este .npz y taggear correctamente en MLflow.
    _probe_rec = load_npz_recordings(subject_codes=train_codes[:1])[0]
    n_input_channels = _probe_rec.X.shape[1]
    signals_used_tag = SIGNALS_USED_BY_N_CHANNELS.get(n_input_channels, f"{n_input_channels}_canales")
    del _probe_rec

    mlflow.start_run(run_name=f"tinysleepnet_cv_{signals_used_tag}")
    try:
        mlflow.set_tag("model_type", "advanced")
        mlflow.set_tag("signals_used", signals_used_tag)
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
    finally:
        try:
            mlflow.end_run()
        except Exception as e:
            print(f"\n[AVISO] No se pudo cerrar limpiamente el run "
                  f"(no afecta los resultados ya logueados): {e}")




# =====================================================================
# ENTRENAMIENTO FINAL (antes en train_final.py)
# =====================================================================

# --- Presupuesto de entrenamiento del modelo final (una sola corrida,
#     por eso se permite un poco más de margen que en cada fold de CV) ---
# --- Presupuesto de entrenamiento del modelo final (una sola corrida,
#     dejada corriendo sin supervisión ~10h mientras se duerme).
#     Con el ritmo observado hoy en la CV (~2 min/época con
#     MAX_EPOCHS_PER_RECORDING=500), 100 épocas caben cómodamente en
#     el presupuesto de tiempo con margen de sobra. patience=15 evita
#     que corte demasiado pronto si val_macro_f1 sigue siendo ruidosa
#     (como se observó hoy), sin arriesgar correr 10h completas sobre
#     un modelo ya convergido.
MAX_EPOCHS_FINAL = CONFIG["training"]["final"]["max_epochs"]
PATIENCE_FINAL = CONFIG["training"]["final"]["early_stopping_patience"]
VAL_FRACTION = CONFIG["data"]["val_fraction"]  # % de los sujetos de train reservado para early stopping

# MAX_EPOCHS_PER_RECORDING y SIGNALS_USED_BY_N_CHANNELS son los mismos de la validación cruzada (definidos arriba).
# En el entrenamiento final el TEST se carga completo, sin recortar.

# Tracking local (modo offline): usado automáticamente si no hay
# MLFLOW_TRACKING_URI configurada. Nombre distinto al de train.py para
# poder correr ambos scripts sin que se pisen los archivos .db.
LOCAL_TRACKING_DB_FINAL = "mlflow_local_final.db"

# Checkpoint local periódico: guarda el mejor modelo cada N épocas
# (no solo al final), para no perder todo el progreso si algo falla
# tarde en la noche (corte de luz, error de red, etc.)
CHECKPOINT_EVERY_N_EPOCHS = CONFIG["training"]["final"]["checkpoint_every_n_epochs"]


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


def run_final():
    set_seed(CONFIG["seed"])
    tracking_uri = MLFLOW_TRACKING_URI
    if not tracking_uri:
        tracking_uri = f"sqlite:///{LOCAL_TRACKING_DB_FINAL}"
        print(f"[MODO OFFLINE] MLFLOW_TRACKING_URI no configurada -- "
              f"logueando localmente en {LOCAL_TRACKING_DB_FINAL}")
        print(f"Cuando termine, corre sync_to_mlflow.py {LOCAL_TRACKING_DB_FINAL} "
              f"(con la EC2 prendida) para subir todo al MLflow del equipo.\n")
    else:
        print(f"[MODO REMOTO] Logueando en vivo a {tracking_uri}\n")
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

    # numero de canales tomado de los datos reales (1=EEG, 2=EEG+EOG, ...),
    # NO asumido fijo, para que este mismo script sirva para cualquier variante multicanal.
    n_input_channels = fit_recs[0].X.shape[1]
    assert all(rec.X.shape[1] == n_input_channels for rec in fit_recs + val_recs + test_recs), (
        "Grabaciones con distinto numero de canales entre fit/val/test -- revisar channel_picks"
    )
    config = TinySleepNetConfig(
        use_rnn=True, seq_length=DEFAULT_SEQ_LENGTH, class_weights=class_weights,
        input_channels=n_input_channels,
    )
    model = TinySleepNet(config).to(DEVICE)
    criterion = build_class_weighted_loss(class_weights, device=DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS)

    # NOTA (descubierto en la corrida nocturna del 9/3-9/4): usar
    # "with mlflow.start_run(...):" es riesgoso en corridas largas sin
    # supervisión, porque al cerrar el bloque `with`, MLflow intenta
    # AUTOMÁTICAMENTE una llamada de red más (para registrar la URL del
    # run) que NO está bajo nuestro control y no se puede envolver en
    # try/except desde afuera. Si la red falla justo en ese instante
    # (ej. la EC2 se apagó por inactividad a mitad de la noche), el
    # script se cae con un traceback -- aunque todo el trabajo real ya
    # esté guardado localmente para ese punto. Se maneja el run de
    # forma manual para poder envolver también ESE paso final.
    # se calcula ANTES de start_run() para poder usarlo en el
    # nombre del run (ej. "tinysleepnet_final_EEG+EOG") y asi identificar
    # cada variante de canales directamente en la lista de runs de MLflow,
    # sin tener que renombrar a mano despues.
    signals_used_tag = SIGNALS_USED_BY_N_CHANNELS.get(n_input_channels, f"{n_input_channels}_canales")
    # slug seguro para nombres de archivo (sin espacios ni "+"),
    # usado en checkpoint/modelo/metricas para que cada combinacion de
    # canales guarde en su propio archivo y nunca sobreescriba a otra.
    signals_used_slug = signals_used_tag.replace(" ", "_").replace("+", "-")
    mlflow.start_run(run_name=f"tinysleepnet_final_{signals_used_tag}")
    try:
        mlflow.set_tag("model_type", "advanced")
        mlflow.set_tag("signals_used", signals_used_tag)
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

            # Corrida larga sin supervisión: un hipo de red no debe
            # tumbar el script. Si falla el logueo a MLflow, se avisa
            # y se sigue entrenando -- las métricas de esa época se
            # pierden, pero no el entrenamiento completo.
            try:
                mlflow.log_metrics(
                    {
                        "train_loss": train_loss,
                        "train_macro_f1": train_macro_f1,
                        "val_loss": val_loss,
                        "val_macro_f1": val_metrics["macro_f1"],
                    },
                    step=epoch,
                )
            except Exception as e:
                print(f"  [aviso] no se pudo loguear a MLflow en esta época (se continúa): {e}")

            if val_metrics["macro_f1"] > best_val_macro_f1:
                best_val_macro_f1 = val_metrics["macro_f1"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= PATIENCE_FINAL:
                    print(f"Early stopping en epoch {epoch} (sin mejora en {PATIENCE_FINAL} epochs)")
                    break

            # Checkpoint local periódico: por si algo falla tarde en la
            # noche, siempre queda un progreso reciente guardado en disco
            # (no solo el guardado final, que solo ocurre si TODO el
            # script termina sin errores).
            if epoch % CHECKPOINT_EVERY_N_EPOCHS == 0:
                torch.save(best_state, f"checkpoint_intermedio_{signals_used_slug}.pth")
                print(f"  [checkpoint] guardado en checkpoint_intermedio_{signals_used_slug}.pth (mejor hasta ahora: "
                      f"val_macroF1={best_val_macro_f1:.4f})")

        model.load_state_dict(best_state)

        # --- Evaluación FINAL sobre test -- UNA SOLA VEZ ---
        test_loss, y_true_test, y_pred_test = run_epoch(model, test_loader, criterion, optimizer=None)
        test_metrics = compute_metrics(y_true_test, y_pred_test)

        print("\n=== EVALUACIÓN FINAL SOBRE TEST (una sola vez, nunca antes tocado) ===")
        for k in ["accuracy", "macro_f1", "balanced_accuracy", "cohens_kappa"]:
            print(f"  {k}: {test_metrics[k]:.4f}")
        print(f"  matriz de confusión: {test_metrics['confusion_matrix']}")

        # IMPORTANTE: guardado LOCAL primero, siempre, ANTES de cualquier
        # llamada de red a MLflow. Así, sin importar qué falle después
        # (log_metrics, log_artifact, log_model -- cualquier hipo de red
        # en una corrida larga sin supervisión), el modelo y las métricas
        # ya están a salvo en disco.
        cm_path = f"confusion_matrix_test_{signals_used_slug}.png"
        save_confusion_matrix_plot(test_metrics["confusion_matrix"], CLASS_NAMES, cm_path)
        print(f"Matriz de confusión guardada localmente en: {cm_path}")

        model_path = f"tinysleepnet_final_{signals_used_slug}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Modelo guardado localmente en: {model_path}")

        import json
        metrics_path = f"test_metrics_final_{signals_used_slug}.json"
        with open(metrics_path, "w") as f:
            json.dump(test_metrics, f, indent=2)
        print(f"Métricas de test guardadas localmente en: {metrics_path}")

        # Todo lo de red (métricas, artifacts, modelo) va en UN solo
        # bloque try/except -- si algo falla (esperado, ver chat del
        # equipo sobre permisos del bucket S3), no se pierde nada de lo
        # ya guardado arriba, y el script termina limpio de todas formas.
        try:
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

            mlflow.log_artifact(cm_path)
            example_X, _ = next(iter(test_loader))
            input_example = example_X[:1].cpu().numpy()
            mlflow.pytorch.log_model(
                model, artifact_path="model",
                input_example=input_example, serialization_format="pickle",
            )
            print("Métricas y artifacts subidos a MLflow/S3 correctamente.")
        except Exception as e:
            print(f"\n[AVISO] No se pudo completar la subida a MLflow "
                  f"(puede ser el tema de permisos del bucket, o la red): {e}")
            print(f"Los archivos SÍ quedaron guardados localmente:")
            print(f"  - {os.path.abspath(cm_path)}")
            print(f"  - {os.path.abspath(model_path)}")
            print(f"  - {os.path.abspath(metrics_path)}")
            print("Envíaselos a Daniel para que los suba manualmente, o reintenta el logueo a MLflow.")
    finally:
        # Cierre del run, también protegido: si la red falla justo aquí
        # (el bug descubierto anoche), no debe tumbar el script -- todo
        # lo que importa (modelo, métricas, matriz) ya está en disco.
        try:
            mlflow.end_run()
        except Exception as e:
            print(f"\n[AVISO] No se pudo cerrar limpiamente el run en MLflow "
                  f"(no afecta los resultados, ya están guardados localmente): {e}")

    print("\nListo. Revisa el run 'tinysleepnet_final' en MLflow para armar la "
          "tabla comparativa contra el baseline de Santiago.")


if __name__ == "__main__":
    run_cv()
