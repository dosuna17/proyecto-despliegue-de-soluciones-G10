"""Comprueba, fila por fila, que la columna "Nuestro modelo final (EMG-MLP)" de la tabla de hiperparámetros del artículo es lo que
de verdad usa el código y lo que quedó guardado del modelo campeón.

Uso, desde la raíz del repositorio (entorno con torch y mlflow):
    python scripts/verificacion/verificar_tabla_hiperparametros.py
    (pide el URI, el usuario y la clave de MLflow si no están en el entorno)

No entrena ni modifica nada. Lee el código, el JSON del modelo final y los pliegues del run campeón en MLflow,
y SOLO las etiquetas (y) de los .npz; nunca las señales del test.
"""
import argparse
import getpass
import inspect
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

RAIZ = Path.cwd()
sys.path.insert(0, str(RAIZ / "src"))
from data import dataset as ds  # noqa: E402
from sleep_staging.models import tinysleepnet as tsn  # noqa: E402
from sleep_staging.models import tinysleepnet_emg as temg  # noqa: E402
from sleep_staging.models.tinysleepnet import TinySleepNetConfig  # noqa: E402
from sleep_staging.training import canales_flexibles as cf  # noqa: E402
from sleep_staging.training import train_flexible as tf  # noqa: E402
from sleep_staging.training import train_tinysleepnet as base  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--npz", default=str(RAIZ / "data/interim/epochs_por_sujeto_3canales_recortado"))
p.add_argument("--run-id", default="95951e2a2cfd4368970636cc6bdea9b5")
args = p.parse_args()

fallos = 0
def check(fila, afirmacion, observado, ok):
    global fallos
    fallos += not ok
    print(f"[{'OK ' if ok else 'NO '}] {fila}: {afirmacion}\n       observado: {observado}")

# --- lo guardado del campeón ---
import mlflow  # noqa: E402
if not os.environ.get("MLFLOW_TRACKING_URI"):
    os.environ["MLFLOW_TRACKING_URI"] = input("MLFLOW_TRACKING_URI: ").strip()
    u = input("MLFLOW_TRACKING_USERNAME (vacío si no hace falta): ").strip()
    if u:
        os.environ["MLFLOW_TRACKING_USERNAME"] = u
        os.environ["MLFLOW_TRACKING_PASSWORD"] = getpass.getpass("MLFLOW_TRACKING_PASSWORD: ")
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
tmp = tempfile.mkdtemp(prefix="hiperparametros_")
info = json.load(open(mlflow.artifacts.download_artifacts(run_id=args.run_id,
                      artifact_path="modelo_final/flexible_emg_mlp_final.json", dst_path=tmp), encoding="utf-8"))
dres = Path(mlflow.artifacts.download_artifacts(run_id=args.run_id, artifact_path="resultados", dst_path=tmp))
pliegues = [json.load(open(a, encoding="utf-8")) for a in sorted(dres.glob("flexible_emg_mlp_pliegue[0-9].json"))]

m = temg.TinySleepNetEMG(TinySleepNetConfig(use_rnn=True, input_channels=8, seq_length=15), rama="mlp")
src_pl = inspect.getsource(tf.entrenar_pliegue)
src_fin = inspect.getsource(tf.run_final_flexible)
src_run = inspect.getsource(base.run_epoch)
print()

# 1. Entrada
check("Entrada", "cuatro posiciones con marca de presencia; EMG de 1 Hz por una rama aparte",
      f"posiciones={cf.POSICIONES_EMG}; n_canales_entrada (JSON)={info['n_canales_entrada']}; emg (JSON)={info['emg']}; "
      f"la CNN recibe {temg.IDX_CNN} (sin el EMG); rama={type(m.emg_branch).__name__}",
      len(cf.POSICIONES_EMG) == 4 and info["n_canales_entrada"] == 8 and info["emg"] == "mlp"
      and temg.IDX_EMG not in temg.IDX_CNN and isinstance(m.emg_branch, temg.RamaEMGMLP))

# 2. Optimizador
eps = torch.optim.Adam([torch.zeros(1)], lr=base.LEARNING_RATE, betas=base.ADAM_BETAS).defaults["eps"]
usa_adam = all("torch.optim.Adam(model.parameters(), lr=base.LEARNING_RATE, betas=base.ADAM_BETAS)" in s for s in (src_pl, src_fin))
check("Optimizador", "Adam, tasa 1e-4, β1 0.9, β2 0.999, ε 1e-8",
      f"Adam en VC y en el final con esos argumentos: {usa_adam}; tasa={base.LEARNING_RATE}, betas={base.ADAM_BETAS}, ε={eps} (no se pasa: valor por defecto)",
      usa_adam and base.LEARNING_RATE == 1e-4 and tuple(base.ADAM_BETAS) == (0.9, 0.999) and eps == 1e-8)

# 3. Recorte del gradiente
usa_run_epoch = "base.run_epoch(" in src_pl and "base.run_epoch(" in src_fin
check("Recorte del gradiente", "norma global 5.0",
      f"run_epoch usa clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM): {'clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)' in src_run}; "
      f"GRAD_CLIP_NORM={base.GRAD_CLIP_NORM}; VC y final entrenan con run_epoch: {usa_run_epoch}",
      "clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)" in src_run and base.GRAD_CLIP_NORM == 5.0 and usa_run_epoch)

# 4 y 5. Lote y secuencia
sin_lote_propio = all("batch_size" not in l for s in (src_pl, src_fin) for l in s.splitlines() if "build_flexible_dataloader(" in l)
x = [ds.SubjectRecording("r", "00", np.zeros((60, 4, 3000), dtype=np.float32), np.zeros(60, dtype=np.int64))]
loader = cf.build_flexible_dataloader(x, cf.combinaciones_validas(cf.POSICIONES_EMG), posiciones=cf.POSICIONES_EMG, shuffle=False)
xb, _ = next(iter(loader))
check("Lote", "20 secuencias", f"DEFAULT_BATCH_SIZE={ds.DEFAULT_BATCH_SIZE}; VC y final no cambian el lote: {sin_lote_propio}; "
      f"primer lote de prueba={tuple(xb.shape)}", ds.DEFAULT_BATCH_SIZE == 20 and sin_lote_propio and xb.shape[0] == 20)
check("Secuencia", "15 épocas", f"DEFAULT_SEQ_LENGTH={ds.DEFAULT_SEQ_LENGTH}; seq_length (JSON)={info['seq_length']}; "
      f"épocas por secuencia en el lote={xb.shape[1]}", ds.DEFAULT_SEQ_LENGTH == 15 and info["seq_length"] == 15 and xb.shape[1] == 15)

# 6. L2
convs = [c for c in m.cnn.modules() if isinstance(c, torch.nn.Conv1d)]
with torch.no_grad():
    suma = sum(float((c.weight ** 2).sum()) for c in convs)
    l2 = float(m.l2_regularization_loss())
check("L2", "1e-3 sobre las 4 convoluciones, con la suma completa de pesos² (el doble que la convención de TensorFlow)",
      f"{len(convs)} convoluciones; l2_regularization_loss={l2:.6f}; 1e-3 × Σw²={1e-3 * suma:.6f}; 1e-3 × Σw²/2={1e-3 * suma / 2:.6f}; "
      f"se suma a la pérdida en run_epoch: {'l2_regularization_loss()' in src_run}",
      len(convs) == 4 and abs(l2 - 1e-3 * suma) < 1e-6 * max(1, l2) and "l2_regularization_loss()" in src_run)

# 7. Abandono
fwd = inspect.getsource(temg.TinySleepNetEMG.forward)
check("Abandono", "0.5 tras cada bloque convolucional y 0.5 a la salida de la LSTM",
      f"drop1={m.cnn.drop1.p}, drop2={m.cnn.drop2.p}, salida de la LSTM={m.rnn_dropout.p} (forward aplica rnn_dropout(lstm_out): {'self.rnn_dropout(lstm_out)' in fwd})",
      m.cnn.drop1.p == m.cnn.drop2.p == m.rnn_dropout.p == 0.5 and "self.rnn_dropout(lstm_out)" in fwd)

# 8. LSTM
check("LSTM", "1 capa, 128 unidades, sin peephole",
      f"{type(m.lstm).__name__}: capas={m.lstm.num_layers}, unidades={m.lstm.hidden_size}, bidireccional={m.lstm.bidirectional} "
      f"(torch.nn.LSTM no tiene peephole)", isinstance(m.lstm, torch.nn.LSTM) and m.lstm.num_layers == 1 and m.lstm.hidden_size == 128)

# 9. Pesos de clase (con las etiquetas reales de entrenamiento; las señales no se leen)
train, test = ds.get_train_test_subject_codes()
recs_y, n_test_files = [], 0
for f in sorted(Path(args.npz).glob("*_epochs.npz")):
    d = np.load(f, allow_pickle=True)
    if str(d["subject_code"]) not in train:
        n_test_files += str(d["subject_code"]) in test
        continue
    y = d["y"]
    y = np.array([ds.CLASS_TO_INT[str(v)] for v in y]) if y.dtype.kind in ("U", "S", "O") else y.astype(np.int64)
    recs_y.append(ds.SubjectRecording(str(d["recording_id"]), str(d["subject_code"]), None, y))
w = ds.compute_class_weights(recs_y)
w_json = info.get("class_weights")
iguales = w_json is not None and np.allclose(w, w_json, atol=1e-6)
check("Pesos de clase", "1/√frecuencia de cada clase, con las etiquetas de entrenamiento",
      f"calculados ahora con {len(recs_y)} grabaciones de entrenamiento: {[round(v, 3) for v in w]}; guardados en el JSON: "
      f"{[round(v, 3) for v in w_json] if w_json else None}; iguales={iguales}; épocas={sum(len(r.y) for r in recs_y):,} "
      f"(JSON: {info['n_epocas_totales_entrenamiento']:,})",
      iguales and "compute_class_weights(train_recs)" in src_pl and "compute_class_weights(recs)" in src_fin
      and sum(len(r.y) for r in recs_y) == info["n_epocas_totales_entrenamiento"])

# 10. Aumento de datos
d1 = cf.FlexibleSequenceDataset(x, cf.combinaciones_validas(cf.POSICIONES_EMG), posiciones=cf.POSICIONES_EMG)
x[0].X[:] = np.random.default_rng(0).normal(size=x[0].X.shape).astype(np.float32)
a, b = d1[5][0], d1[5][0]
src_seq = inspect.getsource(ds.SleepSequenceDataset) + inspect.getsource(cf.FlexibleSequenceDataset)
check("Aumento de datos", "ninguno", f"el mismo elemento pedido dos veces es idéntico: {torch.equal(a, b)}; "
      f"sin azar en los datasets: {not any(t in src_seq for t in ('np.random', 'torch.rand', 'np.roll', 'random.'))}",
      torch.equal(a, b) and not any(t in src_seq for t in ("np.random", "torch.rand", "np.roll", "random.")))

# 11. Épocas de entrenamiento
mejores = [p_["mejor_epoca"] for p_ in pliegues]
tras = [len(p_["historial"]) - p_["mejor_epoca"] for p_ in pliegues]
check("Épocas (VC)", "paciencia 5 sobre el F1 medio de las 12 combinaciones",
      f"paciencia del yaml={base.EARLY_STOPPING_PATIENCE}; épocas tras la mejor en los 5 pliegues={tras}; combinaciones validadas={len(pliegues[0]['metricas_por_combinacion'])}",
      base.EARLY_STOPPING_PATIENCE == 5 and all(t == 5 for t in tras) and len(pliegues[0]["metricas_por_combinacion"]) == 12)
sin_val = not any(t in src_fin for t in ("val_", "patience", "paciencia"))
check("Épocas (final)", "12 épocas fijas (media de las mejores de la VC), sin validación",
      f"mejores épocas de la VC={mejores} (media {np.mean(mejores):.1f} -> {round(np.mean(mejores))}); n_epocas (JSON)={info['n_epocas']}; "
      f"el entrenamiento final no valida ni tiene parada temprana: {sin_val}",
      info["n_epocas"] == round(np.mean(mejores)) == 12 and sin_val)

# 12. Datos
todas, max_ini, max_fin = set(), 0, 0
for f in sorted(Path(args.npz).glob("*_epochs.npz")):
    y = np.load(f, allow_pickle=True)["y"]
    y = np.array([str(v) for v in y]) if y.dtype.kind in ("U", "S", "O") else np.array([["W", "N1", "N2", "N3", "REM"][int(v)] for v in y])
    todas |= set(y)
    sueno = np.flatnonzero(y != "W")
    max_ini, max_fin = max(max_ini, sueno[0]), max(max_fin, len(y) - 1 - sueno[-1])
check("Datos", "vigilia recortada a 30 min antes y después del sueño, sin M ni \"?\"",
      f"estadios en los .npz={sorted(str(t) for t in todas)}; vigilia máxima al inicio={max_ini} épocas, al final={max_fin} épocas (60 épocas = 30 min)",
      todas <= {"W", "N1", "N2", "N3", "REM"} and max_ini <= 60 and max_fin <= 60)

# 13. Evaluación
folds = ds.make_subject_kfold_splits(train)
vals = [set(v) for _, v in folds]
disj = all(not (vals[i] & vals[j]) for i in range(len(vals)) for j in range(i + 1, len(vals)))
check("Evaluación", "VC de 5 pliegues sobre 62 personas y prueba aparte con 16",
      f"pliegues={len(folds)}; personas de entrenamiento={len(train)}, de prueba={len(test)}; validaciones disjuntas que cubren a todos: "
      f"{disj and set().union(*vals) == set(train)}; train y prueba sin cruce: {not set(train) & set(test)}",
      len(folds) == 5 and len(train) == 62 and len(test) == 16 and disj and set().union(*vals) == set(train) and not set(train) & set(test))

print(f"\n{'Todas las filas de la tabla de hiperparámetros coinciden con el código y el modelo guardado.' if fallos == 0 else f'{fallos} fila(s) por revisar.'}")
