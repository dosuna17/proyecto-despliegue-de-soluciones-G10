"""Verifica, con el código y el JSON del modelo final, los hiperparámetros de NUESTRO modelo para la Tabla 3 (G2).

Uso, desde la raíz del repositorio:
    python scripts/verificacion/verificar_g2_hiperparametros.py
    (pide el URI, el usuario y la clave de MLflow si no están en el entorno, para leer el JSON del modelo final)

La columna del TinySleepNet original no se verifica aquí: sale del repositorio oficial de los autores
(github.com/akaraspt/tinysleepnet, config/sleepedf.py, model.py, nn.py, train.py, minibatching.py).
"""
import getpass
import inspect
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml

RAIZ = Path.cwd()
sys.path.insert(0, str(RAIZ / "src"))
from data import dataset as ds  # noqa: E402
from sleep_staging.models import tinysleepnet as tsn  # noqa: E402
from sleep_staging.models.tinysleepnet import TinySleepNetConfig  # noqa: E402
from sleep_staging.models.tinysleepnet_emg import TinySleepNetEMG  # noqa: E402
from sleep_staging.training import train_tinysleepnet as base  # noqa: E402
from sleep_staging.training import train_flexible as tf  # noqa: E402

cfg = yaml.safe_load(open(RAIZ / "configs/models/tinysleepnet.yaml", encoding="utf-8"))
m = TinySleepNetEMG(TinySleepNetConfig(use_rnn=True, input_channels=8, seq_length=15), rama="mlp")

def fila(nombre, valor, fuente):
    print(f"  {nombre:34s} {valor}\n  {'':34s} fuente: {fuente}")

print("== Optimización ==")
src_pl = inspect.getsource(tf.entrenar_pliegue) + inspect.getsource(tf.run_final_flexible)
fila("Optimizador", f"Adam, tasa {base.LEARNING_RATE}, betas {base.ADAM_BETAS}, epsilon {torch.optim.Adam([torch.zeros(1)]).defaults['eps']} (valor por defecto de PyTorch)",
     "configs/models/tinysleepnet.yaml + torch.optim.Adam en train_flexible.py")
src_run = inspect.getsource(base.run_epoch)
fila("Recorte del gradiente", f"norma global {base.GRAD_CLIP_NORM} (clip_grad_norm_ en run_epoch: {'clip_grad_norm_' in src_run})",
     "train_tinysleepnet.run_epoch")
fila("Lote y secuencia", f"lote {ds.DEFAULT_BATCH_SIZE} secuencias de {ds.DEFAULT_SEQ_LENGTH} épocas (yaml: {cfg['data']['batch_size']} y {cfg['data']['seq_length']})",
     "src/data/dataset.py y el yaml")
fila("Ventana por grabación en la VC", f"{cfg['data']['max_epochs_per_recording']} épocas continuas por noche; el modelo final usa la noche completa",
     "yaml (data.max_epochs_per_recording) y el JSON del modelo final")

print("\n== Regularización ==")
src_l2 = inspect.getsource(tsn.TinySleepNet.l2_regularization_loss)
convs = [mod for mod in m.cnn.modules() if isinstance(mod, torch.nn.Conv1d)]
fila("L2", f"{m.config.l2_weight_decay} x suma de pesos² de las {len(convs)} convoluciones (sin dividir por 2); "
          f"se suma a la pérdida en run_epoch: {'l2_regularization_loss' in src_run}", "TinySleepNet.l2_regularization_loss")
fila("Abandono", f"CNN {m.cnn.drop1.p} y {m.cnn.drop2.p}; salida de la LSTM {m.rnn_dropout.p}", "modelo instanciado")
bn = m.cnn.block1.bn
fila("BatchNorm", f"eps {bn.eps}, momentum PyTorch {bn.momentum} (= 0.9 en la convención de TensorFlow)", "modelo instanciado")
fila("LSTM", f"{m.lstm.num_layers} capa, {m.lstm.hidden_size} unidades, bidireccional={m.lstm.bidirectional}, "
             f"sin peepholes (nn.LSTM de PyTorch no las tiene)", "modelo instanciado")

print("\n== Pérdida ==")
y = np.array([0] * 30 + [1] * 10 + [2] * 40 + [3] * 8 + [4] * 12)
w = ds.compute_class_weights([ds.SubjectRecording("x", "00", np.zeros((len(y), 1, 1)), y)])
fila("Pesos de clase", f"1/sqrt(frecuencia) calculados con el entrenamiento de cada pliegue "
                       f"(ejemplo con 30/10/40/8/12 %: {[round(v, 2) for v in w]})", "data.dataset.compute_class_weights")

print("\n== Aumento de datos ==")
src_ds = inspect.getsource(ds.SleepSequenceDataset)
usa_azar = any(t in src_ds for t in ("random", "np.roll", "shuffle"))
fila("Aumento de datos", f"ninguno: el corte de secuencias es fijo y determinista (azar en SleepSequenceDataset: {usa_azar})",
     "data.dataset.SleepSequenceDataset")

print("\n== Épocas de entrenamiento ==")
import mlflow  # noqa: E402
if not os.environ.get("MLFLOW_TRACKING_URI"):
    os.environ["MLFLOW_TRACKING_URI"] = input("MLFLOW_TRACKING_URI: ").strip()
    u = input("MLFLOW_TRACKING_USERNAME (vacío si no hace falta): ").strip()
    if u:
        os.environ["MLFLOW_TRACKING_USERNAME"] = u
        os.environ["MLFLOW_TRACKING_PASSWORD"] = getpass.getpass("MLFLOW_TRACKING_PASSWORD: ")
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
fj = mlflow.artifacts.download_artifacts(run_id="95951e2a2cfd4368970636cc6bdea9b5",
                                         artifact_path="modelo_final/flexible_emg_mlp_final.json", dst_path=tempfile.mkdtemp())
info = json.load(open(fj, encoding="utf-8"))
fila("Validación cruzada", "paciencia 5 épocas propias sobre el F1 macro medio de las 12 combinaciones (los 5 pliegues pararon por paciencia)",
     "verificar_g1_entrenamiento.py, parte C")
fila("Modelo final", f"{info['n_epocas']} épocas fijas (media de las mejores de la VC = {info['epocas_cv']}), sin validación ni parada temprana; "
                     f"noche completa={info['noche_completa']}, {info['n_grabaciones']} grabaciones, {info['n_epocas_totales_entrenamiento']:,} épocas de sueño",
     "modelo_final/flexible_emg_mlp_final.json del run campeón")
