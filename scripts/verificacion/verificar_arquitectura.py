"""Verifica, con el código del repositorio y los datos, cada afirmación del párrafo de arquitectura del artículo.

Uso (desde la raíz del repositorio, con el entorno donde está torch):
    python verificar_arquitectura.py
    python verificar_arquitectura.py --npz <carpeta .npz 3 canales> --emg <carpeta .npz EMG>

No entrena ni modifica nada: construye los modelos, los inspecciona y lee un archivo de datos.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

RAIZ = Path.cwd()
sys.path.insert(0, str(RAIZ / "src"))
from sleep_staging.models.tinysleepnet import TinySleepNet, TinySleepNetConfig  # noqa: E402
from sleep_staging.models.tinysleepnet_emg import TinySleepNetEMG, IDX_CNN  # noqa: E402
from data.dataset import DEFAULT_SEQ_LENGTH, load_npz_recordings  # noqa: E402
from segmentation import CLASS_TO_INT  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--npz", default=str(RAIZ / "data/interim/epochs_por_sujeto_3canales_recortado"))
p.add_argument("--emg", default=str(Path.home() / "Desktop/experimento_emg/datos_emg/epochs_emg_recortado"))
args = p.parse_args()

fallos = 0
def check(afirmacion, observado, ok):
    global fallos
    fallos += not ok
    print(f"[{'OK ' if ok else 'NO '}] {afirmacion}\n       observado: {observado}")

n_par = lambda m: sum(q.numel() for q in m.parameters())
torch.manual_seed(0)
emg = TinySleepNetEMG(TinySleepNetConfig(use_rnn=True, input_channels=8, seq_length=15), rama="mlp").eval()
base = TinySleepNet(TinySleepNetConfig(use_rnn=True, input_channels=1, seq_length=15)).eval()
cnn = emg.cnn
c1 = cnn.block1.conv

print("== Red convolucional por época ==")
check("Primera convolución: 50 muestras, paso 6, 128 filtros",
      f"kernel={c1.kernel_size[0]}, paso={c1.stride[0]}, filtros={c1.out_channels}",
      (c1.kernel_size[0], c1.stride[0], c1.out_channels) == (50, 6, 128))
check("Agrupamiento máximo de 8", f"pool1={cnn.pool1.kernel_size}", cnn.pool1.kernel_size == 8)
check("Abandono de 0.5", f"drop1={cnn.drop1.p}, drop2={cnn.drop2.p}", cnn.drop1.p == 0.5 and cnn.drop2.p == 0.5)
ks = [b.conv.kernel_size[0] for b in cnn.block2]
check("Tres convoluciones de 8 muestras", f"kernels={ks}", ks == [8, 8, 8])
check("Agrupamiento máximo de 4", f"pool2={cnn.pool2.kernel_size}", cnn.pool2.kernel_size == 4)
with torch.no_grad():
    salida = cnn(torch.zeros(1, len(IDX_CNN), 3000))
check("2,048 valores por época (época de 30 s = 3,000 muestras a 100 Hz)",
      f"salida de la CNN {tuple(salida.shape)} -> {salida.numel()} valores; cnn_out_dim={emg.cnn_out_dim}",
      salida.numel() == 2048 == emg.cnn_out_dim)

print("\n== LSTM y salida ==")
check("LSTM de 128 unidades", f"hidden_size={emg.lstm.hidden_size}, capas={emg.lstm.num_layers}, "
      f"bidireccional={emg.lstm.bidirectional}", emg.lstm.hidden_size == 128)
check("Secuencias de 15 épocas consecutivas", f"DEFAULT_SEQ_LENGTH={DEFAULT_SEQ_LENGTH}", DEFAULT_SEQ_LENGTH == 15)
clases = sorted(CLASS_TO_INT, key=CLASS_TO_INT.get)[:5]   # M y "?" existen en las etiquetas pero no se modelan
check("Capa lineal con 5 salidas: W, N1, N2, N3 y REM",
      f"fc={emg.fc.in_features}->{emg.fc.out_features}; índices 0-4 = {clases}",
      emg.fc.out_features == 5 and clases == ["W", "N1", "N2", "N3", "REM"])
with torch.no_grad():
    logits, _ = emg(torch.zeros(1, 15, 8, 3000))
probs = torch.softmax(logits, dim=-1)
check("Salida = probabilidades (softmax sobre la capa lineal)",
      f"logits {tuple(logits.shape)}; suma de softmax por época = {probs.sum(-1).min().item():.4f}..{probs.sum(-1).max().item():.4f}",
      torch.allclose(probs.sum(-1), torch.ones(1, 15)))

print("\n== Señales en microvoltios, sin normalizar ==")
archivos = sorted(Path(args.npz).glob("*_epochs.npz"))
if not archivos:
    print(f"[-- ] No encuentro .npz en {args.npz}; pasa --npz con la carpeta de 3 canales.")
    fallos += 1
else:
    d = np.load(archivos[0], allow_pickle=True)
    check("Los .npz guardan las señales en µV", f"{archivos[0].name}: signal_unit={d['signal_unit']}, "
          f"canales={[str(c) for c in d['channel_names']]}, sfreq={d['sfreq']}", str(d["signal_unit"]) in ("uV", "µV"))
    rec = [r for r in load_npz_recordings(Path(args.npz), subject_codes=[str(d["subject_code"])])
           if r.recording_id == str(d["recording_id"])][0]   # solo ese sujeto, para no cargar las 153 noches
    check("El cargador entrega los mismos valores, sin normalizar ni escalar",
          f"iguales={np.array_equal(rec.X, d['X'])}; media={rec.X.mean():.2f}, desv={rec.X.std():.2f}, "
          f"rango=[{rec.X.min():.1f}, {rec.X.max():.1f}]", np.array_equal(rec.X, d["X"]))

print("\n== Cambio 1: la CNN recibe 6 canales ==")
check("CNN con 6 canales: las 3 señales de 100 Hz y sus 3 marcas",
      f"canales que van a la CNN={IDX_CNN} (0-2 señales, 4-6 marcas); in_channels={c1.in_channels}; "
      f"base de un canal: in_channels={base.cnn.block1.conv.in_channels}",
      c1.in_channels == 6 and tuple(IDX_CNN) == (0, 1, 2, 4, 5, 6) and base.cnn.block1.conv.in_channels == 1)

print("\n== Cambio 2: EMG por una rama aparte ==")
ae = sorted(Path(args.emg).glob("*_epochs.npz"))
if not ae:
    print(f"[-- ] No encuentro .npz de EMG en {args.emg}; pasa --emg con su carpeta.")
    fallos += 1
else:
    e = np.load(ae[0], allow_pickle=True)
    check("El EMG submental está a 1 Hz: 30 valores por época",
          f"{ae[0].name}: canal={[str(c) for c in e['channel_names']]}, sfreq={e['sfreq']}, X={e['X'].shape}, unidad={e['signal_unit']}",
          float(e["sfreq"]) == 1.0 and e["X"].shape[1:] == (1, 30))
check("La CNN no puede usar 30 valores: su primera convolución ya abarca 50 muestras",
      f"kernel de la primera convolución={c1.kernel_size[0]} > 30 valores de EMG por época", c1.kernel_size[0] > 30)
capas = [m for m in emg.emg_branch.red]
lin = [m for m in capas if isinstance(m, torch.nn.Linear)]
check("Perceptrón de dos capas 31 -> 32 -> 16 con ReLU (30 valores + marca)",
      " - ".join(f"Linear({m.in_features},{m.out_features})" if isinstance(m, torch.nn.Linear) else type(m).__name__ for m in capas),
      [(m.in_features, m.out_features) for m in lin] == [(31, 32), (32, 16)]
      and sum(isinstance(m, torch.nn.ReLU) for m in capas) == 2)
check("Los 16 valores se concatenan con los 2,048 antes de la LSTM",
      f"entrada de la LSTM={emg.lstm.input_size} = {emg.cnn_out_dim} + 16", emg.lstm.input_size == emg.cnn_out_dim + 16)

print("\n== Parámetros ==")
pr, pt, pb = n_par(emg.emg_branch), n_par(emg), n_par(base)
check("La rama añade 1,552 parámetros", f"{pr:,}", pr == 1552)
check("Modelo completo 1,558,165 frente a 1,516,421 de TinySleepNet de un canal (2.8 % más)",
      f"EMG-MLP={pt:,}; base de un canal={pb:,}; diferencia={100 * (pt - pb) / pb:.1f} %",
      pt == 1558165 and pb == 1516421 and round(100 * (pt - pb) / pb, 1) == 2.8)

print(f"\n{'Todo comprobado.' if fallos == 0 else f'{fallos} afirmación(es) por revisar.'}")
