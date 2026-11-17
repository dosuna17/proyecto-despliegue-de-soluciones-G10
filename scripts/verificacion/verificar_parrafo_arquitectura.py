"""Verifica el párrafo «Arquitectura» del artículo contra el código y contra los pesos del modelo seleccionado.

Pone lado a lado lo que dice el texto, lo que construye el código del repositorio y lo que tienen los pesos
guardados del EMG-MLP (modelo_final/flexible_emg_mlp_final.pth del run 95951e2a2cfd4368970636cc6bdea9b5).
Los pesos guardan las formas de las capas (filtros, tamaños de kernel, unidades, capas de la LSTM, salidas);
el paso, el agrupamiento y el abandono no son pesos, así que esos se leen solo del código.
No entrena ni evalúa nada.

Uso, desde la raíz del repositorio:
    python scripts/verificacion/verificar_parrafo_arquitectura.py
    (baja los pesos de MLflow; pide el URI, el usuario y la clave si no están en el entorno)
    --pth RUTA usa un flexible_emg_mlp_final.pth ya bajado.
"""
import argparse
import getpass
import hashlib
import os
import sys
import tempfile
from pathlib import Path

import torch

RAIZ = Path.cwd()
sys.path.insert(0, str(RAIZ / "src"))
from sleep_staging.models.tinysleepnet import TinySleepNetConfig  # noqa: E402
from sleep_staging.models.tinysleepnet_emg import TinySleepNetEMG, IDX_CNN  # noqa: E402
from data.dataset import DEFAULT_SEQ_LENGTH  # noqa: E402
from segmentation import CLASS_TO_INT  # noqa: E402

RUN = "95951e2a2cfd4368970636cc6bdea9b5"
MD5 = "a3fb82ce27e33159d30b38ed3df3850c"

ap = argparse.ArgumentParser()
ap.add_argument("--pth", default=None)
args = ap.parse_args()
if args.pth:
    pth = Path(args.pth)
else:
    import mlflow
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        os.environ["MLFLOW_TRACKING_URI"] = input("MLFLOW_TRACKING_URI: ").strip()
        u = input("MLFLOW_TRACKING_USERNAME (vacío si no hace falta): ").strip()
        if u:
            os.environ["MLFLOW_TRACKING_USERNAME"] = u
            os.environ["MLFLOW_TRACKING_PASSWORD"] = getpass.getpass("MLFLOW_TRACKING_PASSWORD: ")
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    d = Path(mlflow.artifacts.download_artifacts(run_id=RUN, artifact_path="modelo_final", dst_path=tempfile.mkdtemp(prefix="arq_")))
    pth = next(d.rglob("flexible_emg_mlp_final.pth"))

md5 = hashlib.md5(pth.read_bytes()).hexdigest()
W = torch.load(pth, map_location="cpu")
modelo = TinySleepNetEMG(TinySleepNetConfig(use_rnn=True, input_channels=8, seq_length=15), rama="mlp")
modelo.load_state_dict(W, strict=True)          # falla si una sola forma del código no coincide con los pesos
modelo.eval()
cnn, lstm = modelo.cnn, modelo.lstm
c1 = cnn.block1.conv

print("=" * 100)
print(f"Pesos: {pth.name}  md5 {md5}  ({'igual' if md5 == MD5 else 'DISTINTO de'} al registrado en el lock del test)")
print("load_state_dict(strict=True): OK, el código y los pesos tienen exactamente las mismas capas y formas")
print("=" * 100)
print("\nArquitectura tal como la construye el código (print del modelo):\n")
print(modelo)
print("\nCapas guardadas en los pesos (nombre y forma):\n")
for k, v in W.items():
    if v.dim() > 0:
        print(f"   {k:<40} {tuple(v.shape)}")

with torch.no_grad():
    sal_cnn = cnn(torch.zeros(1, len(IDX_CNN), 3000))
    res, _ = modelo.predict_proba(torch.randn(1, DEFAULT_SEQ_LENGTH, 8, 3000))
suma = res["probabilities"].sum(-1)
capas_lstm = sorted({k.split("_l")[-1].split("_")[0] for k in W if k.startswith("lstm.weight")})
inversa = [k for k in W if k.startswith("lstm.") and "reverse" in k]
clases = sorted(CLASS_TO_INT, key=CLASS_TO_INT.get)[:5]
k2 = [tuple(W[f"cnn.block2.{i}.conv.weight"].shape) for i in range(3)]

# (afirmación del texto, valor afirmado, lo que dice el código, lo que dicen los pesos, ¿coincide?)
filas = [
    ("Época de 30 s", "3,000 muestras a 100 Hz", f"input_size={modelo.config.input_size}, fs={modelo.config.sampling_rate}",
     f"entrada de la CNN: {len(IDX_CNN)} canales × 3000", modelo.config.input_size == 3000 and modelo.config.sampling_rate == 100),
    ("1.ª convolución: tamaño", "50 muestras", f"kernel_size={c1.kernel_size[0]}", f"conv1 {tuple(W['cnn.block1.conv.weight'].shape)} → {W['cnn.block1.conv.weight'].shape[2]}",
     c1.kernel_size[0] == 50 == W["cnn.block1.conv.weight"].shape[2]),
    ("1.ª convolución: paso", "6", f"stride={c1.stride[0]}", "(no es un peso)", c1.stride[0] == 6),
    ("1.ª convolución: filtros", "128", f"out_channels={c1.out_channels}", f"conv1 {tuple(W['cnn.block1.conv.weight'].shape)} → {W['cnn.block1.conv.weight'].shape[0]}",
     c1.out_channels == 128 == W["cnn.block1.conv.weight"].shape[0]),
    ("Agrupamiento máximo", "8", f"pool1={cnn.pool1.kernel_size}", "(no es un peso)", cnn.pool1.kernel_size == 8),
    ("Abandono", "0.5", f"drop1={cnn.drop1.p}", "(no es un peso)", cnn.drop1.p == 0.5),
    ("Tres convoluciones de 8 muestras", "3 × 8", f"{len(cnn.block2)} capas, kernels {[b.conv.kernel_size[0] for b in cnn.block2]}",
     f"{k2}", len(cnn.block2) == 3 and all(s[2] == 8 for s in k2)),
    ("Agrupamiento máximo", "4", f"pool2={cnn.pool2.kernel_size}", "(no es un peso)", cnn.pool2.kernel_size == 4),
    ("Valores por época", "2,048", f"salida de la CNN {tuple(sal_cnn.shape)}", f"entrada de la LSTM {W['lstm.weight_ih_l0'].shape[1]} = 2,048 + 16 del EMG",
     sal_cnn.shape[-1] == 2048 and W["lstm.weight_ih_l0"].shape[1] == 2048 + 16),
    ("LSTM unidireccional", "sí", f"bidirectional={lstm.bidirectional}", f"capas _reverse: {len(inversa)}", not lstm.bidirectional and not inversa),
    ("LSTM de una capa", "1", f"num_layers={lstm.num_layers}", f"capas en los pesos: {capas_lstm}", lstm.num_layers == 1 and capas_lstm == ["0"]),
    ("LSTM: unidades", "128", f"hidden_size={lstm.hidden_size}", f"weight_hh_l0 {tuple(W['lstm.weight_hh_l0'].shape)} → 128 (4 compuertas × 128 = 512)",
     lstm.hidden_size == 128 and tuple(W["lstm.weight_hh_l0"].shape) == (512, 128)),
    ("Secuencias de épocas consecutivas", "15", f"seq_length={modelo.config.seq_length}; DEFAULT_SEQ_LENGTH={DEFAULT_SEQ_LENGTH}", "(no es un peso)",
     modelo.config.seq_length == 15 == DEFAULT_SEQ_LENGTH),
    ("Capa lineal final", "5 salidas", f"fc {modelo.fc.in_features} → {modelo.fc.out_features}", f"fc.weight {tuple(W['fc.weight'].shape)}",
     modelo.fc.out_features == 5 and tuple(W["fc.weight"].shape) == (5, 128)),
    ("Clases de salida", "W, N1, N2, N3, REM", f"índices 0–4 = {', '.join(clases)}", "(no es un peso)", clases == ["W", "N1", "N2", "N3", "REM"]),
    ("Softmax da probabilidades", "suman 1", f"predict_proba: suma por época {suma.min():.4f}–{suma.max():.4f}", "(con los pesos entrenados)",
     torch.allclose(suma, torch.ones_like(suma))),
]

print("\n" + "=" * 100)
print("LADO A LADO: texto del artículo frente a código y pesos")
print("=" * 100)
ancho = (34, 20, 44, 46)
print(f"{'Afirmación':<{ancho[0]}} {'Texto':<{ancho[1]}} {'Código':<{ancho[2]}} {'Pesos guardados':<{ancho[3]}} ¿Coincide?")
print("-" * (sum(ancho) + 14))
fallos = 0
for a, t, c, w, ok in filas:
    fallos += not ok
    print(f"{a:<{ancho[0]}} {t:<{ancho[1]}} {c:<{ancho[2]}} {w:<{ancho[3]}} {'OK' if ok else 'NO'}")

print("\nEn el código pero no en el párrafo (no son errores; decidir si se mencionan):")
print(f"   - Segundo abandono de {cnn.drop2.p} después del agrupamiento de 4.")
print(f"   - Abandono de {modelo.rnn_dropout.p} en la salida de la LSTM, antes de la capa lineal.")
print("   - Cada convolución va seguida de normalización por lotes y ReLU; las convoluciones no tienen término independiente "
      f"(bias={c1.bias is not None}).")
print("   - nn.LSTM de PyTorch no tiene conexiones peephole; el comentario del código dice que la LSTM del repositorio oficial sí las usa.")
print(f"   - La CNN recibe {c1.in_channels} canales (3 señales + 3 marcas) y la LSTM además los 16 valores de la rama EMG.")
print(f"\n{'Todo el párrafo coincide.' if fallos == 0 else f'{fallos} afirmación(es) no coinciden: revisar.'}")
