"""Verifica que el modelo campeón guardado (flexible_emg_mlp_final.pth) es la arquitectura descrita en el artículo.

Por defecto descarga los archivos del run campeón de MLflow (los mismos que usa la API):
    python scripts/verificacion/verificar_pesos.py
    (pide el URI, el usuario y la clave de MLflow si no están en el entorno)
O con archivos locales (por ejemplo, los de modelos_finales/):
    python scripts/verificacion/verificar_pesos.py --pth <ruta .pth> --json <ruta .json>

No entrena ni modifica nada. La prueba de predicción usa UN sujeto de entrenamiento (nunca de prueba).
"""
import argparse
import getpass
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

RAIZ = Path.cwd()
sys.path.insert(0, str(RAIZ / "src"))
from sleep_staging.models.tinysleepnet import TinySleepNetConfig  # noqa: E402
from sleep_staging.models.tinysleepnet_emg import TinySleepNetEMG  # noqa: E402
from sleep_staging.training.canales_flexibles import POSICIONES_EMG, combinaciones_validas, preparar_entrada  # noqa: E402
from data.dataset import DEFAULT_SEQ_LENGTH, get_train_test_subject_codes  # noqa: E402

RUN_CAMPEON = "95951e2a2cfd4368970636cc6bdea9b5"
p = argparse.ArgumentParser()
p.add_argument("--pth", default=None)
p.add_argument("--json", default=None)
p.add_argument("--run-id", default=RUN_CAMPEON)
p.add_argument("--npz", default=str(RAIZ / "data/interim/epochs_por_sujeto_3canales_recortado"))
p.add_argument("--emg", default=str(Path.home() / "Desktop/experimento_emg/datos_emg/epochs_emg_recortado"))
args = p.parse_args()

fallos = 0
def check(afirmacion, observado, ok):
    global fallos
    fallos += not ok
    print(f"[{'OK ' if ok else 'NO '}] {afirmacion}\n       observado: {observado}")

# --- 1. Archivos ---
if args.pth and args.json:
    ruta_pth, ruta_json, origen = Path(args.pth), Path(args.json), "archivos locales"
else:
    import mlflow
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        os.environ["MLFLOW_TRACKING_URI"] = input("MLFLOW_TRACKING_URI: ").strip()
        u = input("MLFLOW_TRACKING_USERNAME (vacío si no hace falta): ").strip()
        if u:
            os.environ["MLFLOW_TRACKING_USERNAME"] = u
            os.environ["MLFLOW_TRACKING_PASSWORD"] = getpass.getpass("MLFLOW_TRACKING_PASSWORD: ")
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    run = mlflow.MlflowClient().get_run(args.run_id)
    print(f"Run {args.run_id}: {run.data.tags.get('mlflow.runName')} | champion={run.data.tags.get('champion')}")
    destino = Path(tempfile.mkdtemp(prefix="campeon_"))
    carpeta = Path(mlflow.artifacts.download_artifacts(run_id=args.run_id, artifact_path="modelo_final", dst_path=str(destino)))
    ruta_pth, ruta_json = carpeta / "flexible_emg_mlp_final.pth", carpeta / "flexible_emg_mlp_final.json"
    origen = f"MLflow, run {args.run_id}, modelo_final/"
md5 = hashlib.md5(ruta_pth.read_bytes()).hexdigest()
print(f"Pesos: {ruta_pth.name} ({ruta_pth.stat().st_size:,} bytes, md5 {md5}) desde {origen}\n")

# --- 2. JSON del modelo final ---
info = json.load(open(ruta_json, encoding="utf-8"))
print("== JSON del modelo final ==")
check("Es el modelo con la rama EMG tipo MLP", f"run={info.get('run')}, emg={info.get('emg')}, sin_lstm={info.get('sin_lstm')}",
      info.get("run") == "flexible_emg_mlp" and info.get("emg") == "mlp" and not info.get("sin_lstm"))
esperadas = [n for n, _ in combinaciones_validas(POSICIONES_EMG)]
check("Entrenado con las 12 combinaciones", f"{len(info.get('combinaciones', []))} combinaciones; iguales a las del código: "
      f"{info.get('combinaciones') == esperadas}", info.get("combinaciones") == esperadas)
check("8 canales de entrada, secuencias de 15, clases W N1 N2 N3 REM",
      f"n_canales_entrada={info.get('n_canales_entrada')}, seq_length={info.get('seq_length')}, clases={info.get('clases')}",
      info.get("n_canales_entrada") == 8 and info.get("seq_length") == DEFAULT_SEQ_LENGTH == 15
      and info.get("clases") == ["W", "N1", "N2", "N3", "REM"])
print(f"       (otros datos del JSON: épocas de entrenamiento={info.get('n_epocas')}, épocas de la CV={info.get('epocas_cv')}, "
      f"noche completa={info.get('noche_completa')}, grabaciones={info.get('n_grabaciones')}, "
      f"épocas de sueño usadas={info.get('n_epocas_totales_entrenamiento')}, minutos={info.get('segundos_totales', 0) / 60:.0f})")

# --- 3. Pesos frente a la arquitectura ---
print("\n== Pesos frente a la arquitectura descrita ==")
sd = torch.load(ruta_pth, map_location="cpu")
modelo = TinySleepNetEMG(TinySleepNetConfig(use_rnn=True, input_channels=8, seq_length=15), rama="mlp")
faltan, sobran = modelo.load_state_dict(sd, strict=False)
check("Los pesos encajan exactamente en TinySleepNetEMG (ni faltan ni sobran)", f"faltan={list(faltan)}, sobran={list(sobran)}",
      not faltan and not sobran)
modelo.load_state_dict(sd, strict=True)
modelo.eval()
forma = lambda k: tuple(sd[k].shape)
check("Primera convolución: 128 filtros, 6 canales, 50 muestras", f"cnn.block1.conv.weight {forma('cnn.block1.conv.weight')}",
      forma("cnn.block1.conv.weight") == (128, 6, 50))
check("Rama EMG: Linear 31 -> 32 -> 16", f"emg_branch.red.0.weight {forma('emg_branch.red.0.weight')}, "
      f"emg_branch.red.2.weight {forma('emg_branch.red.2.weight')}",
      forma("emg_branch.red.0.weight") == (32, 31) and forma("emg_branch.red.2.weight") == (16, 32))
check("LSTM de 128 unidades que recibe 2,048 + 16 = 2,064 valores", f"lstm.weight_ih_l0 {forma('lstm.weight_ih_l0')} (4 compuertas x 128, 2,064)",
      forma("lstm.weight_ih_l0") == (512, 2064))
check("Capa lineal 128 -> 5 clases", f"fc.weight {forma('fc.weight')}", forma("fc.weight") == (5, 128))
n = sum(q.numel() for q in modelo.parameters())
check("1,558,165 parámetros entrenables", f"{n:,}", n == 1558165)
torch.manual_seed(0)
nuevo = TinySleepNetEMG(TinySleepNetConfig(use_rnn=True, input_channels=8, seq_length=15), rama="mlp")
dif = (sd["fc.weight"] - nuevo.state_dict()["fc.weight"]).abs().mean().item()
finitos = all(torch.isfinite(v).all().item() for v in sd.values() if v.is_floating_point())
check("Son pesos entrenados y válidos (no los iniciales, sin NaN ni infinitos)",
      f"BatchNorm vio {int(sd['cnn.block1.bn.num_batches_tracked'])} lotes; todos finitos={finitos}; "
      f"diferencia media con una inicialización nueva en fc={dif:.4f}",
      finitos and int(sd["cnn.block1.bn.num_batches_tracked"]) > 0)

# --- 4. Predicción en las 12 combinaciones con un sujeto de ENTRENAMIENTO ---
print("\n== Predicción con un sujeto de entrenamiento (15 épocas) ==")
train, test = get_train_test_subject_codes()
archivo = None
for f in sorted(Path(args.npz).glob("*_epochs.npz")):
    d = np.load(f, allow_pickle=True)
    if str(d["subject_code"]) in train and (Path(args.emg) / f.name).exists():
        archivo = f
        break
if archivo is None:
    print(f"[-- ] No encuentro un sujeto de entrenamiento con sus dos archivos en {args.npz} y {args.emg}.")
    fallos += 1
else:
    d = np.load(archivo, allow_pickle=True)
    e = np.load(Path(args.emg) / archivo.name, allow_pickle=True)
    assert str(d["subject_code"]) not in test
    y = np.array([str(v) for v in d["y"]])
    i0 = int(np.argmax(y != "W")) if (y != "W").any() else 0          # un tramo con sueño, no solo vigilia
    sl = slice(i0, i0 + 15)
    nombres = list(POSICIONES_EMG[:3])
    print(f"Sujeto {d['subject_code']} ({archivo.name}, de entrenamiento), épocas {i0}-{i0 + 14}; etiquetas: {' '.join(y[sl])}")
    sumas_ok = True
    for nombre, presentes in combinaciones_validas(POSICIONES_EMG):
        usados = [c for c, pr in zip(nombres, presentes[:3]) if pr]
        X = d["X"][sl][:, [nombres.index(c) for c in usados]]
        entrada, _ = preparar_entrada(X, usados, posiciones=POSICIONES_EMG, emg=e["X"][sl][:, 0] if presentes[3] else None)
        with torch.no_grad():
            pr = torch.softmax(modelo(entrada.unsqueeze(0))[0], dim=-1)[0]
        sumas_ok &= torch.allclose(pr.sum(-1), torch.ones(15), atol=1e-5)
        pred = [["W", "N1", "N2", "N3", "REM"][k] for k in pr.argmax(-1).tolist()]
        acierto = np.mean(np.array(pred) == y[sl])
        print(f"   {nombre:52s} {' '.join(pred)}  ({acierto:.0%} de 15)")
    check("El modelo corre con las 12 combinaciones y entrega probabilidades que suman 1", f"suma de probabilidades = 1 en todas: {sumas_ok}", sumas_ok)
    print("       (los aciertos son de un sujeto de ENTRENAMIENTO: solo muestran que el modelo funciona, no son una métrica)")

print(f"\n{'Todo comprobado.' if fallos == 0 else f'{fallos} comprobación(es) por revisar.'}")
