"""Cómputo: parámetros y tiempos de entrenamiento de los 8 modelos, desde MLflow.

Lee de cada run los JSON por pliegue (resultados/MODELO_pliegueN.json: segundos por época de entrenamiento y
de validación) y el JSON y los pesos del modelo final (modelo_final/MODELO_final.json y .pth). No entrena nada.
Los tiempos son de cómputo (suma de pliegues), no de reloj: los pliegues pudieron correr en sesiones paralelas.

Uso, desde la raíz del repositorio:
    python scripts/verificacion/verificar_computo.py --inferencia
    (pide el URI, el usuario y la clave de MLflow si no están en el entorno)
"""
import argparse
import getpass
import json
import os
import tempfile
from pathlib import Path

import numpy as np

RUNS = {"flexible_emg_mlp": "95951e2a2cfd4368970636cc6bdea9b5", "flexible": "029cc677c3734b10959735a6e9246cb8",
        "flexible_sin_lstm": "c424e04708c34f04a5bfec810b705982",
        "control_eeg_fpz_cz": "2bd024695ad44d19b175490e5abbdedb", "control_eeg_pz_oz": "30877a344b304c5a8a2b5f2e3163805f",
        "control_eeg_fpz_cz__eeg_pz_oz": "076282f5e5534691b441514b50e90441",
        "control_eeg_fpz_cz__eog_horizontal": "038a654fc695444ab650f4e2aac9056f",
        "control_eeg_fpz_cz__eeg_pz_oz__eog_horizontal__emg_submental": "74b16a9d1a854bb5a5589353fec14f91"}

p = argparse.ArgumentParser()
p.add_argument("--local", default=None, help="(pruebas) carpeta con MODELO_pliegueN.json, MODELO_final.json y .pth")
p.add_argument("--inferencia", action="store_true", help="además mide cuánto tarda en este equipo una noche completa")
args = p.parse_args()

fallos = 0
def check(afirmacion, observado, ok):
    global fallos
    fallos += not ok
    print(f"[{'OK ' if ok else 'NO '}] {afirmacion}\n       observado: {observado}")

if args.local:
    carp = {m: Path(args.local) for m in RUNS}
    fin = {m: Path(args.local) for m in RUNS}
    tags = {m: {} for m in RUNS}
else:
    import mlflow
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        os.environ["MLFLOW_TRACKING_URI"] = input("MLFLOW_TRACKING_URI: ").strip()
        u = input("MLFLOW_TRACKING_USERNAME (vacío si no hace falta): ").strip()
        if u:
            os.environ["MLFLOW_TRACKING_USERNAME"] = u
            os.environ["MLFLOW_TRACKING_PASSWORD"] = getpass.getpass("MLFLOW_TRACKING_PASSWORD: ")
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    tmp = Path(tempfile.mkdtemp(prefix="computo_"))
    carp, fin, tags = {}, {}, {}
    for m, rid in RUNS.items():
        carp[m] = Path(mlflow.artifacts.download_artifacts(run_id=rid, artifact_path="resultados", dst_path=str(tmp / m)))
        fin[m] = Path(mlflow.artifacts.download_artifacts(run_id=rid, artifact_path="modelo_final", dst_path=str(tmp / m)))
        r = mlflow.get_run(rid)
        tags[m] = {k: v for k, v in {**r.data.params, **r.data.tags}.items()
                   if any(s in k.lower() for s in ("gpu", "device", "dispositivo", "cuda", "hardware"))}

def n_parametros(pth):
    import torch
    sd = torch.load(pth, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd.get("model_state_dict", sd)) if isinstance(sd, dict) else sd
    return sum(v.numel() for k, v in sd.items()
               if hasattr(v, "numel") and "running_" not in k and "num_batches_tracked" not in k)

T = {}
print("== Tiempo de cómputo por modelo (suma de pliegues; segundos de GPU) ==")
print(f"   {'modelo':62s} {'VC (h)':>7s} {'s/época entr.':>14s} {'final (min)':>11s} {'s/época final':>14s} {'parámetros':>11s}")
for m in RUNS:
    pl = sorted((json.load(open(a, encoding="utf-8")) for a in carp[m].glob(f"{m}_pliegue[0-9].json")), key=lambda x: x["pliegue"])
    jf = json.load(open(next(fin[m].glob(f"{m}_final.json")), encoding="utf-8"))
    pth = next(fin[m].glob(f"{m}_final.pth"), None)
    ent = [h["segundos_entrenar"] for x in pl for h in x["historial"]]
    T[m] = dict(n_pl=len(pl), vc=sum(x["segundos_pliegue"] for x in pl), por_pl=[x["segundos_pliegue"] for x in pl],
                ep_ent=np.mean(ent), ep_val=np.mean([h["segundos_validar"] for x in pl for h in x["historial"]]),
                fin=jf.get("segundos_totales"), ep_fin=np.mean([h["segundos"] for h in jf["historial"]]),
                n_fin=len(jf["historial"]), params=n_parametros(pth) if pth else None)
    t = T[m]
    print(f"   {m:62s} {t['vc'] / 3600:7.2f} {t['ep_ent']:14.0f} {t['fin'] / 60:11.1f} {t['ep_fin']:14.0f} {t['params'] or 0:>11,}")
    if tags[m]:
        print(f"   {'':62s} etiquetas/parámetros de hardware en MLflow: {tags[m]}")

e = T["flexible_emg_mlp"]
check("EMG-MLP: validación cruzada 9,300 s (≈ 2.6 h) en 5 pliegues, de 1,361 a 2,239 s por pliegue",
      f"{e['vc']:,.0f} s = {e['vc'] / 3600:.2f} h; pliegues {[round(x) for x in e['por_pl']]}",
      e["n_pl"] == 5 and round(e["vc"], -2) == 9300 and (round(min(e["por_pl"])), round(max(e["por_pl"]))) == (1361, 2239))
check("EMG-MLP: modelo final 12 épocas de unos 300 s (noche completa) = 61 min; 1,558,165 parámetros",
      f"{e['n_fin']} épocas, {e['ep_fin']:.0f} s por época, {e['fin'] / 60:.1f} min; {e['params']:,} parámetros",
      e["n_fin"] == 12 and round(e["fin"] / 60) == 61 and e["params"] == 1558165)
tot = sum(t["vc"] + t["fin"] for t in T.values())
gpu_h = {"flexibles": sum(T[m]["vc"] + T[m]["fin"] for m in T if m.startswith("flexible")) / 3600,
         "controles": sum(T[m]["vc"] + T[m]["fin"] for m in T if m.startswith("control")) / 3600}
print(f"\n   Flexibles (3 modelos): {gpu_h['flexibles']:.1f} h; controles (5 modelos): {gpu_h['controles']:.1f} h")
print(f"\n   Total de los 8 modelos (validación cruzada + final): {tot / 3600:.1f} h de GPU")
# ---------------- Inferencia en este equipo ----------------
if args.inferencia:
    import platform
    import time
    import sys
    import torch
    sys.path.insert(0, str(Path.cwd() / "src"))
    from sleep_staging.models.tinysleepnet_emg import TinySleepNetEMG
    from sleep_staging.models.tinysleepnet import TinySleepNetConfig
    torch.manual_seed(0)
    mod = TinySleepNetEMG(TinySleepNetConfig(use_rnn=True, input_channels=8, seq_length=15), rama="mlp").eval()
    n_ep = 1095                                   # una noche típica de prueba: 73 secuencias de 15 épocas
    x = torch.randn(73, 15, 8, 3000)
    print(f"\n== Inferencia de una noche ({n_ep} épocas, 12 combinaciones posibles; se mide una) en este equipo ==")
    dispositivos = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for dev in dispositivos:
        m2, x2 = mod.to(dev), x.to(dev)
        with torch.no_grad():
            m2(x2[:2])
            if dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for i in range(0, len(x2), 20):
                m2(x2[i:i + 20])
            if dev == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
        nombre = torch.cuda.get_device_name(0) if dev == "cuda" else (platform.processor() or platform.machine())
        print(f"   {dev}: {dt:.1f} s por noche ({1000 * dt / n_ep:.1f} ms por época) | {nombre} | torch {torch.__version__}, "
              f"{torch.get_num_threads()} hilos")

print(f"\n{'Todo coincide.' if fallos == 0 else f'{fallos} afirmación(es) no coinciden: revisar.'}")
