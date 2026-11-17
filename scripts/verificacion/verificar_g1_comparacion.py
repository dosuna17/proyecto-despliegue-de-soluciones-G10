"""Revisa a fondo la comparación "modelo dedicado frente al modelo seleccionado (EMG-MLP)" de G1.

Parte 1 (siempre): validación cruzada pliegue por pliegue, desde los JSON de cada pliegue en MLflow.
    - ¿Los dos modelos se validaron sobre las mismas épocas en cada pliegue?
    - La diferencia pareada por pliegue (mismas personas): ¿en cuántos pliegues gana el dedicado?
Parte 2 (opcional): prueba con bootstrap PAREADO por persona, desde las probabilidades ya guardadas.
    No vuelve a evaluar el test ni cambia nada: analiza las predicciones del único test que se corrió.
    Pasa --probs con la carpeta resultados_test (bajada de Drive sueno_colab/resultados_test; basta con
    resultados_test.json y los probs_*.npz de flexible_emg_mlp y de los 5 controles).

Uso, desde la raíz del repositorio:
    python scripts/verificacion/verificar_g1_comparacion.py
    python scripts/verificacion/verificar_g1_comparacion.py --probs "C:\\...\\resultados_test"
"""
import argparse
import getpass
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

RUNS = {"flexible_emg_mlp": "95951e2a2cfd4368970636cc6bdea9b5",
        "control_eeg_fpz_cz": "2bd024695ad44d19b175490e5abbdedb", "control_eeg_pz_oz": "30877a344b304c5a8a2b5f2e3163805f",
        "control_eeg_fpz_cz__eeg_pz_oz": "076282f5e5534691b441514b50e90441",
        "control_eeg_fpz_cz__eog_horizontal": "038a654fc695444ab650f4e2aac9056f",
        "control_eeg_fpz_cz__eeg_pz_oz__eog_horizontal__emg_submental": "74b16a9d1a854bb5a5589353fec14f91"}
NOMBRE = {"eeg_fpz_cz": "Fpz-Cz", "eeg_pz_oz": "Pz-Oz", "eeg_fpz_cz__eeg_pz_oz": "Fpz-Cz + Pz-Oz",
          "eeg_fpz_cz__eog_horizontal": "Fpz-Cz + EOG",
          "eeg_fpz_cz__eeg_pz_oz__eog_horizontal__emg_submental": "Las cuatro señales"}

p = argparse.ArgumentParser()
p.add_argument("--probs", default=None, help="carpeta resultados_test con resultados_test.json y probs_*.npz")
p.add_argument("--runs-json", default=None, help="(pruebas) otro mapa modelo -> run_id")
p.add_argument("--n-boot", type=int, default=1000)
args = p.parse_args()
if args.runs_json:
    RUNS = json.load(open(args.runs_json, encoding="utf-8"))

# --- Parte 1: validación cruzada pliegue por pliegue ---
import mlflow  # noqa: E402
if not os.environ.get("MLFLOW_TRACKING_URI"):
    os.environ["MLFLOW_TRACKING_URI"] = input("MLFLOW_TRACKING_URI: ").strip()
    u = input("MLFLOW_TRACKING_USERNAME (vacío si no hace falta): ").strip()
    if u:
        os.environ["MLFLOW_TRACKING_USERNAME"] = u
        os.environ["MLFLOW_TRACKING_PASSWORD"] = getpass.getpass("MLFLOW_TRACKING_PASSWORD: ")
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
tmp = Path(tempfile.mkdtemp(prefix="verif_comp_"))
pl = {}
for m, rid in RUNS.items():
    d = Path(mlflow.artifacts.download_artifacts(run_id=rid, artifact_path="resultados", dst_path=str(tmp / m)))
    pl[m] = {json.load(open(a, encoding="utf-8"))["pliegue"]: json.load(open(a, encoding="utf-8"))
             for a in sorted(d.glob(f"{m}_pliegue[0-9].json"))}

print("== Parte 1. Validación cruzada, pliegue por pliegue (mismas personas en cada pliegue) ==")
print("   Diferencia = dedicado menos EMG-MLP, en puntos de F1 macro de la mejor época de cada modelo.\n")
mismas_epocas = True
resumen = {}
for c, n in NOMBRE.items():
    ded, emg = pl[f"control_{c}"], pl["flexible_emg_mlp"]
    difs, linea = [], []
    for k in sorted(ded):
        md, me = ded[k]["metricas_por_combinacion"][c], emg[k]["metricas_por_combinacion"][c]
        sd, se = np.array(md["confusion_matrix"]).sum(1), np.array(me["confusion_matrix"]).sum(1)
        mismas_epocas &= bool((sd == se).all())
        difs.append(100 * (md["macro_f1"] - me["macro_f1"]))
        linea.append(f"P{k}: {md['macro_f1']:.3f} vs {me['macro_f1']:.3f} ({difs[-1]:+.1f})")
    difs = np.array(difs)
    gana = int((difs > 0).sum())
    resumen[c] = (difs.mean(), difs.std(ddof=1), gana, len(difs))
    print(f"   {n:20s} media {difs.mean():+.1f} ± {difs.std(ddof=1):.1f} | el dedicado gana en {gana} de {len(difs)} pliegues")
    print(f"   {'':20s} " + " | ".join(linea))
print(f"\n   ¿Mismas épocas de validación (mismo número por estadio) en los dos modelos y en cada pliegue? {mismas_epocas}")
print("   Ojo: en la validación cruzada cada modelo se mide en la época que mejor le fue en ese mismo pliegue.")
print("   El dedicado elige esa época por su propio F1; el EMG-MLP, por el promedio de sus 12 combinaciones.")
print("   Eso favorece un poco al dedicado en esta comparación. La prueba no tiene ese sesgo.")

# --- Parte 2: prueba con bootstrap pareado por persona ---
if not args.probs:
    print("\n(Parte 2 omitida: pasa --probs con la carpeta resultados_test para el bootstrap pareado de la prueba.)")
else:
    car = Path(args.probs)
    R = json.load(open(car / "resultados_test.json", encoding="utf-8"))
    suj_rec = np.array([str(r)[3:5] for r in R["grabaciones"]])     # SCxssN...: ss = código de la persona
    print(f"\n== Parte 2. Prueba: bootstrap PAREADO por persona ({len(set(suj_rec))} personas, {args.n_boot} remuestreos, semilla 42) ==")
    print("   Diferencia = dedicado menos EMG-MLP, F1 macro sobre las mismas épocas.\n")
    rng = np.random.default_rng(42)
    for c, n in NOMBRE.items():
        a = np.load(car / f"probs_control_{c}__{c}.npz")
        b = np.load(car / f"probs_flexible_emg_mlp__{c}.npz")
        assert np.array_equal(a["y_true"], b["y_true"]) and np.array_equal(a["rec_idx"], b["rec_idx"]), "no son las mismas épocas"
        y, pa, pb = a["y_true"], a["probs"].argmax(1), b["probs"].argmax(1)
        suj = suj_rec[a["rec_idx"]]
        personas = np.unique(suj)
        idx = {s: np.flatnonzero(suj == s) for s in personas}
        f1 = lambda yy, pp: f1_score(yy, pp, average="macro", labels=[0, 1, 2, 3, 4], zero_division=0)
        obs = 100 * (f1(y, pa) - f1(y, pb))
        boots = []
        for _ in range(args.n_boot):
            sel = np.concatenate([idx[s] for s in rng.choice(personas, size=len(personas), replace=True)])
            boots.append(100 * (f1(y[sel], pa[sel]) - f1(y[sel], pb[sel])))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        frac = np.mean(np.array(boots) > 0)
        veredicto = "el intervalo incluye 0: sin diferencia clara" if lo <= 0 <= hi else (
            "el dedicado es mejor" if lo > 0 else "el EMG-MLP es mejor")
        print(f"   {n:20s} {obs:+.1f} puntos, IC95 pareado [{lo:+.1f}, {hi:+.1f}] | dedicado arriba en {frac:.0%} de los remuestreos | {veredicto}")
