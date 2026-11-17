"""Verifica, con el código, los datos y los resultados guardados en MLflow, el párrafo "Entrenamiento con
todas las combinaciones" de G1.

Uso (desde la raíz del repositorio, con el entorno donde están torch y mlflow):
    python scripts/verificacion/verificar_g1_entrenamiento.py
    (pide el URI, el usuario y la clave de MLflow si no están en el entorno)

No entrena ni modifica nada. Descarga solo JSON y CSV pequeños de los 8 runs (no los pesos).
La parte con datos usa UN sujeto de entrenamiento (nunca de prueba).
"""
import argparse
import csv
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
from data.dataset import get_train_test_subject_codes  # noqa: E402
from sleep_staging.training import train_flexible as tf  # noqa: E402
from sleep_staging.training.canales_flexibles import (  # noqa: E402
    POSICIONES_EMG, FlexibleSequenceDataset, combinaciones_validas, preparar_entrada)
from sleep_staging.training.emg_datos import cargar_con_emg  # noqa: E402

RUNS = {"flexible": "029cc677c3734b10959735a6e9246cb8", "flexible_emg_mlp": "95951e2a2cfd4368970636cc6bdea9b5",
        "flexible_sin_lstm": "c424e04708c34f04a5bfec810b705982",
        "control_eeg_fpz_cz": "2bd024695ad44d19b175490e5abbdedb", "control_eeg_pz_oz": "30877a344b304c5a8a2b5f2e3163805f",
        "control_eeg_fpz_cz__eeg_pz_oz": "076282f5e5534691b441514b50e90441",
        "control_eeg_fpz_cz__eog_horizontal": "038a654fc695444ab650f4e2aac9056f",
        "control_eeg_fpz_cz__eeg_pz_oz__eog_horizontal__emg_submental": "74b16a9d1a854bb5a5589353fec14f91"}
NOMBRE = {"eeg_fpz_cz": "Fpz-Cz", "eeg_pz_oz": "Pz-Oz", "eeg_fpz_cz__eeg_pz_oz": "Fpz-Cz + Pz-Oz",
          "eeg_fpz_cz__eog_horizontal": "Fpz-Cz + EOG",
          "eeg_fpz_cz__eeg_pz_oz__eog_horizontal__emg_submental": "Las cuatro señales"}

p = argparse.ArgumentParser()
p.add_argument("--npz", default=str(RAIZ / "data/interim/epochs_por_sujeto_3canales_recortado"))
p.add_argument("--emg", default=str(Path.home() / "Desktop/experimento_emg/datos_emg/epochs_emg_recortado"))
p.add_argument("--runs-json", default=None, help="(pruebas) otro mapa modelo -> run_id")
args = p.parse_args()
if args.runs_json:
    RUNS = json.load(open(args.runs_json, encoding="utf-8"))

fallos = 0
def check(afirmacion, observado, ok):
    global fallos
    fallos += not ok
    print(f"[{'OK ' if ok else 'NO '}] {afirmacion}\n       observado: {observado}")

combos = combinaciones_validas(POSICIONES_EMG)

# --- A. Cómo se arman las presentaciones (código + un sujeto de entrenamiento) ---
print("== A. Cada secuencia, una vez por combinación ==")
train, test = get_train_test_subject_codes()
sujeto = None
for f in sorted(Path(args.npz).glob("*_epochs.npz")):
    s = str(np.load(f, allow_pickle=True)["subject_code"])
    if s in train and (Path(args.emg) / f.name).exists():
        sujeto = s
        break
if sujeto is None:
    print(f"[-- ] No encuentro un sujeto de entrenamiento con sus dos archivos en {args.npz} y {args.emg}.")
    fallos += 1
else:
    recs = cargar_con_emg(Path(args.npz), subject_codes=[sujeto], max_epochs_per_recording=tf.base.MAX_EPOCHS_PER_RECORDING,
                          emg_dir=Path(args.emg))
    ds = FlexibleSequenceDataset(recs, combos, posiciones=POSICIONES_EMG)
    nb = len(ds.base)
    check("Cada secuencia de 15 épocas se presenta una vez por combinación (12 veces) en cada época de entrenamiento",
          f"sujeto {sujeto} (entrenamiento), {len(recs)} grabaciones: {nb} secuencias -> {len(ds)} presentaciones = {len(ds) // nb} x {nb}",
          len(ds) == 12 * nb)
    vistas, mismas_y, ausentes_ok, marca_ok, presentes_ok, constante_ok = [], True, True, True, True, True
    x0, y0 = ds.base[0]
    for i in range(12):
        x, y = ds[i]
        pres = ds.combinaciones[i % 12][1]
        vistas.append(ds.combinaciones[i % 12][0])
        mismas_y &= torch.equal(y, y0)
        for k, pr in enumerate(pres):
            marca = x[:, 4 + k]
            constante_ok &= bool((marca == marca.flatten()[0]).all())       # misma marca en las 15 épocas
            if pr:
                presentes_ok &= torch.equal(x[:, k], x0[:, k]) and bool((marca == 1).all())
            else:
                ausentes_ok &= bool((x[:, k] == 0).all())
                marca_ok &= bool((marca == 0).all())
    check("Las 12 presentaciones de la secuencia 0 cubren las 12 combinaciones, con las mismas etiquetas",
          f"{len(set(vistas))} combinaciones distintas; etiquetas iguales={mismas_y}", len(set(vistas)) == 12 and mismas_y)
    check("Canales ausentes en ceros con marca 0; presentes intactos con marca 1",
          f"ausentes en cero={ausentes_ok}, su marca en 0={marca_ok}, presentes intactos y marca 1={presentes_ok}",
          ausentes_ok and marca_ok and presentes_ok)
    check("Las 15 épocas de una presentación comparten la misma combinación",
          f"cada marca es constante a lo largo de las 15 épocas y de las 3,000 muestras: {constante_ok}", constante_ok)
codigo_cv = inspect.getsource(tf.run_cv_flexible)
check("En la validación cruzada se entrena con todas las combinaciones (las mismas que se validan)",
      "run_cv_flexible llama entrenar_pliegue(k, ..., combos, combos, ...)", "entrenar_pliegue(k, fold_train, fold_val, combos, combos" in codigo_cv)
_, pres_api = preparar_entrada(torch.zeros(30, 1, 3000), ["EEG Fpz-Cz"], posiciones=POSICIONES_EMG)
check("Al usar el modelo, un estudio entra con una sola combinación para toda la noche",
      f"preparar_entrada devuelve una sola combinación para las 30 épocas del ejemplo: {pres_api}", isinstance(pres_api, tuple))
print("       (que un estudio llegue con los mismos canales toda la noche es el supuesto de diseño; el código lo respeta)")

# --- B. Parada temprana y mejor época: promedio sobre las combinaciones ---
print("\n== B. Parada temprana y mejor época ==")
codigo_pl = inspect.getsource(tf.entrenar_pliegue)
check("La mejor época y la parada temprana se deciden con el promedio del F1 macro de validación de todas las combinaciones",
      "entrenar_pliegue: f1_medio = media de macro_f1 por combinación; 'if f1_medio > mejor_f1' decide mejor época y paciencia",
      'f1_medio = float(np.mean([m["macro_f1"] for m in metricas.values()]))' in codigo_pl and "if f1_medio > mejor_f1" in codigo_pl)

# --- C. Lo que pasó de verdad en cada pliegue (MLflow) ---
import mlflow  # noqa: E402
if not os.environ.get("MLFLOW_TRACKING_URI"):
    os.environ["MLFLOW_TRACKING_URI"] = input("MLFLOW_TRACKING_URI: ").strip()
    u = input("MLFLOW_TRACKING_USERNAME (vacío si no hace falta): ").strip()
    if u:
        os.environ["MLFLOW_TRACKING_USERNAME"] = u
        os.environ["MLFLOW_TRACKING_PASSWORD"] = getpass.getpass("MLFLOW_TRACKING_PASSWORD: ")
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
tmp = Path(tempfile.mkdtemp(prefix="verif_g1_"))
res, fin, pliegues = {}, {}, {}
for m, rid in RUNS.items():
    d = Path(mlflow.artifacts.download_artifacts(run_id=rid, artifact_path="resultados", dst_path=str(tmp / m)))
    res[m] = json.load(open(d / f"{m}_resumen.json", encoding="utf-8"))
    pliegues[m] = [json.load(open(a, encoding="utf-8")) for a in sorted(d.glob(f"{m}_pliegue[0-9].json"))]
    fj = Path(mlflow.artifacts.download_artifacts(run_id=rid, artifact_path=f"modelo_final/{m}_final.json", dst_path=str(tmp / m)))
    fin[m] = json.load(open(fj, encoding="utf-8"))
tabla = Path(mlflow.artifacts.download_artifacts(run_id=RUNS["flexible_emg_mlp"], artifact_path="test/tabla_test.csv", dst_path=str(tmp)))

print("\n== C. Lo que pasó en cada pliegue (resultados guardados en MLflow) ==")
for m in ["flexible_emg_mlp", "flexible", "flexible_sin_lstm"]:
    pl = pliegues[m]
    medias_ok = all(abs(x["val_macro_f1_medio"] - np.mean([c["macro_f1"] for c in x["metricas_por_combinacion"].values()])) < 1e-9 for x in pl)
    mejor_ok = all(x["mejor_epoca"] == max(x["historial"], key=lambda h: h["val_macro_f1_medio"])["epoca"] for x in pl)
    ncomb = {len(x["metricas_por_combinacion"]) for x in pl}
    paradas = [len(x["historial"]) - x["mejor_epoca"] for x in pl]
    check(f"{m}: el F1 que eligió la mejor época es el promedio de sus combinaciones, en los {len(pl)} pliegues",
          f"combinaciones validadas por pliegue={sorted(ncomb)}; F1 elegido = media de las combinaciones: {medias_ok}; "
          f"mejor época = la de mayor promedio: {mejor_ok}", medias_ok and mejor_ok)
    print(f"       mejores épocas={[x['mejor_epoca'] for x in pl]}, épocas corridas={[len(x['historial']) for x in pl]}, "
          f"épocas tras la mejor={paradas} (paciencia; si alguna es menor, ese pliegue llegó al máximo de épocas)")
cvs = sorted({len(x["historial"]) - x["mejor_epoca"] for m in RUNS if m.startswith("control") for x in pliegues[m]})
print(f"       controles: épocas tras la mejor en sus pliegues = {cvs}")

print("\n== D. Mismo procedimiento para el flexible sin EMG y la ablación sin LSTM ==")
for m in ["flexible_emg_mlp", "flexible", "flexible_sin_lstm"]:
    f = fin[m]
    print(f"       {m:20s} combinaciones={len(f['combinaciones'])}, emg={f.get('emg')}, sin_lstm={f.get('sin_lstm')}, "
          f"épocas finales={f['n_epocas']} (media de la VC={f.get('epocas_cv')}), noche completa={f.get('noche_completa')}, grabaciones={f['n_grabaciones']}")
iguales = all(fin[m]["n_epocas"] == fin[m].get("epocas_cv") and fin[m].get("noche_completa") is True
              and fin[m]["n_grabaciones"] == fin["flexible_emg_mlp"]["n_grabaciones"] for m in ["flexible", "flexible_sin_lstm"])
check("El flexible sin EMG tiene las 6 combinaciones de 100 Hz y la ablación es el flexible sin LSTM, ambos con el mismo procedimiento",
      f"flexible: {len(fin['flexible']['combinaciones'])} combinaciones, emg={fin['flexible'].get('emg')}; "
      f"sin LSTM: sin_lstm={fin['flexible_sin_lstm'].get('sin_lstm')}; final = media de mejores épocas, noches completas y mismas grabaciones: {iguales}",
      len(fin["flexible"]["combinaciones"]) == 6 and not fin["flexible"].get("emg") and fin["flexible_sin_lstm"].get("sin_lstm") is True and iguales)
check("Un solo archivo de pesos para las 12 combinaciones",
      f"modelo_final/flexible_emg_mlp_final.json declara {len(fin['flexible_emg_mlp']['combinaciones'])} combinaciones",
      len(fin["flexible_emg_mlp"]["combinaciones"]) == 12)

print("\n== E. Costo frente a los modelos dedicados ==")
print("       Validación cruzada, F1 macro medio (dedicado menos flexible, en puntos):")
dif_emg, dif_flex = {}, {}
for c, n in NOMBRE.items():
    ded = res[f"control_{c}"]["combinaciones"][c]["macro_f1_media"]
    e = res["flexible_emg_mlp"]["combinaciones"][c]["macro_f1_media"]
    fl = res["flexible"]["combinaciones"].get(c, {}).get("macro_f1_media")
    dif_emg[c] = 100 * (ded - e)
    if fl is not None:
        dif_flex[c] = 100 * (ded - fl)
    print(f"       {n:20s} dedicado {ded:.3f} | EMG-MLP {e:.3f} ({dif_emg[c]:+.1f}) | flexible "
          + (f"{fl:.3f} ({dif_flex[c]:+.1f})" if fl is not None else "  —"))
sin_pz_emg = [v for c, v in dif_emg.items() if c != "eeg_pz_oz"]
sin_pz_flex = [v for c, v in dif_flex.items() if c != "eeg_pz_oz"]
rango = lambda v: f"{min(v):.1f} a {max(v):.1f}"
check("Texto actual: 'el dedicado supera al flexible por 0.4 a 1.4 puntos (3.2 con Pz-Oz sola)'",
      f"frente al EMG-MLP (el modelo descrito): {rango(sin_pz_emg)} y {dif_emg['eeg_pz_oz']:.1f} con Pz-Oz; "
      f"frente al flexible sin EMG: {rango(sin_pz_flex)} y {dif_flex['eeg_pz_oz']:.1f} con Pz-Oz",
      round(min(sin_pz_emg), 1) == 0.4 and round(max(sin_pz_emg), 1) == 1.4 and round(dif_emg["eeg_pz_oz"], 1) == 3.2)
filas = {(r["modelo"], r["combinacion"]): r for r in csv.DictReader(open(tabla, encoding="utf-8-sig"))}
print("       Prueba, F1 macro [IC95 por persona]:")
solapan = True
for c, n in NOMBRE.items():
    d, e = filas[(f"control_{c}", c)], filas[("flexible_emg_mlp", c)]
    s = float(d["macro_f1_ic95_bajo"]) <= float(e["macro_f1_ic95_alto"]) and float(e["macro_f1_ic95_bajo"]) <= float(d["macro_f1_ic95_alto"])
    solapan &= s
    print(f"       {n:20s} dedicado {d['macro_f1']} [{d['macro_f1_ic95_bajo']}, {d['macro_f1_ic95_alto']}] | "
          f"EMG-MLP {e['macro_f1']} [{e['macro_f1_ic95_bajo']}, {e['macro_f1_ic95_alto']}] | se solapan: {s}")
check("En prueba los intervalos de confianza del dedicado y del EMG-MLP se solapan en las 5 combinaciones", f"todas se solapan: {solapan}", solapan)
print("       (que los intervalos se solapen no prueba que no haya diferencia: no se hizo una comparación pareada)")

print(f"\n{'Todo comprobado.' if fallos == 0 else f'{fallos} afirmación(es) por revisar.'}")
