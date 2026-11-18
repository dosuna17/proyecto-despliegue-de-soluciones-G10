"""Validación cruzada de los 8 modelos, F1 macro media ± desviación entre los 5 pliegues,
en validación y en entrenamiento, con el id del run de MLflow de cada cifra.

De cada pliegue se toma su mejor época (la de mayor F1 macro de validación, la que se guardó):
  - F1 de validación: el de cada combinación en esa época (metricas_por_combinacion) y, para los flexibles,
    el promedio de sus combinaciones (todas, y solo las 6 sin EMG, que es el criterio de selección).
  - F1 de entrenamiento: train_macro_f1 de esa época en el historial. Se mide durante la época, con abandono
    activo y con los pesos cambiando, y en los flexibles junta todas las combinaciones.
Desviación entre pliegues poblacional (ddof = 0), igual que el resumen del script 06.

Uso, desde la raíz del repositorio:
    python scripts/verificacion/tabla_validacion_cruzada.py --salida reports/tables
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
NOMBRE = {"flexible_emg_mlp": "Flexible con EMG (EMG-MLP)", "flexible": "Flexible sin EMG", "flexible_sin_lstm": "Flexible sin LSTM",
          "control_eeg_fpz_cz": "Dedicado Fpz-Cz", "control_eeg_pz_oz": "Dedicado Pz-Oz",
          "control_eeg_fpz_cz__eeg_pz_oz": "Dedicado Fpz-Cz + Pz-Oz", "control_eeg_fpz_cz__eog_horizontal": "Dedicado Fpz-Cz + EOG",
          "control_eeg_fpz_cz__eeg_pz_oz__eog_horizontal__emg_submental": "Dedicado de 4 señales"}
E, P, EO, M = "eeg_fpz_cz", "eeg_pz_oz", "eog_horizontal", "emg_submental"
COMB = {E: "Fpz-Cz", P: "Pz-Oz", f"{E}__{P}": "Fpz-Cz + Pz-Oz", f"{E}__{EO}": "Fpz-Cz + EOG", f"{P}__{EO}": "Pz-Oz + EOG",
        f"{E}__{P}__{EO}": "Fpz-Cz + Pz-Oz + EOG", f"{E}__{M}": "Fpz-Cz + EMG", f"{P}__{M}": "Pz-Oz + EMG",
        f"{E}__{P}__{M}": "Fpz-Cz + Pz-Oz + EMG", f"{E}__{EO}__{M}": "Fpz-Cz + EOG + EMG",
        f"{P}__{EO}__{M}": "Pz-Oz + EOG + EMG", f"{E}__{P}__{EO}__{M}": "Fpz-Cz + Pz-Oz + EOG + EMG"}
SIN_EMG = list(COMB)[:6]

ap = argparse.ArgumentParser()
ap.add_argument("--salida", default="reports/tables")
ap.add_argument("--cv-dir", default=None, help="(pruebas) carpeta con MODELO_pliegueN.json ya bajados")
args = ap.parse_args()
sal = Path(args.salida)
sal.mkdir(parents=True, exist_ok=True)

if args.cv_dir:
    carpetas = {m: Path(args.cv_dir) for m in RUNS}
else:
    import mlflow
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        os.environ["MLFLOW_TRACKING_URI"] = input("MLFLOW_TRACKING_URI: ").strip()
        u = input("MLFLOW_TRACKING_USERNAME (vacío si no hace falta): ").strip()
        if u:
            os.environ["MLFLOW_TRACKING_USERNAME"] = u
            os.environ["MLFLOW_TRACKING_PASSWORD"] = getpass.getpass("MLFLOW_TRACKING_PASSWORD: ")
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    tmp = Path(tempfile.mkdtemp(prefix="tabla_vc_"))
    carpetas = {m: Path(mlflow.artifacts.download_artifacts(run_id=rid, artifact_path="resultados", dst_path=str(tmp / m)))
                for m, rid in RUNS.items()}

ms = lambda v: (float(np.mean(v)), float(np.std(v)))          # ddof = 0
fmt = lambda t: f"{t[0]:.3f} ± {t[1]:.3f}"
res, por_comb = {}, {}
for m in RUNS:
    pl = sorted((json.load(open(a, encoding="utf-8")) for a in carpetas[m].glob(f"{m}_pliegue[0-9].json")), key=lambda x: x["pliegue"])
    if len(pl) != 5:
        print(f"   (omitido {m}: {len(pl)} pliegues)")
        continue
    combos = list(pl[0]["metricas_por_combinacion"])
    val_c = {c: [x["metricas_por_combinacion"][c]["macro_f1"] for x in pl] for c in combos}
    val_todas = [np.mean([x["metricas_por_combinacion"][c]["macro_f1"] for c in combos]) for x in pl]
    comunes = [c for c in combos if c in SIN_EMG]
    val_6 = [np.mean([x["metricas_por_combinacion"][c]["macro_f1"] for c in comunes]) for x in pl]
    ent = [x["historial"][x["mejor_epoca"] - 1]["train_macro_f1"] for x in pl]
    res[m] = dict(n=len(combos), val=ms(val_todas), val6=ms(val_6) if comunes else None, ent=ms(ent),
                  brecha=ms(np.array(ent) - np.array(val_todas)), mejores=[x["mejor_epoca"] for x in pl])
    por_comb[m] = {c: ms(v) for c, v in val_c.items()}

# ---- por modelo ----
md = ["**Validación cruzada por modelo** (5 pliegues por persona, 62 personas de entrenamiento; media ± desviación entre pliegues, "
      "F1 macro en la mejor época de cada pliegue)", "",
      "| Modelo | Run de MLflow | Combinaciones | F1 validación (todas sus combinaciones) | F1 validación (6 sin EMG) | F1 entrenamiento | Entrenamiento − validación | Mejores épocas |",
      "|---|---|---|---|---|---|---|---|"]
csv = ["modelo;run_id;n_combinaciones;f1_val_media;f1_val_desv;f1_val6_media;f1_val6_desv;f1_ent_media;f1_ent_desv;brecha_media;brecha_desv;mejores_epocas"]
for m, r in res.items():
    md.append(f"| {NOMBRE[m]} | `{RUNS[m]}` | {r['n']} | {fmt(r['val'])} | {fmt(r['val6']) if r['val6'] and r['n'] > 1 else '—'} | "
              f"{fmt(r['ent'])} | {r['brecha'][0]:+.3f} ± {r['brecha'][1]:.3f} | {r['mejores']} |")
    v6 = r["val6"] or (float("nan"), float("nan"))
    csv.append(";".join([m, RUNS[m], str(r["n"])] + [f"{x:.4f}" for x in (*r["val"], *v6, *r["ent"], *r["brecha"])] + [str(r["mejores"])]))

# ---- por combinación (con run) ----
md += ["", "**F1 macro de validación por combinación** (media ± desviación entre pliegues)", "",
       "| Combinación | " + " | ".join(NOMBRE[m] for m in res if m.startswith("flexible")) + " | Dedicado |",
       "|---" * (2 + sum(m.startswith("flexible") for m in res)) + "|"]
for c, n in COMB.items():
    celdas = [fmt(por_comb[m][c]) if c in por_comb[m] else "—" for m in res if m.startswith("flexible")]
    ded = f"control_{c}"
    celdas.append(fmt(por_comb[ded][c]) if ded in por_comb else "—")
    md.append(f"| {n} | " + " | ".join(celdas) + " |")
md += ["", "Runs: " + "; ".join(f"{NOMBRE[m]} `{RUNS[m]}`" for m in res) + "."]
(sal / "tabla_validacion_cruzada.md").write_text("\n".join(md) + "\n", encoding="utf-8")
(sal / "tabla_validacion_cruzada.csv").write_text("\n".join(csv) + "\n", encoding="utf-8")
print("\n".join(md))

# ---- comprobaciones contra lo ya verificado (verificar_validacion_cruzada.py) ----
fallos = 0
def check(a, o, ok):
    global fallos
    fallos += not ok
    print(f"[{'OK ' if ok else 'NO '}] {a}\n       observado: {o}")
r3 = lambda x: float(f"{x:.3f}")
print()
if "flexible_emg_mlp" in res:
    check("EMG-MLP: F1 de validación por combinación igual a lo verificado en verificar_validacion_cruzada.py (Fpz-Cz 0.759 ± 0.022; las 4 señales 0.771 ± 0.034)",
          f"{fmt(por_comb['flexible_emg_mlp'][E])}; {fmt(por_comb['flexible_emg_mlp'][f'{E}__{P}__{EO}__{M}'])}",
          (r3(por_comb["flexible_emg_mlp"][E][0]), r3(por_comb["flexible_emg_mlp"][E][1])) == (0.759, 0.022)
          and (r3(por_comb["flexible_emg_mlp"][f"{E}__{P}__{EO}__{M}"][0]), r3(por_comb["flexible_emg_mlp"][f"{E}__{P}__{EO}__{M}"][1])) == (0.771, 0.034))
if {"flexible_emg_mlp", "flexible"} <= set(res):
    check("Criterio de selección (6 combinaciones sin EMG): EMG-MLP 0.7445 frente a flexible 0.7381 (a 3 decimales, ver el valor exacto)",
          f"{res['flexible_emg_mlp']['val6'][0]:.6f} frente a {res['flexible']['val6'][0]:.6f}",
          abs(res["flexible_emg_mlp"]["val6"][0] - 0.7445) < 1e-4 and abs(res["flexible"]["val6"][0] - 0.7381) < 1e-4)
if len(res) == 8:
    check("Brecha media entrenamiento − validación: 0.011 a 0.036 en los flexibles y 0.036 a 0.082 en los dedicados (figura_sobreajuste.py)",
          ", ".join(f"{NOMBRE[m]} {r['brecha'][0]:+.3f}" for m, r in res.items()),
          (r3(min(r["brecha"][0] for m, r in res.items() if m.startswith("flex"))), r3(max(r["brecha"][0] for m, r in res.items() if m.startswith("flex")))) == (0.011, 0.036)
          and (r3(min(r["brecha"][0] for m, r in res.items() if m.startswith("control"))), r3(max(r["brecha"][0] for m, r in res.items() if m.startswith("control")))) == (0.036, 0.082))
print(f"\nArchivos en {sal}. {'Todo coincide.' if fallos == 0 else f'{fallos} no coinciden: revisar.'}")
