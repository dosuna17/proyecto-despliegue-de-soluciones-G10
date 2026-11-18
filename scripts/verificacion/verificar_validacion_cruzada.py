"""Comprueba cada afirmación y cada cifra de la validación cruzada contra los JSON de los 8 runs en MLflow.

Uso, desde la raíz del repositorio:
    python scripts/verificacion/verificar_validacion_cruzada.py
    (pide el URI, el usuario y la clave de MLflow si no están en el entorno)

Recalcula media y desviación de cada combinación a partir de los 5 pliegues (no del resumen) y las compara
con las tablas de validación cruzada del artículo (F1 macro y kappa). Lee solo etiquetas de los .npz de entrenamiento, nunca señales.
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

RAIZ = Path.cwd()
sys.path.insert(0, str(RAIZ / "src"))
from data import dataset as ds  # noqa: E402
from sleep_staging.training import train_flexible as tf  # noqa: E402
from sleep_staging.training import train_tinysleepnet as base  # noqa: E402

RUNS = {"flexible": "029cc677c3734b10959735a6e9246cb8", "flexible_emg_mlp": "95951e2a2cfd4368970636cc6bdea9b5",
        "flexible_sin_lstm": "c424e04708c34f04a5bfec810b705982",
        "control_eeg_fpz_cz": "2bd024695ad44d19b175490e5abbdedb", "control_eeg_pz_oz": "30877a344b304c5a8a2b5f2e3163805f",
        "control_eeg_fpz_cz__eeg_pz_oz": "076282f5e5534691b441514b50e90441",
        "control_eeg_fpz_cz__eog_horizontal": "038a654fc695444ab650f4e2aac9056f",
        "control_eeg_fpz_cz__eeg_pz_oz__eog_horizontal__emg_submental": "74b16a9d1a854bb5a5589353fec14f91"}
C = ["eeg_fpz_cz", "eeg_pz_oz", "eeg_fpz_cz__eeg_pz_oz", "eeg_fpz_cz__eog_horizontal", "eeg_pz_oz__eog_horizontal",
     "eeg_fpz_cz__eeg_pz_oz__eog_horizontal", "eeg_fpz_cz__emg_submental", "eeg_pz_oz__emg_submental",
     "eeg_fpz_cz__eeg_pz_oz__emg_submental", "eeg_fpz_cz__eog_horizontal__emg_submental",
     "eeg_pz_oz__eog_horizontal__emg_submental", "eeg_fpz_cz__eeg_pz_oz__eog_horizontal__emg_submental"]
COMUNES = C[:6]
CTRL = ["eeg_fpz_cz", "eeg_pz_oz", "eeg_fpz_cz__eeg_pz_oz", "eeg_fpz_cz__eog_horizontal",
        "eeg_fpz_cz__eeg_pz_oz__eog_horizontal__emg_submental"]
# Tablas de validación cruzada (F1 macro y kappa) tal como están en el artículo: (media, desviación) por combinación; columnas Flexible, EMG-MLP, Sin LSTM, Dedicado
T4A = {"eeg_fpz_cz": [(0.753, 0.026), (0.759, 0.022), (0.706, 0.025), (0.766, 0.027)],
       "eeg_pz_oz": [(0.667, 0.052), (0.678, 0.036), (0.612, 0.028), (0.710, 0.033)],
       "eeg_fpz_cz__eeg_pz_oz": [(0.754, 0.032), (0.764, 0.023), (0.720, 0.032), (0.768, 0.030)],
       "eeg_fpz_cz__eog_horizontal": [(0.762, 0.037), (0.766, 0.029), (0.732, 0.035), (0.774, 0.031)],
       "eeg_pz_oz__eog_horizontal": [(0.725, 0.042), (0.729, 0.041), (0.683, 0.039), None],
       "eeg_fpz_cz__eeg_pz_oz__eog_horizontal": [(0.767, 0.036), (0.771, 0.032), (0.741, 0.036), None],
       "eeg_fpz_cz__emg_submental": [None, (0.762, 0.024), None, None],
       "eeg_pz_oz__emg_submental": [None, (0.683, 0.035), None, None],
       "eeg_fpz_cz__eeg_pz_oz__emg_submental": [None, (0.766, 0.023), None, None],
       "eeg_fpz_cz__eog_horizontal__emg_submental": [None, (0.768, 0.032), None, None],
       "eeg_pz_oz__eog_horizontal__emg_submental": [None, (0.732, 0.039), None, None],
       "eeg_fpz_cz__eeg_pz_oz__eog_horizontal__emg_submental": [None, (0.771, 0.034), None, (0.781, 0.030)]}
T4B = {"eeg_fpz_cz": [(0.719, 0.045), (0.731, 0.044), (0.661, 0.051), (0.736, 0.049)],
       "eeg_pz_oz": [(0.605, 0.084), (0.626, 0.059), (0.564, 0.043), (0.667, 0.051)],
       "eeg_fpz_cz__eeg_pz_oz": [(0.721, 0.048), (0.733, 0.045), (0.677, 0.058), (0.737, 0.049)],
       "eeg_fpz_cz__eog_horizontal": [(0.728, 0.054), (0.733, 0.050), (0.692, 0.058), (0.736, 0.050)],
       "eeg_pz_oz__eog_horizontal": [(0.682, 0.062), (0.689, 0.060), (0.636, 0.062), None],
       "eeg_fpz_cz__eeg_pz_oz__eog_horizontal": [(0.735, 0.051), (0.739, 0.052), (0.704, 0.059), None],
       "eeg_fpz_cz__emg_submental": [None, (0.732, 0.044), None, None],
       "eeg_pz_oz__emg_submental": [None, (0.630, 0.057), None, None],
       "eeg_fpz_cz__eeg_pz_oz__emg_submental": [None, (0.734, 0.043), None, None],
       "eeg_fpz_cz__eog_horizontal__emg_submental": [None, (0.735, 0.053), None, None],
       "eeg_pz_oz__eog_horizontal__emg_submental": [None, (0.693, 0.055), None, None],
       "eeg_fpz_cz__eeg_pz_oz__eog_horizontal__emg_submental": [None, (0.739, 0.053), None, (0.748, 0.049)]}

p = argparse.ArgumentParser()
p.add_argument("--npz", default=str(RAIZ / "data/interim/epochs_por_sujeto_3canales_recortado"))
p.add_argument("--runs-json", default=None, help="(pruebas) otro mapa modelo -> run_id")
args = p.parse_args()
if args.runs_json:
    RUNS = json.load(open(args.runs_json, encoding="utf-8"))

fallos = 0
def check(afirmacion, observado, ok):
    global fallos
    fallos += not ok
    print(f"[{'OK ' if ok else 'NO '}] {afirmacion}\n       observado: {observado}")

import mlflow  # noqa: E402
if not os.environ.get("MLFLOW_TRACKING_URI"):
    os.environ["MLFLOW_TRACKING_URI"] = input("MLFLOW_TRACKING_URI: ").strip()
    u = input("MLFLOW_TRACKING_USERNAME (vacío si no hace falta): ").strip()
    if u:
        os.environ["MLFLOW_TRACKING_USERNAME"] = u
        os.environ["MLFLOW_TRACKING_PASSWORD"] = getpass.getpass("MLFLOW_TRACKING_PASSWORD: ")
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
tmp = Path(tempfile.mkdtemp(prefix="validacion_cruzada_"))
pl, fin = {}, {}
for m, rid in RUNS.items():
    d = Path(mlflow.artifacts.download_artifacts(run_id=rid, artifact_path="resultados", dst_path=str(tmp / m)))
    pl[m] = sorted((json.load(open(a, encoding="utf-8")) for a in d.glob(f"{m}_pliegue[0-9].json")), key=lambda x: x["pliegue"])
    fin[m] = json.load(open(mlflow.artifacts.download_artifacts(run_id=rid, artifact_path=f"modelo_final/{m}_final.json",
                                                                dst_path=str(tmp / m)), encoding="utf-8"))

def stats(m, c, k):
    v = np.array([x["metricas_por_combinacion"][c][k] for x in pl[m]])
    return v.mean(), v.std()            # desviación poblacional (np.std, ddof=0), igual que resumir()

print("\n== Protocolo ==")
train, test = ds.get_train_test_subject_codes()
folds = ds.make_subject_kfold_splits(train)
mismos = "make_subject_kfold_splits(train_codes)" in inspect.getsource(tf.run_cv_flexible) and \
         "make_subject_kfold_splits(train_codes)" in inspect.getsource(base)
check("5 pliegues por persona sobre las 62 personas de entrenamiento, los mismos de la versión base",
      f"{len(folds)} pliegues, {len(train)} personas; la versión base y la flexible arman los pliegues con la misma función: {mismos}; "
      f"todos los runs tienen 5 pliegues: {all(len(v) == 5 for v in pl.values())}",
      len(folds) == 5 and len(train) == 62 and mismos and all(len(v) == 5 for v in pl.values()))
ok_sel = all(x["mejor_epoca"] == max(x["historial"], key=lambda h: h["val_macro_f1_medio"])["epoca"] for v in pl.values() for x in v)
check("En cada pliegue se toma la época con mayor F1 macro de validación (en los flexibles, el promedio de sus combinaciones)",
      f"en los 40 pliegues (8 modelos × 5), la mejor época es la de mayor F1 de validación: {ok_sel}", ok_sel)
print("       La desviación de las tablas es la poblacional de los 5 pliegues (np.std, ddof=0), como en resumir().")

print("\n== Validación cruzada, F1 macro y kappa: recalculadas desde los 5 pliegues ==")
col = ["flexible", "flexible_emg_mlp", "flexible_sin_lstm", None]
malas = []
for tabla, clave, nombre in [(T4A, "macro_f1", "4a"), (T4B, "cohens_kappa", "4b")]:
    n = 0
    for c in C:
        for j, esperado in enumerate(tabla[c]):
            m = col[j] or f"control_{c}"
            tiene = c in pl.get(m, [{}])[0].get("metricas_por_combinacion", {}) if m in pl else False
            if esperado is None:
                if tiene:
                    malas.append(f"Tabla {nombre}, {c}, {m}: hay dato y la tabla dice —")
                continue
            mu, sd = stats(m, c, clave)
            n += 1
            if (round(mu, 3), round(sd, 3)) != esperado:
                malas.append(f"Tabla {nombre}, {c}, {m}: tabla {esperado[0]:.3f} ± {esperado[1]:.3f}, recalculado {mu:.3f} ± {sd:.3f}")
    print(f"       Tabla {nombre}: {n} celdas comparadas")
check("Cada celda de las tablas de validación cruzada coincide con la media ± desviación recalculada de los 5 pliegues",
      "todas coinciden" if not malas else "; ".join(malas), not malas)

print("\n== Presupuesto de entrenamiento ==")
for m in ["flexible", "flexible_emg_mlp", "flexible_sin_lstm"]:
    tras = [len(x["historial"]) - x["mejor_epoca"] for x in pl[m]]
    print(f"       {m:20s} épocas corridas={[len(x['historial']) for x in pl[m]]}, tras la mejor={tras}")
ctr = [m for m in RUNS if m.startswith("control")]
corridas = {m: [len(x["historial"]) for x in pl[m]] for m in ctr}
tras_c = {m: [len(x["historial"]) - x["mejor_epoca"] for x in pl[m]] for m in ctr}
for m in ctr:
    print(f"       {m[:40]:40s} épocas corridas={corridas[m]}, tras la mejor={tras_c[m]}")
topes = {n for m in ctr for n, t in zip(corridas[m], tras_c[m]) if t < 60}
check("Dedicados: hasta 240 épocas y paciencia 60",
      f"todos los pliegues corrieron ≤ 240 épocas: {all(n <= 240 for v in corridas.values() for n in v)}; "
      f"los que pararon antes de 60 épocas sin mejora terminaron en {sorted(topes) or 'ninguno'} (si es 240, llegaron al tope); "
      f"los demás pararon a las 60 sin mejora: {all(t == 60 or n == 240 for m in ctr for n, t in zip(corridas[m], tras_c[m]))}",
      all(n <= 240 for v in corridas.values() for n in v) and topes <= {240}
      and all(t == 60 or n == 240 for m in ctr for n, t in zip(corridas[m], tras_c[m])))
check("Flexibles: paciencia de 5 épocas propias",
      f"todos los pliegues del flexible, el EMG-MLP y la ablación pararon 5 épocas después de su mejor: "
      f"{all(len(x['historial']) - x['mejor_epoca'] == 5 for m in ['flexible', 'flexible_emg_mlp', 'flexible_sin_lstm'] for x in pl[m])}",
      all(len(x["historial"]) - x["mejor_epoca"] == 5 for m in ["flexible", "flexible_emg_mlp", "flexible_sin_lstm"] for x in pl[m]))
print("       (el máximo de épocas del flexible y del EMG-MLP no se puede leer de aquí si ningún pliegue llegó a él)")
media = {m: np.mean([x["mejor_epoca"] for x in pl[m]]) for m in RUNS}
for m in RUNS:
    print(f"       {m[:40]:40s} mejores épocas={[x['mejor_epoca'] for x in pl[m]]} -> media {media[m]:.1f}; final={fin[m]['n_epocas']}")
ctr_rango = (min(round(media[m]) for m in ctr), max(round(media[m]) for m in ctr))
check("Mejor época media: flexible 11 (≈ 66 de dedicado), EMG-MLP 12 (≈ 144), controles entre 72 y 125",
      f"flexible {round(media['flexible'])} (×6 = {6 * round(media['flexible'])}), EMG-MLP {round(media['flexible_emg_mlp'])} "
      f"(×12 = {12 * round(media['flexible_emg_mlp'])}), controles {ctr_rango[0]} a {ctr_rango[1]}",
      round(media["flexible"]) == 11 and round(media["flexible_emg_mlp"]) == 12 and ctr_rango == (72, 125))
iguales = all(fin[m]["n_epocas"] == round(media[m]) == fin[m].get("epocas_cv") and fin[m].get("noche_completa") is True for m in RUNS)
check("Cada modelo final se entrenó con todas las personas de entrenamiento durante su media de mejores épocas, con noches completas",
      f"en los 8: épocas finales = media redondeada de la VC y noche completa: {iguales}; grabaciones={sorted({fin[m]['n_grabaciones'] for m in RUNS})}",
      iguales and {fin[m]["n_grabaciones"] for m in RUNS} == {121})
n_full, n_vent = 0, 0
for f in sorted(Path(args.npz).glob("*_epochs.npz")):
    d = np.load(f, allow_pickle=True)
    if str(d["subject_code"]) in train:
        n = len(d["y"])
        n_full += n
        n_vent += min(n, base.MAX_EPOCHS_PER_RECORDING)
check("Con noches completas el modelo ve ≈ 2.5 veces más épocas por época de entrenamiento que con la ventana de la VC",
      f"noches completas {n_full:,} épocas; ventana de {base.MAX_EPOCHS_PER_RECORDING}: {n_vent:,}; cociente {n_full / n_vent:.2f}",
      abs(n_full / n_vent - 2.5) < 0.15)

print("\n== Criterio de selección ==")
c6 = {m: np.mean([stats(m, c, "macro_f1")[0] for c in COMUNES]) for m in ["flexible", "flexible_emg_mlp"]}
sd6 = {m: np.mean([stats(m, c, "macro_f1")[1] for c in COMUNES]) for m in ["flexible", "flexible_emg_mlp"]}
check("EMG-MLP 0.745 frente a flexible 0.738 en las 6 combinaciones comunes; diferencia 0.007, menor que la desviación entre pliegues (≈ 0.03)",
      f"EMG-MLP {c6['flexible_emg_mlp']:.4f}, flexible {c6['flexible']:.4f}, diferencia {c6['flexible_emg_mlp'] - c6['flexible']:.4f}; "
      f"desviación media entre pliegues: {sd6['flexible_emg_mlp']:.3f} y {sd6['flexible']:.3f}",
      round(c6["flexible_emg_mlp"], 3) == 0.745 and round(c6["flexible"], 3) == 0.738
      and round(c6["flexible_emg_mlp"] - c6["flexible"], 3) == 0.007 and 0.025 <= min(sd6.values()) <= 0.035)

print("\n== Lectura ==")
de = {c: 100 * (stats(f"control_{c}", c, "macro_f1")[0] - stats("flexible_emg_mlp", c, "macro_f1")[0]) for c in CTRL}
df = {c: 100 * (stats(f"control_{c}", c, "macro_f1")[0] - stats("flexible", c, "macro_f1")[0]) for c in CTRL[:4]}
r = lambda d: (round(float(min(v for c, v in d.items() if c != "eeg_pz_oz")), 1), round(float(max(v for c, v in d.items() if c != "eeg_pz_oz")), 1))
check("El dedicado supera al EMG-MLP por 0.4 a 1.0 puntos (3.2 con Pz-Oz) y al flexible sin EMG por 1.1 a 1.5 (4.3 con Pz-Oz)",
      f"EMG-MLP: {r(de)} y {de['eeg_pz_oz']:.1f}; flexible: {r(df)} y {df['eeg_pz_oz']:.1f}",
      r(de) == (0.4, 1.0) and round(de["eeg_pz_oz"], 1) == 3.2 and r(df) == (1.1, 1.5) and round(df["eeg_pz_oz"], 1) == 4.3)
gana = {c: int(sum(a["metricas_por_combinacion"][c]["macro_f1"] > b["metricas_por_combinacion"][c]["macro_f1"]
               for a, b in zip(pl[f"control_{c}"], pl["flexible_emg_mlp"]))) for c in CTRL}
check("Pliegue por pliegue el dedicado gana al EMG-MLP en 22 de 25 (5 de 5 en Pz-Oz y en Fpz-Cz + EOG; 4 de 5 en las demás)",
      f"{gana} -> total {sum(gana.values())}",
      sum(gana.values()) == 22 and gana["eeg_pz_oz"] == 5 and gana["eeg_fpz_cz__eog_horizontal"] == 5)
sube = {m: (bool(stats(m, "eeg_fpz_cz__eeg_pz_oz", "macro_f1")[0] > stats(m, "eeg_fpz_cz", "macro_f1")[0]),
            bool(stats(m, "eeg_fpz_cz__eog_horizontal", "macro_f1")[0] > stats(m, "eeg_fpz_cz", "macro_f1")[0])) for m in ["flexible", "flexible_emg_mlp"]}
check("Añadir EOG o un segundo EEG sube el F1 frente a Fpz-Cz sola", f"(con Pz-Oz, con EOG) por modelo: {sube}", all(all(v) for v in sube.values()))
emg = [100 * (stats("flexible_emg_mlp", c + "__emg_submental", "macro_f1")[0] - stats("flexible_emg_mlp", c, "macro_f1")[0]) for c in COMUNES]
check("El EMG añade poco: 0.1 a 0.5 puntos", f"aporte del EMG por combinación: {[round(float(v), 2) for v in emg]}",
      round(min(emg), 1) >= 0.1 and round(max(emg), 1) <= 0.5)

print(f"\n{'Toda la validación cruzada comprobada.' if fallos == 0 else f'{fallos} afirmación(es) por revisar.'}")
