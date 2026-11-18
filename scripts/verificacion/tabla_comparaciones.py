"""Comparaciones controladas en prueba, con cifras exactas e intervalos.

Todas son pareadas: los dos lados se evalúan sobre las mismas 44,985 épocas de las mismas 16 personas, y el IC95
sale de un bootstrap que remuestrea personas (1,000 remuestreos, semilla 42) y recalcula la diferencia de F1 macro.
  A. Con y sin LSTM: flexible frente a flexible sin LSTM (6 combinaciones).
  B. Modelo flexible frente a cada dedicado: EMG-MLP (5 dedicados) y flexible sin EMG (4 dedicados sin EMG).
  C. Aporte del EOG: EMG-MLP con y sin EOG (6 pares).
  D. Aporte del EMG: EMG-MLP con y sin EMG (6 pares).
Cada grupo usa su propio generador con semilla 42 y recorre los pares en el orden de la tabla (así se reproducen
la tabla de diferencias con los dedicados y la parte C de verificar_emg_y_clase_m.py).

Uso, desde la raíz del repositorio:
    python scripts/verificacion/tabla_comparaciones.py --probs resultados_test --salida reports/tables
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

RUN = {"flexible_emg_mlp": "95951e2a2cfd4368970636cc6bdea9b5", "flexible": "029cc677c3734b10959735a6e9246cb8",
       "flexible_sin_lstm": "c424e04708c34f04a5bfec810b705982",
       "control_eeg_fpz_cz": "2bd024695ad44d19b175490e5abbdedb", "control_eeg_pz_oz": "30877a344b304c5a8a2b5f2e3163805f",
       "control_eeg_fpz_cz__eeg_pz_oz": "076282f5e5534691b441514b50e90441",
       "control_eeg_fpz_cz__eog_horizontal": "038a654fc695444ab650f4e2aac9056f",
       "control_eeg_fpz_cz__eeg_pz_oz__eog_horizontal__emg_submental": "74b16a9d1a854bb5a5589353fec14f91"}
E, P, EO, M = "eeg_fpz_cz", "eeg_pz_oz", "eog_horizontal", "emg_submental"
NOM = {E: "Fpz-Cz", P: "Pz-Oz", f"{E}__{P}": "Fpz-Cz + Pz-Oz", f"{E}__{EO}": "Fpz-Cz + EOG", f"{P}__{EO}": "Pz-Oz + EOG",
       f"{E}__{P}__{EO}": "Fpz-Cz + Pz-Oz + EOG", f"{E}__{M}": "Fpz-Cz + EMG", f"{P}__{M}": "Pz-Oz + EMG",
       f"{E}__{P}__{M}": "Fpz-Cz + Pz-Oz + EMG", f"{E}__{EO}__{M}": "Fpz-Cz + EOG + EMG",
       f"{P}__{EO}__{M}": "Pz-Oz + EOG + EMG", f"{E}__{P}__{EO}__{M}": "Fpz-Cz + Pz-Oz + EOG + EMG"}
BASES = list(NOM)[:6]
CTRL = [E, P, f"{E}__{P}", f"{E}__{EO}", f"{E}__{P}__{EO}__{M}"]
CLASES = ["W", "N1", "N2", "N3", "REM"]

# (grupo, descripción del lado A, del lado B, lista de (etiqueta, (modelo A, combo A), (modelo B, combo B)))
# La diferencia es A menos B.
GRUPOS = [
    ("A. Con LSTM menos sin LSTM (flexible)", [(NOM[c], ("flexible", c), ("flexible_sin_lstm", c)) for c in BASES]),
    ("B1. Dedicado menos EMG-MLP", [(NOM[c], (f"control_{c}", c), ("flexible_emg_mlp", c)) for c in CTRL]),
    ("B2. Dedicado menos flexible sin EMG", [(NOM[c], (f"control_{c}", c), ("flexible", c)) for c in CTRL[:4]]),
    ("C. Con EOG menos sin EOG (EMG-MLP)", [
        (f"{NOM[b]} → + EOG", ("flexible_emg_mlp", c), ("flexible_emg_mlp", b)) for b, c in [
            (E, f"{E}__{EO}"), (P, f"{P}__{EO}"), (f"{E}__{P}", f"{E}__{P}__{EO}"),
            (f"{E}__{M}", f"{E}__{EO}__{M}"), (f"{P}__{M}", f"{P}__{EO}__{M}"), (f"{E}__{P}__{M}", f"{E}__{P}__{EO}__{M}")]]),
    ("D. Con EMG menos sin EMG (EMG-MLP)", [(f"{NOM[c]} → + EMG", ("flexible_emg_mlp", f"{c}__{M}"), ("flexible_emg_mlp", c)) for c in BASES]),
]

ap = argparse.ArgumentParser()
ap.add_argument("--probs", required=True)
ap.add_argument("--salida", default="reports/tables")
ap.add_argument("--n-boot", type=int, default=1000)
args = ap.parse_args()
car, sal = Path(args.probs), Path(args.salida)
sal.mkdir(parents=True, exist_ok=True)
R = json.load(open(car / "resultados_test.json", encoding="utf-8"))
suj_rec = np.array([str(g)[3:5] for g in R["grabaciones"]])
f1m = lambda y, p: f1_score(y, p, average="macro", labels=[0, 1, 2, 3, 4], zero_division=0)
f1p = lambda y, p: f1_score(y, p, average="macro", labels=np.unique(y), zero_division=0)   # por persona, como el 08


def carga(modelo, combo):
    nombre = f"probs_control_{combo}__{combo}.npz" if modelo.startswith("control_") else f"probs_{modelo}__{combo}.npz"
    return np.load(car / nombre)


RES = {}
md, csv = [], ["grupo;comparacion;run_A;run_B;F1_A;F1_B;diferencia_puntos;ic95_bajo;ic95_alto;personas_A_mejor;"
               + ";".join(f"dif_F1_{c}" for c in CLASES)]
for grupo, pares in GRUPOS:
    rng = np.random.default_rng(42)
    md += [f"**{grupo}** (puntos de F1 macro; IC95 pareado por persona)", "",
           "| Comparación | F1 A | F1 B | Diferencia [IC95] | Personas con A mejor | " + " | ".join(f"Δ F1 {c}" for c in CLASES) + " |",
           "|---" * (5 + len(CLASES)) + "|"]
    for etiqueta, (ma, ca), (mb, cb) in pares:
        a, b = carga(ma, ca), carga(mb, cb)
        assert np.array_equal(a["y_true"], b["y_true"]) and np.array_equal(a["rec_idx"], b["rec_idx"]), "no son las mismas épocas"
        y, pa, pb, s = a["y_true"], a["probs"].argmax(1), b["probs"].argmax(1), suj_rec[a["rec_idx"]]
        personas = np.unique(s)
        idx = {q: np.flatnonzero(s == q) for q in personas}
        fa, fb = f1m(y, pa), f1m(y, pb)
        bo = [100 * (f1m(y[sel], pa[sel]) - f1m(y[sel], pb[sel]))
              for sel in (np.concatenate([idx[q] for q in rng.choice(personas, size=len(personas), replace=True)])
                          for _ in range(args.n_boot))]
        lo, hi = np.percentile(bo, [2.5, 97.5])
        gana = int(sum(f1p(y[idx[q]], pa[idx[q]]) > f1p(y[idx[q]], pb[idx[q]]) for q in personas))
        pc = 100 * (f1_score(y, pa, average=None, labels=range(5)) - f1_score(y, pb, average=None, labels=range(5)))
        d = 100 * (fa - fb)
        RES[(grupo[:2].strip(". "), etiqueta)] = dict(d=d, lo=lo, hi=hi, gana=gana, pc=pc, fa=fa, fb=fb)
        clara = "" if lo <= 0 <= hi else " *"
        md.append(f"| {etiqueta} | {fa:.3f} | {fb:.3f} | {d:+.1f} [{lo:+.1f}, {hi:+.1f}]{clara} | {gana} de 16 | "
                  + " | ".join(f"{v:+.1f}" for v in pc) + " |")
        csv.append(";".join([grupo, etiqueta, RUN[ma], RUN[mb], f"{fa:.4f}", f"{fb:.4f}", f"{d:.3f}", f"{lo:.3f}", f"{hi:.3f}", str(gana)]
                            + [f"{v:.3f}" for v in pc]))
    md += ["", "\\* el IC95 no incluye el 0.", ""]
md += ["Runs: " + "; ".join(f"{k} `{v}`" for k, v in RUN.items()) + "."]
(sal / "tabla_comparaciones.md").write_text("\n".join(md) + "\n", encoding="utf-8")
(sal / "tabla_comparaciones.csv").write_text("\n".join(csv) + "\n", encoding="utf-8")
print("\n".join(md))

# ---- comprobaciones contra lo ya verificado ----
fallos = 0
def check(a, o, ok):
    global fallos
    fallos += not ok
    print(f"[{'OK ' if ok else 'NO '}] {a}\n       observado: {o}")
r1 = lambda x: float(f"{x:.1f}")
print()
t5c = {"Fpz-Cz": (1.5, 0.5, 2.4, 13), "Pz-Oz": (0.6, -0.7, 2.0, 8), "Fpz-Cz + Pz-Oz": (1.1, -0.2, 2.5, 8),
       "Fpz-Cz + EOG": (-1.5, -2.6, -0.3, 4), "Fpz-Cz + Pz-Oz + EOG + EMG": (-1.2, -2.8, 0.2, 4)}
obs = {k: (r1(RES[("B1", k)]["d"]), r1(RES[("B1", k)]["lo"]), r1(RES[("B1", k)]["hi"]), RES[("B1", k)]["gana"]) for k in t5c}
check("B1 reproduce la tabla de diferencias con los dedicados", obs, obs == t5c)
emg = {f"{NOM[c]} → + EMG": v for c, v in zip(BASES, [(0.9, 0.2, 1.5), (0.6, -0.6, 1.8), (1.0, 0.3, 1.6), (0.5, 0.1, 0.8), (0.6, 0.2, 1.2), (0.3, -0.1, 0.6)])}
obs = {k: (r1(RES[("D", k)]["d"]), r1(RES[("D", k)]["lo"]), r1(RES[("D", k)]["hi"])) for k in emg}
check("D reproduce la parte C de verificar_emg_y_clase_m.py (aporte del EMG)", obs, obs == emg)
t6 = [3.1, 3.3, 2.9, 2.2, 2.6, 2.0]
obs = [r1(RES[("A", NOM[c])]["d"]) for c in BASES]
check("A reproduce el Δ de prueba de la ablación sin LSTM (en puntos)", obs, obs == t6)
print(f"\nArchivos en {sal}. {'Todo coincide.' if fallos == 0 else f'{fallos} no coinciden: revisar.'}")
