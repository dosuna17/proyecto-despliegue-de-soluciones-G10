"""Tabla de prueba por combinación (EMG-MLP, 12 combinaciones) y la misma para el flexible
sin EMG (6), cada una con la fila de todas sus combinaciones juntas. Fuente: predicciones guardadas del único test.

Columnas: exactitud, F1 macro [IC95], exactitud balanceada, kappa y F1 por estadio. IC95 por bootstrap por persona
(1,000 remuestreos, semilla 42, como scripts/08_evaluar_test.py). En la fila "las N combinaciones" se juntan las
épocas de todas las combinaciones del modelo, y en el bootstrap cada persona lleva sus épocas de todas ellas.

Uso, desde la raíz del repositorio:
    python scripts/verificacion/tabla_test_por_combinacion.py --probs resultados_test --salida reports/tables
Escribe tabla_test_<modelo>.csv y .md, y comprueba las filas contra los valores de prueba reportados en el artículo.
"""
import argparse
from pathlib import Path
import json

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score

RUN = {"flexible_emg_mlp": "95951e2a2cfd4368970636cc6bdea9b5", "flexible": "029cc677c3734b10959735a6e9246cb8"}
E, P, EO, M = "eeg_fpz_cz", "eeg_pz_oz", "eog_horizontal", "emg_submental"
NOM = {E: "Fpz-Cz", P: "Pz-Oz", f"{E}__{P}": "Fpz-Cz + Pz-Oz", f"{E}__{EO}": "Fpz-Cz + EOG", f"{P}__{EO}": "Pz-Oz + EOG",
       f"{E}__{P}__{EO}": "Fpz-Cz + Pz-Oz + EOG", f"{E}__{M}": "Fpz-Cz + EMG", f"{P}__{M}": "Pz-Oz + EMG",
       f"{E}__{P}__{M}": "Fpz-Cz + Pz-Oz + EMG", f"{E}__{EO}__{M}": "Fpz-Cz + EOG + EMG",
       f"{P}__{EO}__{M}": "Pz-Oz + EOG + EMG", f"{E}__{P}__{EO}__{M}": "Fpz-Cz + Pz-Oz + EOG + EMG"}
COMBOS = {"flexible_emg_mlp": list(NOM), "flexible": list(NOM)[:6]}
CLASES = ["W", "N1", "N2", "N3", "REM"]
# Valores de prueba reportados en el artículo (EMG-MLP): F1 macro, kappa y exactitud, para comprobar
T5 = {E: (.749, .726, .799), P: (.714, .691, .776), f"{E}__{P}": (.756, .735, .806), f"{E}__{EO}": (.779, .758, .823),
      f"{P}__{EO}": (.748, .726, .800), f"{E}__{P}__{EO}": (.781, .761, .824), f"{E}__{M}": (.758, .734, .806),
      f"{P}__{M}": (.720, .699, .782), f"{E}__{P}__{M}": (.766, .745, .814), f"{E}__{EO}__{M}": (.784, .762, .826),
      f"{P}__{EO}__{M}": (.755, .731, .804), f"{E}__{P}__{EO}__{M}": (.784, .764, .827)}

ap = argparse.ArgumentParser()
ap.add_argument("--probs", required=True)
ap.add_argument("--salida", default="reports/tables")
args = ap.parse_args()
car, sal = Path(args.probs), Path(args.salida)
sal.mkdir(parents=True, exist_ok=True)
R = json.load(open(car / "resultados_test.json", encoding="utf-8"))
suj_rec = np.array([str(g)[3:5] for g in R["grabaciones"]])


def ic_f1(y, yp, s, n_boot=1000, semilla=42):
    personas = np.unique(s)
    cms = np.zeros((len(personas), 5, 5))
    for i, q in enumerate(personas):
        m = s == q
        np.add.at(cms[i], (y[m], yp[m]), 1)
    idx = np.random.default_rng(semilla).integers(0, len(personas), size=(n_boot, len(personas)))
    cm = cms[idx].sum(1)
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    den = 2 * tp + (cm.sum(-2) - tp) + (cm.sum(-1) - tp)
    f1 = np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0).mean(-1)
    return np.percentile(f1, [2.5, 97.5])


def fila(y, yp, s):
    f1c = f1_score(y, yp, labels=range(5), average=None, zero_division=0)
    lo, hi = ic_f1(y, yp, s)
    return dict(n=len(y), acc=accuracy_score(y, yp), f1=f1c.mean(), lo=lo, hi=hi,
                bal=balanced_accuracy_score(y, yp), kappa=cohen_kappa_score(y, yp), clase=f1c)


fallos = 0
for modelo, combos in COMBOS.items():
    filas, Y, YP, S = [], [], [], []
    for c in combos:
        d = np.load(car / f"probs_{modelo}__{c}.npz")
        y, yp, s = d["y_true"], d["probs"].argmax(1), suj_rec[d["rec_idx"]]
        filas.append((NOM[c], fila(y, yp, s)))
        Y.append(y); YP.append(yp); S.append(s)
    filas.append((f"Las {len(combos)} combinaciones juntas", fila(np.concatenate(Y), np.concatenate(YP), np.concatenate(S))))
    if len(combos) == 12:   # para comparar con el flexible sobre las mismas 6 combinaciones
        filas.append(("Las 6 combinaciones sin EMG juntas", fila(np.concatenate(Y[:6]), np.concatenate(YP[:6]), np.concatenate(S[:6]))))
    enc = ["Combinación", "Épocas", "Exactitud", "F1 macro", "IC95 bajo", "IC95 alto", "Exact. balanceada", "κ"] + [f"F1 {c}" for c in CLASES]
    csv = [";".join(enc)]
    md = [f"**Prueba, {modelo}** (run MLflow `{RUN[modelo]}`; 16 personas; IC95 bootstrap por persona)", "",
          "| Combinación | Exactitud | F1 macro [IC95] | Exact. balanceada | κ | " + " | ".join(f"F1 {c}" for c in CLASES) + " |",
          "|---" * (5 + len(CLASES)) + "|"]
    for nombre, r in filas:
        csv.append(";".join([nombre, str(r["n"])] + [f"{v:.4f}" for v in (r["acc"], r["f1"], r["lo"], r["hi"], r["bal"], r["kappa"], *r["clase"])]))
        md.append(f"| {nombre} | {r['acc']:.3f} | {r['f1']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}] | {r['bal']:.3f} | {r['kappa']:.3f} | "
                  + " | ".join(f"{v:.3f}" for v in r["clase"]) + " |")
    (sal / f"tabla_test_{modelo}.csv").write_text("\n".join(csv) + "\n", encoding="utf-8")
    (sal / f"tabla_test_{modelo}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md), "\n")
    if modelo == "flexible_emg_mlp":
        malos = [n for (n, r), c in zip(filas, combos) if tuple(float(f"{v:.3f}") for v in (r["f1"], r["kappa"], r["acc"])) != T5[c]]
        print(f"[{'OK ' if not malos else 'NO '}] Las 12 filas coinciden con los valores reportados (F1, κ, exactitud)"
              + (f": difieren {malos}" if malos else ""))
        fallos += bool(malos)
print(f"\nArchivos en {sal}. {'Todo coincide.' if not fallos else 'Revisar.'}")
