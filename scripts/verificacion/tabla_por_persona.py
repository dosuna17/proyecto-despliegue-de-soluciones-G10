"""F1 macro por persona (las 16 de prueba) con su edad, para comparar jóvenes y mayores.

F1 macro de cada persona con sus dos noches juntas, promediado sobre los estadios que esa persona tiene
(igual que metricas_por_persona del script 08). Modelo: EMG-MLP (run 95951e2a2cfd4368970636cc6bdea9b5).
Edad y sexo: data/raw/SC-subjects-split.xlsx (columnas age y "sex (F=1)"; 2 = hombre).
Jóvenes y mayores se separan por la mediana de edad de las 16 personas; con 16 personas la comparación es
descriptiva: se reporta la correlación de Spearman y la prueba de Mann-Whitney como apoyo, no como confirmación.

Uso, desde la raíz del repositorio:
    python scripts/verificacion/tabla_por_persona.py --probs resultados_test --salida reports/tables
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

RUN = "95951e2a2cfd4368970636cc6bdea9b5"
MODELO = "flexible_emg_mlp"
E, P, EO, M = "eeg_fpz_cz", "eeg_pz_oz", "eog_horizontal", "emg_submental"
NOM = {E: "Fpz-Cz", P: "Pz-Oz", f"{E}__{P}": "Fpz-Cz + Pz-Oz", f"{E}__{EO}": "Fpz-Cz + EOG", f"{P}__{EO}": "Pz-Oz + EOG",
       f"{E}__{P}__{EO}": "Fpz-Cz + Pz-Oz + EOG", f"{E}__{M}": "Fpz-Cz + EMG", f"{P}__{M}": "Pz-Oz + EMG",
       f"{E}__{P}__{M}": "Fpz-Cz + Pz-Oz + EMG", f"{E}__{EO}__{M}": "Fpz-Cz + EOG + EMG",
       f"{P}__{EO}__{M}": "Pz-Oz + EOG + EMG", f"{E}__{P}__{EO}__{M}": "Fpz-Cz + Pz-Oz + EOG + EMG"}
MEJOR = f"{E}__{EO}__{M}"
COLS = [E, MEJOR, f"{E}__{P}__{EO}__{M}"]          # columnas de la tabla por persona
CLASES = ["W", "N1", "N2", "N3", "REM"]

ap = argparse.ArgumentParser()
ap.add_argument("--probs", required=True)
ap.add_argument("--edades", default="data/raw/SC-subjects-split.xlsx")
ap.add_argument("--salida", default="reports/tables")
args = ap.parse_args()
car, sal = Path(args.probs), Path(args.salida)
sal.mkdir(parents=True, exist_ok=True)
R = json.load(open(car / "resultados_test.json", encoding="utf-8"))
suj_rec = np.array([str(g)[3:5] for g in R["grabaciones"]])

x = pd.read_excel(args.edades)
x = x[x["split"] == "test"].groupby("subject").agg(edad=("age", "first"), sexo=("sex (F=1)", "first"))
INFO = {f"{int(s):02d}": (int(r.edad), "M" if int(r.sexo) == 1 else "H") for s, r in x.iterrows()}


def f1_persona(combo):
    d = np.load(car / f"probs_{MODELO}__{combo}.npz")
    y, yp, s = d["y_true"], d["probs"].argmax(1), suj_rec[d["rec_idx"]]
    out, frac = {}, {}
    for q in sorted(set(s)):
        k = s == q
        cm = np.zeros((5, 5))
        np.add.at(cm, (y[k], yp[k]), 1)
        tp = np.diag(cm)
        den = 2 * tp + (cm.sum(0) - tp) + (cm.sum(1) - tp)
        f = np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0)
        out[q] = f[cm.sum(1) > 0].mean()
        frac[q] = (int(k.sum()), cm.sum(1) / k.sum())
    return out, frac


F = {c: f1_persona(c) for c in NOM}
frac = F[E][1]
personas = sorted(INFO, key=lambda q: (INFO[q][0], q))
assert set(personas) == set(F[E][0]), "las personas de prueba no coinciden con las del archivo de edades"
edades = np.array([INFO[q][0] for q in personas])
corte = float(np.median(edades))
jov = edades <= corte

md = [f"**F1 macro por persona de prueba** (EMG-MLP, run `{RUN}`; dos noches por persona; ordenadas por edad)", "",
      "| Persona | Edad | Sexo | Épocas | % N1 | % N3 | " + " | ".join(f"F1 {NOM[c]}" for c in COLS) + " |",
      "|---" * (6 + len(COLS)) + "|"]
csv = ["persona;edad;sexo;epocas;pct_N1;pct_N3;" + ";".join(f"f1_{c}" for c in NOM)]
for q in personas:
    n, fr = frac[q]
    md.append(f"| {q} | {INFO[q][0]} | {INFO[q][1]} | {n:,} | {100 * fr[1]:.1f} | {100 * fr[3]:.1f} | "
              + " | ".join(f"{F[c][0][q]:.3f}" for c in COLS) + " |")
    csv.append(";".join([q, str(INFO[q][0]), INFO[q][1], str(n), f"{100 * fr[1]:.2f}", f"{100 * fr[3]:.2f}"]
                        + [f"{F[c][0][q]:.4f}" for c in NOM]))

md += ["", f"**Jóvenes (≤ {corte:.1f} años, n = {jov.sum()}; edades {edades[jov].min()}–{edades[jov].max()}) frente a "
       f"mayores (> {corte:.1f}, n = {(~jov).sum()}; {edades[~jov].min()}–{edades[~jov].max()})**, F1 macro media ± desviación por persona", "",
       "| Combinación | Jóvenes | Mayores | Diferencia | Mann-Whitney p | Spearman ρ edad–F1 (p) |", "|---|---|---|---|---|---|"]
RES = {}
for c, n in NOM.items():
    f = np.array([F[c][0][q] for q in personas])
    a, b = f[jov], f[~jov]
    rho = spearmanr(edades, f)
    mw = mannwhitneyu(a, b, alternative="two-sided").pvalue
    RES[c] = dict(a=a.mean(), b=b.mean(), rho=rho.correlation, p=rho.pvalue, mw=mw)
    md.append(f"| {n} | {a.mean():.3f} ± {a.std(ddof=1):.3f} | {b.mean():.3f} ± {b.std(ddof=1):.3f} | {a.mean() - b.mean():+.3f} | "
              f"{mw:.3f} | {rho.correlation:.2f} ({rho.pvalue:.3f}) |")
pn1 = np.array([frac[q][1][1] for q in personas]); pn3 = np.array([frac[q][1][3] for q in personas])
md += ["", f"Composición del sueño: N1 es el {100 * pn1[jov].mean():.1f} % de las épocas en los jóvenes y el {100 * pn1[~jov].mean():.1f} % en los mayores; "
       f"N3, el {100 * pn3[jov].mean():.1f} % y el {100 * pn3[~jov].mean():.1f} %."]
(sal / "tabla_por_persona.md").write_text("\n".join(md) + "\n", encoding="utf-8")
(sal / "tabla_por_persona.csv").write_text("\n".join(csv) + "\n", encoding="utf-8")
print("\n".join(md))

fallos = 0
def check(a, o, ok):
    global fallos
    fallos += not ok
    print(f"[{'OK ' if ok else 'NO '}] {a}\n       observado: {o}")
print()
v = np.array([F[MEJOR][0][q] for q in personas])
check("Fpz-Cz + EOG + EMG: 0.758 ± 0.072 por persona, de 0.611 (persona 58) a 0.877 (persona 07), igual que figura_sobreajuste.py",
      f"{v.mean():.3f} ± {v.std(ddof=1):.3f}; mín {v.min():.3f} ({personas[v.argmin()]}), máx {v.max():.3f} ({personas[v.argmax()]})",
      (f"{v.mean():.3f}", f"{v.std(ddof=1):.3f}", f"{v.min():.3f}", f"{v.max():.3f}") == ("0.758", "0.072", "0.611", "0.877"))
neg = [n for c, n in NOM.items() if RES[c]["rho"] < 0 and RES[c]["a"] > RES[c]["b"]]
check("En las 12 combinaciones el F1 baja con la edad (ρ < 0) y los jóvenes quedan por encima de los mayores",
      f"{len(neg)} de 12; ρ de {min(r['rho'] for r in RES.values()):.2f} a {max(r['rho'] for r in RES.values()):.2f}", len(neg) == 12)
print(f"\nArchivos en {sal}. {'Todo coincide.' if fallos == 0 else f'{fallos} no coinciden: revisar.'}")
