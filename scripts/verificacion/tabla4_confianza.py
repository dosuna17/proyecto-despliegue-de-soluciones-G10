"""Pedido 4 de María José: confianza en prueba. Por estadio, cobertura y exactitud de lo aceptado, y el % de épocas
que se marcarían para revisión, con los umbrales ya fijados en la validación cruzada (G8). Modelo: EMG-MLP.

Umbrales por clase (δ = 5 puntos), por estadio PREDICHO: una época se acepta si su confianza (probabilidad máxima)
llega al umbral de su estadio; si no, va a revisión. Cobertura de un estadio = aceptadas / predichas como ese estadio.
Umbral global (secundario): 0.55 para todas las épocas. Los umbrales se leen de resultados_test.json (bloque de
validación) y se recalcula todo desde las predicciones. IC95 por bootstrap de personas (1,000, semilla 42), como el 08.

Uso, desde la raíz del repositorio:
    python scripts/verificacion/tabla4_confianza.py --probs resultados_test --salida reports/tables
"""
import argparse
import json
from pathlib import Path

import numpy as np

RUN = "95951e2a2cfd4368970636cc6bdea9b5"
MODELO = "flexible_emg_mlp"
E, P, EO, M = "eeg_fpz_cz", "eeg_pz_oz", "eog_horizontal", "emg_submental"
NOM = {E: "Fpz-Cz", P: "Pz-Oz", f"{E}__{P}": "Fpz-Cz + Pz-Oz", f"{E}__{EO}": "Fpz-Cz + EOG", f"{P}__{EO}": "Pz-Oz + EOG",
       f"{E}__{P}__{EO}": "Fpz-Cz + Pz-Oz + EOG", f"{E}__{M}": "Fpz-Cz + EMG", f"{P}__{M}": "Pz-Oz + EMG",
       f"{E}__{P}__{M}": "Fpz-Cz + Pz-Oz + EMG", f"{E}__{EO}__{M}": "Fpz-Cz + EOG + EMG",
       f"{P}__{EO}__{M}": "Pz-Oz + EOG + EMG", f"{E}__{P}__{EO}__{M}": "Fpz-Cz + Pz-Oz + EOG + EMG"}
CLASES = ["W", "N1", "N2", "N3", "REM"]

ap = argparse.ArgumentParser()
ap.add_argument("--probs", required=True)
ap.add_argument("--salida", default="reports/tables")
args = ap.parse_args()
car, sal = Path(args.probs), Path(args.salida)
sal.mkdir(parents=True, exist_ok=True)
R = json.load(open(car / "resultados_test.json", encoding="utf-8"))
J = R["modelos"][MODELO]
bloque = J["umbrales_validacion"]["por_clase"]["global"]["0.05"]
TAU = {c: bloque["clases"][c]["umbral"] for c in CLASES}
META = {c: bloque["clases"][c]["meta_exactitud"] for c in CLASES}
VAL = {c: bloque["clases"][c]["validacion"] for c in CLASES}
T_GLOBAL = J["umbrales_validacion"]["global"]["umbral"]
suj_rec = np.array([str(g)[3:5] for g in R["grabaciones"]])


def resumen(cnt, idx):
    npred, nacc, nok = cnt
    b = lambda v: v[idx].sum(1)
    cob_b, ex_b = b(nacc) / np.maximum(b(npred), 1), b(nok) / np.maximum(b(nacc), 1)
    return dict(cob=nacc.sum() / npred.sum(), ex=nok.sum() / nacc.sum() if nacc.sum() else np.nan,
                cob_ic=np.percentile(cob_b, [2.5, 97.5]), ex_ic=np.nanpercentile(ex_b, [2.5, 97.5]),
                n_pred=int(npred.sum()), n_acc=int(nacc.sum()))


def evaluar(probs, y, s, n_boot=1000, semilla=42):
    conf, yp = probs.max(1), probs.argmax(1)
    personas = np.unique(s)
    idx = np.random.default_rng(semilla).integers(0, len(personas), size=(n_boot, len(personas)))
    out, tot = {}, np.zeros((3, len(personas)))
    for k, c in enumerate(CLASES):
        es = yp == k
        acep = es & (conf >= TAU[c])
        cnt = np.array([[(es & (s == q)).sum(), (acep & (s == q)).sum(), (acep & (y == k) & (s == q)).sum()] for q in personas], float).T
        tot += cnt
        out[c] = resumen(cnt, idx)
    out["total"] = resumen(tot, idx)
    ok = yp == y
    acep = conf >= T_GLOBAL
    cnt = np.array([[(s == q).sum(), (acep & (s == q)).sum(), (acep & ok & (s == q)).sum()] for q in personas], float).T
    out["global"] = resumen(cnt, idx)
    out["sin_umbral"] = ok.mean()
    return out


P_, Y_, S_, por_comb = [], [], [], {}
for c in NOM:
    d = np.load(car / f"probs_{MODELO}__{c}.npz")
    s = suj_rec[d["rec_idx"]]
    por_comb[c] = evaluar(d["probs"], d["y_true"], s)
    P_.append(d["probs"]); Y_.append(d["y_true"]); S_.append(s)
junto = evaluar(np.concatenate(P_), np.concatenate(Y_), np.concatenate(S_))

md = [f"**Tabla 4a. Confianza por estadio en prueba** (EMG-MLP, run `{RUN}`; las 12 combinaciones juntas, 539,820 épocas; "
      f"umbrales por clase fijados en la validación cruzada con δ = 5 puntos; IC95 por persona)", "",
      "| Estadio predicho | Umbral | Meta | Validación: cobertura / exactitud | Prueba: cobertura [IC95] | Prueba: exactitud de lo aceptado [IC95] | ¿Cumple la meta? |",
      "|---|---|---|---|---|---|---|"]
csv = ["tabla;fila;umbral;meta;val_cobertura;val_exactitud;cobertura;cob_ic_bajo;cob_ic_alto;exactitud;ex_ic_bajo;ex_ic_alto;pct_revision"]
for c in CLASES:
    r, v = junto[c], VAL[c]
    cumple = ("sí" if r["ex_ic"][0] >= META[c] else "solo en el valor puntual" if r["ex"] >= META[c] else
              "no concluyente (meta dentro del IC)" if r["ex_ic"][1] >= META[c] else "no")
    md.append(f"| {c} | {TAU[c]:.2f} | {META[c]:.3f} | {v['cobertura']:.3f} / {v['exactitud_aceptadas']:.3f} | "
              f"{r['cob']:.3f} [{r['cob_ic'][0]:.3f}, {r['cob_ic'][1]:.3f}] | {r['ex']:.3f} [{r['ex_ic'][0]:.3f}, {r['ex_ic'][1]:.3f}] | {cumple} |")
    csv.append(";".join(["4a", c, f"{TAU[c]:.2f}", f"{META[c]:.3f}", f"{v['cobertura']:.4f}", f"{v['exactitud_aceptadas']:.4f}"]
                        + [f"{x:.4f}" for x in (r["cob"], *r["cob_ic"], r["ex"], *r["ex_ic"])] + [f"{100 * (1 - r['cob']):.2f}"]))
t, g = junto["total"], junto["global"]
md += [f"| Todas (umbrales por clase) | — | — | {bloque['validacion_global']['cobertura']:.3f} / {bloque['validacion_global']['exactitud_aceptadas']:.3f} | "
       f"{t['cob']:.3f} [{t['cob_ic'][0]:.3f}, {t['cob_ic'][1]:.3f}] | {t['ex']:.3f} [{t['ex_ic'][0]:.3f}, {t['ex_ic'][1]:.3f}] | — |",
       f"| Todas (umbral global {T_GLOBAL:.2f}) | {T_GLOBAL:.2f} | 0.826 | {J['umbrales_validacion']['global']['validacion']['cobertura']:.3f} / "
       f"{J['umbrales_validacion']['global']['validacion']['exactitud']:.3f} | {g['cob']:.3f} [{g['cob_ic'][0]:.3f}, {g['cob_ic'][1]:.3f}] | "
       f"{g['ex']:.3f} [{g['ex_ic'][0]:.3f}, {g['ex_ic'][1]:.3f}] | {'sí' if g['ex_ic'][0] >= 0.826 else 'solo en el valor puntual' if g['ex'] >= 0.826 else 'no'} |",
       "", f"Épocas para revisión: **{100 * (1 - t['cob']):.1f} %** con los umbrales por clase y **{100 * (1 - g['cob']):.1f} %** con el umbral global "
       f"(sin umbral, la exactitud de todas las épocas es {junto['sin_umbral']:.3f}).", "",
       "**Tabla 4b. Por combinación** (mismos umbrales aplicados a cada combinación sola)", "",
       "| Combinación | Revisión, umbrales por clase | Exactitud de lo aceptado | Revisión, umbral global | Exactitud de lo aceptado | Exactitud sin umbral |",
       "|---|---|---|---|---|---|"]
for c, n in NOM.items():
    r = por_comb[c]
    md.append(f"| {n} | {100 * (1 - r['total']['cob']):.1f} % | {r['total']['ex']:.3f} | {100 * (1 - r['global']['cob']):.1f} % | "
              f"{r['global']['ex']:.3f} | {r['sin_umbral']:.3f} |")
    csv.append(";".join(["4b", n, "", "", "", "", f"{r['total']['cob']:.4f}", "", "", f"{r['total']['ex']:.4f}", "", "",
                         f"{100 * (1 - r['total']['cob']):.2f}"]))
(sal / "tabla4_confianza.md").write_text("\n".join(md) + "\n", encoding="utf-8")
(sal / "tabla4_confianza.csv").write_text("\n".join(csv) + "\n", encoding="utf-8")
print("\n".join(md))

# ---- comprobaciones contra G8 y contra lo que guardó el 08 ----
fallos = 0
def check(a, o, ok):
    global fallos
    fallos += not ok
    print(f"[{'OK ' if ok else 'NO '}] {a}\n       observado: {o}")
f3 = lambda *v: tuple(float(f"{x:.3f}") for x in v)
print()
jt = J["umbral_global_junto"]["por_clase"]["test"]
check("Las 12 combinaciones juntas: cobertura y exactitud por estadio (con IC) iguales a las que guardó el 08",
      {c: f3(junto[c]["cob"], junto[c]["ex"], *junto[c]["ex_ic"]) for c in CLASES},
      all(f3(junto[c]["cob"], junto[c]["ex"], *junto[c]["ex_ic"]) == f3(jt["clases"][c]["cobertura"], jt["clases"][c]["exactitud_aceptadas"],
                                                                         *jt["clases"][c]["exactitud_ic95"]) for c in CLASES))
check("Revisión: 11.9 % con los umbrales por clase y 9.2 % con el umbral global (G8)",
      f"{100 * (1 - t['cob']):.1f} % y {100 * (1 - g['cob']):.1f} %",
      (f"{100 * (1 - t['cob']):.1f}", f"{100 * (1 - g['cob']):.1f}") == ("11.9", "9.2"))
dif = [c for c in NOM if f3(por_comb[c]["total"]["cob"], por_comb[c]["total"]["ex"]) !=
       f3(J["combinaciones"][c]["umbral_por_clase"]["global"]["test"]["global"]["cobertura"],
          J["combinaciones"][c]["umbral_por_clase"]["global"]["test"]["global"]["exactitud_aceptadas"])]
check("Por combinación: cobertura y exactitud iguales a las que guardó el 08", dif or "las 12 iguales", not dif)
print(f"\nArchivos en {sal}. {'Todo coincide.' if fallos == 0 else f'{fallos} no coinciden: revisar.'}")
