"""Figura y verificación de los umbrales (cobertura vs exactitud y umbrales por clase) del EMG-MLP.

Recalcula todo de forma independiente, sin cargar modelos ni volver a evaluar:
  Parte A (validación): con las probabilidades de validación de los 5 pliegues y las 12 combinaciones
          (probs_validacion/flexible_emg_mlp_pliegueK__COMBO.npz, las mismas que usó el script 08 ANTES del test),
          vuelve a fijar el umbral global de 82.6 % y los umbrales por clase para delta 0, 5 y 10 puntos,
          y la elección de delta = 5.
  Parte B (prueba): aplica esos umbrales a las predicciones del único test (las 12 combinaciones juntas)
          y recalcula cobertura, exactitud de lo aceptado e IC95 por persona.
  Compara todo con lo guardado en resultados_test.json y con lo que dice el texto.

Uso, desde la raíz del repositorio:
    python scripts/verificacion/figura_umbrales.py --probs resultados_test ^
           --val-probs res_emg/probs_validacion
Salida (por defecto en la carpeta del script; usar --salida reports/figures): figura_umbrales.png (300 dpi), .pdf y una copia en gris.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODELO = "flexible_emg_mlp"
E, P, EO, M = "eeg_fpz_cz", "eeg_pz_oz", "eog_horizontal", "emg_submental"
COMBOS = [E, P, f"{E}__{P}", f"{E}__{EO}", f"{P}__{EO}", f"{E}__{P}__{EO}", f"{E}__{M}", f"{P}__{M}",
          f"{E}__{P}__{M}", f"{E}__{EO}__{M}", f"{P}__{EO}__{M}", f"{E}__{P}__{EO}__{M}"]
CLASES = ["W", "N1", "N2", "N3", "REM"]
UMBRALES = np.round(np.arange(0.0, 1.0, 0.01), 2)
OBJETIVO = 0.826
REF_SLEEPFM = {"W": 0.875, "N1": 0.560, "N2": 0.808, "N3": 0.862, "REM": 0.937}   # sensibilidad, SleepFM Fig. 1d
DELTAS = (0.0, 0.05, 0.10)
COLOR_CLASE = {"W": "#2a78d6", "N1": "#eb6834", "N2": "#1baf7a", "N3": "#eda100", "REM": "#e87ba4"}   # tablero
C_VAL, C_TEST, TINTA, TINTA2, REJILLA = "#5598e7", "#104281", "#0b0b0b", "#52514e", "#e6e5e0"

p = argparse.ArgumentParser()
p.add_argument("--probs", required=True, help="carpeta con resultados_test.json y los probs_*.npz del test")
p.add_argument("--val-probs", required=True, help="carpeta con flexible_emg_mlp_pliegueK__COMBO.npz (validación)")
p.add_argument("--salida", default=None)
args = p.parse_args()
sal = Path(args.salida) if args.salida else Path(__file__).resolve().parent

fallos = 0
def check(afirmacion, observado, ok):
    global fallos
    fallos += not ok
    print(f"[{'OK ' if ok else 'NO '}] {afirmacion}\n       observado: {observado}")

f3 = lambda *v: tuple(None if x is None else float(f"{x:.3f}") for x in v)

R = json.load(open(Path(args.probs) / "resultados_test.json", encoding="utf-8"))
J = R["modelos"][MODELO]

# ---------------- Carga ----------------
def cargar_val():
    pv, yv, faltan = [], [], []
    for c in COMBOS:
        for k in range(5):
            f = Path(args.val_probs) / f"{MODELO}_pliegue{k}__{c}.npz"
            if not f.exists():
                hall = list(Path(args.val_probs).rglob(f.name))
                if not hall:
                    faltan.append(f.name)
                    continue
                f = hall[0]
            d = np.load(f)
            pv.append(d["probs"])
            yv.append(d["y_true"])
    return pv, yv, faltan

pv, yv, faltan = cargar_val()
if faltan:
    raise SystemExit(f"Faltan {len(faltan)} archivos de validación, por ejemplo {faltan[:3]}")
PV, YV = np.concatenate(pv), np.concatenate(yv)
suj_rec = np.array([str(g)[3:5] for g in R["grabaciones"]])
pt, yt, st = [], [], []
for c in COMBOS:
    d = np.load(Path(args.probs) / f"probs_{MODELO}__{c}.npz")
    pt.append(d["probs"]); yt.append(d["y_true"]); st.append(suj_rec[d["rec_idx"]])
PT, YT, ST = np.concatenate(pt), np.concatenate(yt), np.concatenate(st)

# ---------------- Reglas (reimplementadas aquí) ----------------
def umbral_global(probs, y):
    conf, ok = probs.max(1), probs.argmax(1) == y
    for t in UMBRALES:
        m = conf >= t
        if m.any() and ok[m].mean() >= OBJETIVO:
            return float(t), m.mean(), ok[m].mean()
    return None, None, None

def umbrales_clase(probs, y, delta):
    conf, yp = probs.max(1), probs.argmax(1)
    out, acc, okt = {}, 0, 0
    for k, c in enumerate(CLASES):
        meta = round(REF_SLEEPFM[c] - delta, 4)
        es = yp == k
        tau = None
        for t in UMBRALES:
            m = es & (conf >= t)
            if m.any() and (y[m] == k).mean() >= meta:
                tau = float(t)
                break
        if tau is not None:
            m = es & (conf >= tau)
            out[c] = dict(meta=meta, umbral=tau, cobertura=m.sum() / es.sum(), exactitud=(y[m] == k).mean())
            acc += m.sum(); okt += (y[m] == k).sum()
        else:
            out[c] = dict(meta=meta, umbral=None, cobertura=0.0, exactitud=None)
    return out, acc / len(y), okt / acc

def aplicar_clase(probs, y, suj, taus, n_boot=1000, semilla=42):
    """Como evaluar_umbrales_clase de 08: por clase predicha, IC95 remuestreando personas."""
    conf, yp = probs.max(1), probs.argmax(1)
    personas = np.unique(suj)
    idx = np.random.default_rng(semilla).integers(0, len(personas), size=(n_boot, len(personas)))
    res, tot = {}, np.zeros((3, len(personas)))
    for k, c in enumerate(CLASES):
        es = yp == k
        acep = es & (conf >= taus[c]) if taus[c] is not None else np.zeros_like(es)
        cnt = np.array([[(es & (suj == s)).sum(), (acep & (suj == s)).sum(), (acep & (y == k) & (suj == s)).sum()]
                        for s in personas], dtype=float).T
        tot += cnt
        res[c] = resumen(cnt, idx)
    res["global"] = resumen(tot, idx)
    return res

def resumen(cnt, idx):
    npred, nacc, nok = cnt
    b = lambda v: v[idx].sum(1)
    cob_b, ex_b = b(nacc) / b(npred), b(nok) / np.maximum(b(nacc), 1)
    return dict(cobertura=nacc.sum() / npred.sum(), exactitud=nok.sum() / nacc.sum(),
                cob_ic=np.percentile(cob_b, [2.5, 97.5]), ex_ic=np.nanpercentile(ex_b, [2.5, 97.5]),
                n_pred=int(npred.sum()), n_acc=int(nacc.sum()))

def aplicar_global(probs, y, suj, t, n_boot=1000, semilla=42):
    conf, ok = probs.max(1), probs.argmax(1) == y
    personas = np.unique(suj)
    idx = np.random.default_rng(semilla).integers(0, len(personas), size=(n_boot, len(personas)))
    cnt = np.array([[(suj == s).sum(), ((conf >= t) & (suj == s)).sum(), ((conf >= t) & ok & (suj == s)).sum()]
                    for s in personas], dtype=float).T
    return resumen(cnt, idx)

# ---------------- Parte A: validación ----------------
print("== Parte A. Umbrales fijados con la validación cruzada (antes del test) ==")
check("Validación: 5 pliegues x 12 combinaciones = 60 archivos, 59,895 épocas por combinación (718,740 en total)",
      f"{len(pv)} archivos; {len(YV):,} épocas", len(pv) == 60 and len(YV) == 718740)
tg, cg, eg = umbral_global(PV, YV)
jg = J["umbrales_validacion"]["global"]
check("Umbral global (exactitud de lo aceptado >= 82.6 %): 0.55, con cobertura 90.8 % y exactitud 0.828 en validación",
      f"umbral {tg}, cobertura {cg:.4f}, exactitud {eg:.4f} | JSON: {jg['umbral']}, {jg['validacion']['cobertura']:.4f}, "
      f"{jg['validacion']['exactitud']:.4f}",
      tg == 0.55 == jg["umbral"] and f3(cg, eg) == f3(jg["validacion"]["cobertura"], jg["validacion"]["exactitud"]))
UC = {d: umbrales_clase(PV, YV, d) for d in DELTAS}
for d in DELTAS:
    u, cob, ex = UC[d]
    jb = J["umbrales_validacion"]["por_clase"]["global"][f"{d:.2f}"]
    igual = all(u[c]["umbral"] == jb["clases"][c]["umbral"] for c in CLASES) and \
        f3(cob, ex) == f3(jb["validacion_global"]["cobertura"], jb["validacion_global"]["exactitud_aceptadas"])
    check(f"Delta {d * 100:.0f} puntos: umbrales por clase y exactitud global de lo aceptado iguales a los del JSON",
          " ".join(f"{c} {u[c]['umbral']}" for c in CLASES) + f" | cobertura {cob:.3f}, exactitud {ex:.3f}", igual)
u5 = UC[0.05][0]
check("Con delta 5: W 0.00, N1 0.65, N2 0.00, N3 0.66, REM 0.75 (meta = sensibilidad de SleepFM menos 5 puntos)",
      " ".join(f"{c} {u5[c]['umbral']:.2f} (meta {u5[c]['meta']:.3f})" for c in CLASES),
      [u5[c]["umbral"] for c in CLASES] == [0.0, 0.65, 0.0, 0.66, 0.75])
ex_d = {d: UC[d][2] for d in DELTAS}
cob_d = {d: UC[d][1] for d in DELTAS}
elegido = max(d for d in DELTAS if ex_d[d] >= OBJETIVO)
check("Delta 5 es el mayor de {0, 5, 10} con exactitud global de lo aceptado >= 82.6 % en validación "
      "(0.843; con 10 baja a 0.814); con 0 la cobertura cae de 0.879 a 0.805",
      f"exactitud {', '.join(f'{d * 100:.0f}: {ex_d[d]:.3f}' for d in DELTAS)}; cobertura "
      f"{', '.join(f'{d * 100:.0f}: {cob_d[d]:.3f}' for d in DELTAS)}; elegido {elegido}",
      elegido == 0.05 and f3(ex_d[0.05], ex_d[0.10], cob_d[0.05], cob_d[0.0]) == (0.843, 0.814, 0.879, 0.805))

# ---------------- Parte B: prueba ----------------
print("\n== Parte B. Verificación en prueba (12 combinaciones juntas, umbrales ya fijados) ==")
check("Prueba: 12 combinaciones x 44,985 épocas = 539,820; 16 personas", f"{len(YT):,} épocas; {len(np.unique(ST))} personas",
      len(YT) == 539820 and len(np.unique(ST)) == 16)
taus = {c: u5[c]["umbral"] for c in CLASES}
TB = aplicar_clase(PT, YT, ST, taus)
jt = J["umbral_global_junto"]["por_clase"]["test"]
for c in CLASES:
    r, j = TB[c], jt["clases"][c]
    igual = f3(r["cobertura"], r["exactitud"], *r["ex_ic"]) == f3(j["cobertura"], j["exactitud_aceptadas"], *j["exactitud_ic95"])
    cumple = r["exactitud"] >= u5[c]["meta"]
    check(f"{c}: exactitud de lo aceptado en prueba frente a la meta {u5[c]['meta']:.3f} (igual al JSON)",
          f"{r['exactitud']:.3f} [{r['ex_ic'][0]:.3f}, {r['ex_ic'][1]:.3f}], cobertura {r['cobertura']:.3f} | "
          f"{'cumple' if cumple else 'NO cumple'} en el valor puntual; meta dentro del IC: {r['ex_ic'][0] <= u5[c]['meta'] <= r['ex_ic'][1]}",
          igual)
cumplen = [c for c in CLASES if TB[c]["exactitud"] >= u5[c]["meta"]]
n3 = TB["N3"]
check("Cumplen la meta W, N1, N2 y REM; N3 queda en 0.752 [0.653, 0.839] frente a 0.812, dentro del intervalo: no concluyente",
      f"cumplen {cumplen}; N3 {n3['exactitud']:.3f} [{n3['ex_ic'][0]:.3f}, {n3['ex_ic'][1]:.3f}]",
      cumplen == ["W", "N1", "N2", "REM"] and f3(n3["exactitud"], *n3["ex_ic"]) == (0.752, 0.653, 0.839)
      and n3["ex_ic"][0] <= 0.812 <= n3["ex_ic"][1])
bajo = [c for c in cumplen if TB[c]["ex_ic"][0] < u5[c]["meta"]]
check("Matiz: en N1 y REM el límite inferior del IC queda por debajo de la meta (N1 0.492 frente a 0.510; REM 0.886 frente a 0.887)",
      ", ".join(f"{c} {TB[c]['ex_ic'][0]:.3f} frente a {u5[c]['meta']:.3f}" for c in bajo),
      bajo == ["N1", "REM"] and f3(TB["N1"]["ex_ic"][0], TB["REM"]["ex_ic"][0]) == (0.492, 0.886))
g = TB["global"]
check("Con los umbrales por clase, lo aceptado tiene exactitud 0.861 con 88.1 % de cobertura (el 11.9 % va a revisión)",
      f"exactitud {g['exactitud']:.4f} [{g['ex_ic'][0]:.3f}, {g['ex_ic'][1]:.3f}], cobertura {g['cobertura']:.4f} "
      f"[{g['cob_ic'][0]:.3f}, {g['cob_ic'][1]:.3f}]",
      f3(g["exactitud"], g["cobertura"]) == (0.861, 0.881))
G = aplicar_global(PT, YT, ST, 0.55)
jg_t = J["umbral_global_junto"]["umbral_global_82_6"]["test"]
check("Umbral global 0.55 en prueba: cobertura 90.8 % y exactitud 0.843 [0.816, 0.872] (igual al JSON)",
      f"cobertura {G['cobertura']:.4f}, exactitud {G['exactitud']:.4f} [{G['ex_ic'][0]:.3f}, {G['ex_ic'][1]:.3f}]",
      f3(G["cobertura"], G["exactitud"], *G["ex_ic"]) == f3(jg_t["cobertura"], jg_t["exactitud_aceptadas"], *jg_t["exactitud_ic95"])
      == (0.908, 0.843, 0.816, 0.872))
ok_t = PT.argmax(1) == YT
check("Sin umbral, la exactitud de todas las épocas es 0.807; el umbral 0.55 la sube a 0.843 dejando fuera el 9.2 %",
      f"exactitud sin umbral {ok_t.mean():.4f}", f3(ok_t.mean()) == (0.807,))

# ---------------- Figura ----------------
def curva(probs, y):
    conf, ok = probs.max(1), probs.argmax(1) == y
    orden = np.argsort(-conf)
    acum = np.cumsum(ok[orden]) / np.arange(1, len(ok) + 1)
    cob = np.arange(1, len(ok) + 1) / len(ok)
    sel = np.unique(np.linspace(len(ok) * 0.3, len(ok) - 1, 400).astype(int))
    return cob[sel] * 100, acum[sel] * 100

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": TINTA2,
                     "xtick.color": TINTA2, "ytick.color": TINTA2})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True, gridspec_kw={"width_ratios": [1, 1.15]})
for (pp_, yy, col, ls, lab, t) in [(PV, YV, C_VAL, "--", "Validación cruzada", (cg, eg)),
                                    (PT, YT, C_TEST, "-", "Prueba", (G["cobertura"], G["exactitud"]))]:
    x, yv_ = curva(pp_, yy)
    a1.plot(x, yv_, color=col, ls=ls, lw=1.4, label=lab)
    a1.plot(t[0] * 100, t[1] * 100, "o", ms=5, mfc="white", mec=col, mew=1.4, zorder=5)
a1.axhline(OBJETIVO * 100, color=TINTA2, lw=.8, ls=":")
a1.text(62, OBJETIVO * 100 - .4, "82.6 % (acuerdo entre expertos)", fontsize=8, color=TINTA2, va="top")
a1.annotate("umbral\n0.55", (G["cobertura"] * 100, G["exactitud"] * 100), xytext=(99, 88.5), fontsize=8, color=TINTA,
            arrowprops=dict(arrowstyle="-", color=TINTA2, lw=.7))
a1.invert_xaxis()
a1.set_xlabel("Épocas aceptadas (%)")
a1.set_ylabel("Exactitud de las aceptadas (%)")
a1.set_title("(a) Cobertura frente a exactitud", loc="left", fontsize=9)
a1.legend(fontsize=8, frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.2))
a1.grid(color=REJILLA, lw=.6)
xs = np.arange(len(CLASES))
uv5 = UC[0.05][0]
for i, c in enumerate(CLASES):
    r = TB[c]
    a2.plot([i - .3, i + .3], [u5[c]["meta"] * 100] * 2, color=TINTA, lw=1.4)
    a2.errorbar(i + .08, r["exactitud"] * 100, yerr=[[100 * (r["exactitud"] - r["ex_ic"][0])], [100 * (r["ex_ic"][1] - r["exactitud"])]],
                fmt="o", ms=5.5, color=COLOR_CLASE[c], mec=TINTA, mew=.7, ecolor=TINTA2, elinewidth=1, capsize=2.5, zorder=4)
    a2.plot(i - .12, uv5[c]["exactitud"] * 100, "D", ms=4, mfc="white", mec=TINTA2, mew=1, zorder=4)
    a2.text(i, 26, f"τ {u5[c]['umbral']:.2f}\n{r['cobertura'] * 100:.0f} %", ha="center", va="bottom", fontsize=8, color=TINTA2)
a2.set_xticks(xs, CLASES)
a2.set_xlim(-.5, len(CLASES) - .5)
a2.set_ylim(25, 100)
a2.set_ylabel("Exactitud de lo aceptado (%)")
a2.set_title("(b) Umbrales por clase", loc="left", fontsize=9)
a2.grid(axis="y", color=REJILLA, lw=.6)
from matplotlib.lines import Line2D  # noqa: E402
a2.legend(handles=[Line2D([], [], color=TINTA, lw=1.4, label="Meta"),
                   Line2D([], [], ls="", marker="D", mfc="white", mec=TINTA2, label="Validación"),
                   Line2D([], [], ls="", marker="o", color=TINTA2, label="Prueba [IC95]")],
          fontsize=8, frameon=False, loc="lower right", ncol=1, bbox_to_anchor=(1.0, 0.16))
for ax in (a1, a2):
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
png, pdf, gris = sal / "figura_umbrales.png", sal / "figura_umbrales.pdf", sal / "figura_umbrales_gris.png"
fig.savefig(png, dpi=300)
fig.savefig(pdf)
plt.close(fig)
from PIL import Image  # noqa: E402
Image.open(png).convert("L").save(gris, dpi=(300, 300))
print(f"\nFigura guardada: {png}\n                 {pdf}\n                 {gris}")
print(f"\n{'Todo coincide.' if fallos == 0 else f'{fallos} afirmación(es) no coinciden: revisar.'}")
