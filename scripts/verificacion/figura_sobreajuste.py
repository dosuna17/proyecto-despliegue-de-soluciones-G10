"""Sobreajuste (curvas por pliegue) y variabilidad entre personas, con verificación del texto.

(a) pérdida y (b) F1 macro de entrenamiento y validación, época a época, en los 5 pliegues de la validación cruzada
    del modelo seleccionado (EMG-MLP). Salen del historial guardado en resultados/flexible_emg_mlp_pliegueN.json.
(c) F1 macro de cada una de las 16 personas de prueba con Fpz-Cz + EOG + EMG, desde las predicciones guardadas.
No se reentrena ni se vuelve a evaluar nada.

Uso, desde la raíz del repositorio:
    python scripts/verificacion/figura_sobreajuste.py --probs resultados_test
    (pide el URI, el usuario y la clave de MLflow para bajar los JSON de cada pliegue de los 8 modelos)
    --cv-dir CARPETA usa JSON ya bajados (MODELO_pliegueN.json) en vez de MLflow.
Salida (por defecto en la carpeta del script; usar --salida reports/figures): figura_sobreajuste.png (300 dpi), .pdf y una copia en gris.
"""
import argparse
import getpass
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RUNS = {"flexible_emg_mlp": "95951e2a2cfd4368970636cc6bdea9b5", "flexible": "029cc677c3734b10959735a6e9246cb8",
        "flexible_sin_lstm": "c424e04708c34f04a5bfec810b705982",
        "control_eeg_fpz_cz": "2bd024695ad44d19b175490e5abbdedb", "control_eeg_pz_oz": "30877a344b304c5a8a2b5f2e3163805f",
        "control_eeg_fpz_cz__eeg_pz_oz": "076282f5e5534691b441514b50e90441",
        "control_eeg_fpz_cz__eog_horizontal": "038a654fc695444ab650f4e2aac9056f",
        "control_eeg_fpz_cz__eeg_pz_oz__eog_horizontal__emg_submental": "74b16a9d1a854bb5a5589353fec14f91"}
SEL = "flexible_emg_mlp"
COMBO_PERSONA = "eeg_fpz_cz__eog_horizontal__emg_submental"
C_TRAIN, C_VAL, C_PUNTO = "#104281", "#eb6834", "#5598e7"      # azul oscuro y naranja del tablero (graficas.py)
TINTA, TINTA2, REJILLA = "#0b0b0b", "#52514e", "#e6e5e0"

p = argparse.ArgumentParser()
p.add_argument("--probs", required=True, help="carpeta con resultados_test.json y los probs_*.npz")
p.add_argument("--cv-dir", default=None, help="carpeta con MODELO_pliegueN.json ya bajados (si no, se bajan de MLflow)")
p.add_argument("--salida", default=None, help="carpeta de salida (por defecto, la de este script)")
args = p.parse_args()
sal = Path(args.salida) if args.salida else Path(__file__).resolve().parent

fallos = 0
def check(afirmacion, observado, ok):
    global fallos
    fallos += not ok
    print(f"[{'OK ' if ok else 'NO '}] {afirmacion}\n       observado: {observado}")

f3 = lambda *v: tuple(float(f"{x:.3f}") for x in v)

# ---------------- Historial de cada pliegue ----------------
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
    tmp = Path(tempfile.mkdtemp(prefix="sobreajuste_"))
    carpetas = {m: Path(mlflow.artifacts.download_artifacts(run_id=rid, artifact_path="resultados", dst_path=str(tmp / m)))
                for m, rid in RUNS.items()}
PL = {}
for m, d in carpetas.items():
    v = sorted((json.load(open(a, encoding="utf-8")) for a in d.glob(f"{m}_pliegue[0-9].json")), key=lambda x: x["pliegue"])
    if v:
        PL[m] = v

# ---------------- Variabilidad entre personas (prueba) ----------------
R = json.load(open(Path(args.probs) / "resultados_test.json", encoding="utf-8"))
suj_rec = np.array([str(g)[3:5] for g in R["grabaciones"]])

def f1_por_persona(combo):
    """Como metricas_por_persona de scripts/08: F1 macro de cada persona sobre las clases que tiene."""
    d = np.load(Path(args.probs) / f"probs_{SEL}__{combo}.npz")
    y, yp, s = d["y_true"], d["probs"].argmax(1), suj_rec[d["rec_idx"]]
    out = {}
    for q in sorted(set(s)):
        m = s == q
        cm = np.zeros((5, 5))
        np.add.at(cm, (y[m], yp[m]), 1)
        tp = np.diag(cm)
        den = 2 * tp + (cm.sum(0) - tp) + (cm.sum(1) - tp)
        f = np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0)
        out[q] = f[cm.sum(1) > 0].mean()
    return out

pp = f1_por_persona(COMBO_PERSONA)

# ---------------- Figura ----------------
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": TINTA2,
                     "axes.labelcolor": TINTA, "xtick.color": TINTA2, "ytick.color": TINTA2})
fig, (axl, axf, axp) = plt.subplots(1, 3, figsize=(7.2, 2.7), constrained_layout=True,
                                    gridspec_kw={"width_ratios": [1, 1, 1.05]})
for pl in PL[SEL]:
    h = pl["historial"]
    ep = [x["epoca"] for x in h]
    b = pl["mejor_epoca"]
    axl.plot(ep, [x["train_loss"] for x in h], color=C_TRAIN, lw=1, alpha=.8)
    axl.plot(ep, [x["val_loss"] for x in h], color=C_VAL, lw=1, ls="--", alpha=.9)
    axf.plot(ep, [x["train_macro_f1"] for x in h], color=C_TRAIN, lw=1, alpha=.8)
    axf.plot(ep, [x["val_macro_f1_medio"] for x in h], color=C_VAL, lw=1, ls="--", alpha=.9)
    axf.plot(b, h[b - 1]["val_macro_f1_medio"], "o", ms=4.5, mfc="white", mec=TINTA, mew=1.1, zorder=5)
    axl.plot(b, h[b - 1]["val_loss"], "o", ms=4.5, mfc="white", mec=TINTA, mew=1.1, zorder=5)
for ax, t, yl in [(axl, "(a) Pérdida", "Pérdida (entropía cruzada + L2)"), (axf, "(b) F1 macro", "F1 macro")]:
    ax.set_title(t, loc="left", fontsize=9)
    ax.set_xlabel("Época")
    ax.set_ylabel(yl)
    ax.grid(color=REJILLA, lw=.6)
    ax.set_xticks(range(0, 21, 5))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
from matplotlib.lines import Line2D  # noqa: E402
axf.legend(handles=[Line2D([], [], color=C_TRAIN, lw=1.2, label="Entrenamiento"),
                    Line2D([], [], color=C_VAL, lw=1.2, ls="--", label="Validación"),
                    Line2D([], [], ls="", marker="o", mfc="white", mec=TINTA, label="Mejor época")],
           fontsize=8, loc="lower right", frameon=False)
orden = sorted(pp, key=pp.get)
vals = np.array([pp[q] for q in orden])
axp.axhspan(vals.mean() - vals.std(ddof=1), vals.mean() + vals.std(ddof=1), color=REJILLA, lw=0)
axp.axhline(vals.mean(), color=TINTA2, lw=1)
axp.plot(range(len(orden)), vals, "o", ms=4.5, color=C_PUNTO, mec=C_TRAIN, mew=.8)
axp.set_xticks(range(len(orden)), orden, fontsize=8, rotation=90)
axp.set_xlabel("Persona de prueba (código)")
axp.set_ylabel("F1 macro de la persona")
axp.set_title("(c) Personas de prueba", loc="left", fontsize=9)
axp.text(0.98, 0.04, f"línea: media {vals.mean():.3f} ± {vals.std(ddof=1):.3f}\nfranja: ± 1 desviación", transform=axp.transAxes,
         ha="right", va="bottom", fontsize=8, color=TINTA2)
axp.grid(axis="y", color=REJILLA, lw=.6)
for s in ("top", "right"):
    axp.spines[s].set_visible(False)
png, pdf, gris = sal / "figura_sobreajuste.png", sal / "figura_sobreajuste.pdf", sal / "figura_sobreajuste_gris.png"
fig.savefig(png, dpi=300)
fig.savefig(pdf)
plt.close(fig)
from PIL import Image  # noqa: E402
Image.open(png).convert("L").save(gris, dpi=(300, 300))
print(f"Figura guardada: {png}\n                 {pdf}\n                 {gris}\n")

# ---------------- Verificación: curvas del EMG-MLP ----------------
print("== Curvas por pliegue del modelo seleccionado (EMG-MLP) ==")
P = PL[SEL]
H = [x["historial"] for x in P]
B = [x["mejor_epoca"] for x in P]
check("5 pliegues; cada uno paró 5 épocas después de su mejor época (paciencia 5): no se cortó por el máximo de épocas",
      f"{len(P)} pliegues; épocas corridas {[len(h) for h in H]}; mejores {B}",
      len(P) == 5 and all(len(h) == b + 5 for h, b in zip(H, B)))
check("La mejor época de cada pliegue es la de mayor F1 macro medio de validación",
      [max(h, key=lambda x: x["val_macro_f1_medio"])["epoca"] for h in H],
      all(max(h, key=lambda x: x["val_macro_f1_medio"])["epoca"] == b for h, b in zip(H, B)))
check("Mejores épocas 7, 15, 9, 12 y 15 (media 11.6): el modelo final se entrena 12 épocas",
      f"{B}, media {np.mean(B):.1f}", B == [7, 15, 9, 12, 15])
tl = all(np.all(np.diff([x["train_loss"] for x in h]) < 0) for h in H)
check("La pérdida de entrenamiento baja en todas las épocas de todos los pliegues", tl, tl)
vmin = [int(np.argmin([x["val_loss"] for x in h])) + 1 for h in H]
check("La pérdida de validación oscila más; su mínimo cae en la mejor época (por F1) en 4 de 5 pliegues",
      f"época de mínima pérdida de validación {vmin} frente a mejores {B}", sum(a == b for a, b in zip(vmin, B)) == 4)
gap = [h[b - 1]["train_macro_f1"] - h[b - 1]["val_macro_f1_medio"] for h, b in zip(H, B)]
check("En la mejor época, F1 de entrenamiento menos F1 de validación: de 0.000 a 0.089, media 0.036",
      f"{[round(g, 3) for g in gap]}, media {np.mean(gap):.3f}",
      f3(min(gap), max(gap), np.mean(gap)) in [(-0.0, 0.089, 0.036), (0.0, 0.089, 0.036)])
dtr = [h[-1]["train_macro_f1"] - h[b - 1]["train_macro_f1"] for h, b in zip(H, B)]
dva = [h[-1]["val_macro_f1_medio"] - h[b - 1]["val_macro_f1_medio"] for h, b in zip(H, B)]
check("En las 5 épocas siguientes a la mejor, el F1 de entrenamiento sigue subiendo (+0.014 a +0.021) y el de validación baja (−0.001 a −0.019)",
      f"entrenamiento {[round(x, 3) for x in dtr]}; validación {[round(x, 3) for x in dva]}",
      all(x > 0 for x in dtr) and all(x < 0 for x in dva)
      and f3(min(dtr), max(dtr), max(dva), min(dva)) == (0.014, 0.021, -0.001, -0.019))
vb = [x["val_macro_f1_medio"] for x in P]
check("Entre pliegues, el F1 de validación en la mejor época va de 0.712 a 0.779: diferencia de 0.068, casi el doble de la brecha media",
      f"{[round(v, 3) for v in vb]}; rango {max(vb) - min(vb):.3f}", f3(min(vb), max(vb), max(vb) - min(vb)) == (0.712, 0.779, 0.068))

# ---------------- Verificación: personas ----------------
print("\n== Variabilidad entre personas de prueba (EMG-MLP, Fpz-Cz + EOG + EMG) ==")
v = np.array(list(pp.values()))
check("16 personas: F1 macro por persona 0.758 ± 0.072, mínimo 0.611 y máximo 0.877",
      f"{len(v)} personas; {v.mean():.4f} ± {v.std(ddof=1):.4f}; mínimo {v.min():.4f}, máximo {v.max():.4f}",
      len(v) == 16 and f3(v.mean(), v.std(ddof=1), v.min(), v.max()) == (0.758, 0.072, 0.611, 0.877))
peor, mejor = min(pp, key=pp.get), max(pp, key=pp.get)
otras = {c: f1_por_persona(c) for c in ["eeg_fpz_cz", "eeg_fpz_cz__eeg_pz_oz__eog_horizontal__emg_submental"]}
misma = all(min(o, key=o.get) == peor and max(o, key=o.get) == mejor for o in otras.values())
check("La persona con peor F1 (58) y la con mejor (07) son las mismas con Fpz-Cz sola y con las cuatro señales: "
      "la variabilidad depende de la persona más que de la combinación",
      f"peor {peor}, mejor {mejor}; con Fpz-Cz: peor {min(otras['eeg_fpz_cz'], key=otras['eeg_fpz_cz'].get)}, "
      f"mejor {max(otras['eeg_fpz_cz'], key=otras['eeg_fpz_cz'].get)}; mismas en las tres combinaciones: {misma}",
      peor == "58" and mejor == "07" and misma)

# ---------------- Los 8 modelos: brecha y épocas ----------------
print("\n== Los 8 modelos (brecha media = F1 de entrenamiento menos F1 de validación en la mejor época) ==")
RES = {}
for m, P in PL.items():
    H = [x["historial"] for x in P]
    B = [x["mejor_epoca"] for x in P]
    RES[m] = dict(corridas=[len(h) for h in H], mejores=B,
                  brecha=np.mean([h[b - 1]["train_macro_f1"] - h[b - 1]["val_macro_f1_medio"] for h, b in zip(H, B)]),
                  despues=np.mean([h[-1]["train_macro_f1"] - h[b - 1]["train_macro_f1"] for h, b in zip(H, B)]))
    r = RES[m]
    print(f"   {m:62s} | corridas {r['corridas']} | mejores {r['mejores']} | brecha {r['brecha']:+.3f} | "
          f"F1 entr. después de la mejor {r['despues']:+.3f}")
if len(RES) == 8:
    flex = [RES[m]["brecha"] for m in RES if m.startswith("flexible")]
    ctrl = [RES[m]["brecha"] for m in RES if m.startswith("control")]
    dctrl = [RES[m]["despues"] for m in RES if m.startswith("control")]
    check("Brecha media de los 3 modelos flexibles: 0.011 a 0.036 (paciencia 5); de los 5 dedicados: 0.036 a 0.082 (paciencia 60)",
          f"flexibles {[round(float(x), 3) for x in flex]}; dedicados {[round(float(x), 3) for x in ctrl]}",
          f3(min(flex), max(flex), min(ctrl), max(ctrl)) == (0.011, 0.036, 0.036, 0.082))
    check("En los dedicados, después de su mejor época el F1 de entrenamiento todavía sube +0.030 a +0.036 (media por modelo)",
          [round(float(x), 3) for x in dctrl], f3(min(dctrl), max(dctrl)) == (0.030, 0.036))
    pl_ctrl = [(m, k, n, b) for m in RES if m.startswith("control")
               for k, (n, b) in enumerate(zip(RES[m]["corridas"], RES[m]["mejores"]))]
    tope = [(m, k) for m, k, n, b in pl_ctrl if n == 240 and n != b + 60]
    pac = sum(n == b + 60 for m, k, n, b in pl_ctrl)
    check("Dos pliegues de los dedicados llegaron al máximo de 240 épocas sin agotar la paciencia; los otros 23 pararon por paciencia (60)",
          f"en el máximo: {tope}; pararon 60 épocas después de su mejor: {pac} de {len(pl_ctrl)}",
          len(tope) == 2 and pac == 23 and len(pl_ctrl) == 25)
else:
    print("   (faltan modelos: los chequeos de los 8 modelos necesitan los 8 runs)")

print(f"\n{'Todo coincide.' if fallos == 0 else f'{fallos} afirmación(es) no coinciden: revisar.'}")
