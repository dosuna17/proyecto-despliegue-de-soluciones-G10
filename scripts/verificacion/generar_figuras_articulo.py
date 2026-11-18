"""Figuras de aprendizaje profundo para el artículo (reports/figures/aprendizaje_profundo).

Nombres y contenido según reports/figures/aprendizaje_profundo/README.md (origin/main):
  01_arquitectura_flexible.pdf      Esquema del modelo flexible: señales + marcas de presencia y rama EMG.
  02_curvas_entrenamiento_cv.pdf    Pérdida (arriba) y F1 macro (abajo) de entrenamiento y validación, un panel por pliegue.
  03_f1_por_combinacion_test.pdf    F1 macro en prueba con IC95 por combinación: EMG-MLP, flexible, sin LSTM y dedicados.
  04_matriz_confusion_campeon.pdf   Matrices normalizadas por fila del EMG-MLP: Fpz-Cz y la mejor combinación (Fpz-Cz + EOG + EMG).
  05_f1_por_persona.pdf             F1 macro por persona de prueba en las 12 combinaciones (cajas), jóvenes y mayores marcados.
  06_umbrales_por_clase.pdf         Cobertura frente a exactitud de lo aceptado por estadio, con el umbral y la meta (SleepFM − 5).
Formato: PDF vectorial (bbox_inches="tight"), sin título dentro de la figura, texto de 8 pt o más, legible en gris.
Cada figura sale de la sección marcada con su número. Nada se reentrena ni se vuelve a evaluar: todo sale de las
predicciones guardadas del único test, de las probabilidades de validación y de los historiales de MLflow.
Modelo seleccionado: EMG-MLP, run 95951e2a2cfd4368970636cc6bdea9b5.

Uso, desde la raíz del repositorio:
    python scripts/verificacion/generar_figuras_articulo.py --probs resultados_test ^
           --val-probs res_emg/probs_validacion --revision figuras_revision
    (pide el URI, el usuario y la clave de MLflow para bajar los historiales de los 5 pliegues)
    --cv-dir CARPETA usa flexible_emg_mlp_pliegueN.json ya bajados en vez de MLflow.
    --revision CARPETA guarda además copias PNG en color y en gris para revisar (no van al repositorio).
"""
import argparse
import getpass
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

RUN = "95951e2a2cfd4368970636cc6bdea9b5"
MODELO = "flexible_emg_mlp"
E, P, EO, M = "eeg_fpz_cz", "eeg_pz_oz", "eog_horizontal", "emg_submental"
NOM = {E: "Fpz-Cz", P: "Pz-Oz", f"{E}__{P}": "Fpz-Cz + Pz-Oz", f"{E}__{EO}": "Fpz-Cz + EOG", f"{P}__{EO}": "Pz-Oz + EOG",
       f"{E}__{P}__{EO}": "Fpz-Cz + Pz-Oz + EOG", f"{E}__{M}": "Fpz-Cz + EMG", f"{P}__{M}": "Pz-Oz + EMG",
       f"{E}__{P}__{M}": "Fpz-Cz + Pz-Oz + EMG", f"{E}__{EO}__{M}": "Fpz-Cz + EOG + EMG",
       f"{P}__{EO}__{M}": "Pz-Oz + EOG + EMG", f"{E}__{P}__{EO}__{M}": "Fpz-Cz + Pz-Oz + EOG + EMG"}
MEJOR = f"{E}__{EO}__{M}"
CLASES = ["W", "N1", "N2", "N3", "REM"]
ARCHIVOS = {"arquitectura": "01_arquitectura_flexible.pdf", "curvas": "02_curvas_entrenamiento_cv.pdf",
            "combinaciones": "03_f1_por_combinacion_test.pdf", "matriz": "04_matriz_confusion_campeon.pdf",
            "personas": "05_f1_por_persona.pdf", "umbrales": "06_umbrales_por_clase.pdf"}
ESCALA = ["#cde2fb", "#5598e7", "#104281"]          # tablero (graficas.matriz_confusion)
OSCURO, CLARO, TINTA, TINTA2, REJILLA = "#104281", "#eb6834", "#000000", "#52514e", "#e6e5e0"

ap = argparse.ArgumentParser()
ap.add_argument("--probs", required=True, help="carpeta con resultados_test.json y los probs_*.npz del test")
ap.add_argument("--val-probs", required=True, help="carpeta con flexible_emg_mlp_pliegueK__COMBO.npz (validación)")
ap.add_argument("--edades", default="data/raw/SC-subjects-split.xlsx")
ap.add_argument("--cv-dir", default=None)
ap.add_argument("--salida", default="reports/figures/aprendizaje_profundo")
ap.add_argument("--revision", default=None)
args = ap.parse_args()
car, sal = Path(args.probs), Path(args.salida)
sal.mkdir(parents=True, exist_ok=True)
rev = Path(args.revision) if args.revision else None
if rev:
    rev.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": TINTA2, "axes.labelcolor": TINTA,
                     "xtick.color": TINTA2, "ytick.color": TINTA2, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "legend.fontsize": 8, "pdf.fonttype": 42})

fallos, LETRA = 0, {}
def check(a, o, ok):
    global fallos
    fallos += not ok
    print(f"[{'OK ' if ok else 'NO '}] {a}\n       observado: {o}")


def letra_minima(fig):
    return min(t.get_fontsize() for t in fig.findobj(matplotlib.text.Text) if t.get_text().strip())


def guardar(fig, clave):
    ruta = sal / ARCHIVOS[clave]
    LETRA[clave] = letra_minima(fig)
    fig.savefig(ruta, bbox_inches="tight", facecolor="white")
    if rev:
        from PIL import Image
        png = rev / ruta.with_suffix(".png").name
        fig.savefig(png, dpi=200, bbox_inches="tight", facecolor="white")
        Image.open(png).convert("L").save(rev / (ruta.stem + "_gris.png"))
    plt.close(fig)
    print(f"Guardada: {ruta}")


def limpiar(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


R = json.load(open(car / "resultados_test.json", encoding="utf-8"))
J = R["modelos"][MODELO]
suj_rec = np.array([str(g)[3:5] for g in R["grabaciones"]])
TEST = {c: np.load(car / f"probs_{MODELO}__{c}.npz") for c in NOM}

# =============== 01 Arquitectura del modelo flexible ===============
fig, ax = plt.subplots(figsize=(7.2, 3.5))
ax.set_xlim(0, 100); ax.set_ylim(2, 52.5); ax.axis("off")
GRIS, FONDO = "#8a8a8a", "#e6e6e6"


def caja(x, y, w, h, txt, fc="white", ec=TINTA, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fc, ec=ec, lw=1.0))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=8, color=TINTA,
            fontweight="bold" if bold else "normal", linespacing=1.25)


def flecha(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9, lw=0.9, color=TINTA, shrinkA=0, shrinkB=0))


filas = [("EEG Fpz-Cz", True, "100 Hz"), ("EEG Pz-Oz", False, "100 Hz"), ("EOG horizontal", False, "100 Hz"), ("EMG submental", True, "1 Hz")]
ys = [41, 33, 25, 10]
ax.text(1, 50.8, "Entrada por época (posiciones fijas)", fontsize=8.5, fontweight="bold")
ax.text(21.1, 47.2, "marca", fontsize=8, ha="center", color=TINTA)
rg = np.random.default_rng(3)
for (nombre, presente, fs), y in zip(filas, ys):
    ax.text(1, y + 2.6, f"{nombre} · {fs}", fontsize=8, color=TINTA)
    xs = np.linspace(2, 17, 120 if fs == "100 Hz" else 30)
    if presente:
        sig = rg.normal(0, 0.9, xs.size) if fs == "100 Hz" else np.clip(rg.normal(0, 0.6, xs.size), -1, 1)
        ax.plot(xs, y + sig * 0.9, lw=0.6, color=TINTA, drawstyle="default" if fs == "100 Hz" else "steps-mid")
    else:
        ax.plot(xs, np.full(xs.size, y), lw=0.9, color=GRIS, ls=(0, (2, 1.5)))
        ax.text(9.5, y - 2.1, "ceros", fontsize=8, color=TINTA2, ha="center")
    caja(19, y - 1.3, 4.2, 2.6, "1" if presente else "0", fc="white" if presente else FONDO, ec=TINTA if presente else GRIS, bold=presente)
caja(29, 22, 21, 23, "CNN por época\n(TinySleepNet)\n\n6 canales:\n3 señales + 3 marcas\n\n→ 2,048 valores")
for y in ys[:3]:
    flecha(24.2, y, 28.6, min(max(y, 23), 44))
caja(29, 4.5, 21, 11, "Rama EMG (MLP)\n30 valores + marca\nLineal 31 → 32 → 16")
flecha(24.2, 10, 28.6, 10)
caja(54, 18, 12, 10, "concatenar\n2,048 + 16")
flecha(50.6, 33, 53.6, 25.5); flecha(50.6, 10, 53.6, 20.5)
caja(70, 14, 14, 18, "LSTM\n128 unidades\n\nsecuencia de\n15 épocas")
flecha(66.6, 23, 69.6, 23)
for i in range(5):
    ax.add_patch(plt.Rectangle((70.5 + i * 2.6, 8.8), 2.2, 2.6, fc=FONDO, ec=GRIS, lw=0.6))
ax.text(77, 6.4, "épocas vecinas", fontsize=8, ha="center", va="top", color=TINTA)
caja(88, 17, 11, 12, "Lineal\n→ 5 clases\nW · N1 · N2\nN3 · REM")
flecha(84.6, 23, 87.6, 23)
guardar(fig, "arquitectura")

# =============== 02 Curvas de entrenamiento por pliegue ===============
if args.cv_dir:
    cvdir = Path(args.cv_dir)
else:
    import mlflow
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        os.environ["MLFLOW_TRACKING_URI"] = input("MLFLOW_TRACKING_URI: ").strip()
        u = input("MLFLOW_TRACKING_USERNAME (vacío si no hace falta): ").strip()
        if u:
            os.environ["MLFLOW_TRACKING_USERNAME"] = u
            os.environ["MLFLOW_TRACKING_PASSWORD"] = getpass.getpass("MLFLOW_TRACKING_PASSWORD: ")
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    cvdir = Path(mlflow.artifacts.download_artifacts(run_id=RUN, artifact_path="resultados",
                                                    dst_path=tempfile.mkdtemp(prefix="figuras_articulo_")))
PL = sorted((json.load(open(a, encoding="utf-8")) for a in cvdir.rglob(f"{MODELO}_pliegue[0-9].json")), key=lambda x: x["pliegue"])
assert len(PL) == 5, f"se esperaban 5 pliegues y hay {len(PL)}"
fig, ejes = plt.subplots(2, 5, figsize=(7.2, 4.0), sharey="row", sharex="col", layout="constrained")
ymax = max(max(h["train_loss"], h["val_loss"]) for x in PL for h in x["historial"])
for k, x in enumerate(PL):
    h, b = x["historial"], x["mejor_epoca"]
    ep = [z["epoca"] for z in h]
    for fila, (tr, va, lim) in enumerate([("train_loss", "val_loss", (0, ymax * 1.08)),
                                          ("train_macro_f1", "val_macro_f1_medio", (0.4, 0.9))]):
        ax = ejes[fila, k]
        ax.plot(ep, [z[tr] for z in h], color=OSCURO, lw=1.3)
        ax.plot(ep, [z[va] for z in h], color=CLARO, lw=1.3, ls="--")
        ax.axvline(b, color=TINTA2, lw=.7, ls=":")
        ax.plot(b, h[b - 1][va], "o", ms=4.5, mfc="white", mec=TINTA, mew=1.1, zorder=5)
        ax.set_xlim(0.5, max(ep) + 0.5)
        ax.set_ylim(*lim)
        ax.grid(color=REJILLA, lw=.6)
        limpiar(ax)
    ejes[0, k].text(0.05, 0.04, f"Pliegue {x['pliegue'] + 1}\nmejor época {b}", transform=ejes[0, k].transAxes,
                    ha="left", va="bottom", fontsize=8, color=TINTA)
    ejes[1, k].set_xlabel("Época")
ejes[0, 0].set_ylabel("Pérdida\n(entropía cruzada + L2)")
ejes[1, 0].set_ylabel("F1 macro")
fig.legend(handles=[Line2D([], [], color=OSCURO, lw=1.3, label="Entrenamiento"),
                    Line2D([], [], color=CLARO, lw=1.3, ls="--", label="Validación (media de las 12 combinaciones)"),
                    Line2D([], [], ls="", marker="o", mfc="white", mec=TINTA, label="Mejor época")],
           loc="outside lower center", ncol=3, frameon=False)
guardar(fig, "curvas")
B = [x["mejor_epoca"] for x in PL]
check("02: mejores épocas 7, 15, 9, 12, 15 y cada pliegue paró 5 épocas después (paciencia 5)",
      f"mejores {B}; corridas {[len(x['historial']) for x in PL]}",
      B == [7, 15, 9, 12, 15] and all(len(x["historial"]) - x["mejor_epoca"] == 5 for x in PL))

# =============== 03 F1 macro en prueba por combinación, con IC95 ===============
def f1_macro(yy, pp):
    c = np.zeros((5, 5))
    np.add.at(c, (yy, pp), 1)
    return float(np.mean(2 * np.diag(c) / (c.sum(0) + c.sum(1))))


SERIES = [("flexible_emg_mlp", "Flexible con EMG (EMG-MLP, seleccionado)", OSCURO, "o", OSCURO),
          ("flexible", "Flexible sin EMG", OSCURO, "s", "white"),
          ("flexible_sin_lstm", "Flexible sin LSTM", TINTA2, "^", "white"),
          ("control", "Dedicado (uno por combinación)", CLARO, "D", CLARO)]
orden = list(NOM)[::-1]
fig, ax = plt.subplots(figsize=(7.2, 4.8), layout="constrained")
desv = {"flexible_emg_mlp": .27, "flexible": .09, "flexible_sin_lstm": -.09, "control": -.27}
dif = []
for clave, etiqueta, col, mk, relleno in SERIES:
    for i, c in enumerate(orden, start=1):
        m = f"control_{c}" if clave == "control" else clave
        if m not in R["modelos"] or c not in R["modelos"][m]["combinaciones"]:
            continue
        jc = R["modelos"][m]["combinaciones"][c]
        nombre = f"probs_{m}__{c}.npz"
        dd = np.load(car / nombre)
        f = f1_macro(dd["y_true"], dd["probs"].argmax(1))
        if abs(f - jc["macro_f1"]) > 1e-9:
            dif.append(nombre)
        lo, hi = jc["macro_f1_ic95"]
        yy = i + desv[clave]
        ax.plot([lo, hi], [yy, yy], color=col, lw=1.1)
        ax.plot(f, yy, mk, ms=4.6, color=col, mfc=relleno, mec=col, mew=1.1, zorder=4)
ax.set_yticks(range(1, 13), [NOM[c] for c in orden])
ax.set_ylim(0.5, 12.5)
ax.axhline(6.5, color=TINTA2, lw=.6)
ax.set_xlim(0.6, 0.83)
ax.text(0.602, 6.6, "↑ sin EMG", ha="left", va="bottom", fontsize=8, color=TINTA2)
ax.text(0.602, 6.4, "↓ con EMG", ha="left", va="top", fontsize=8, color=TINTA2)
ax.set_xlabel("F1 macro en prueba (16 personas, 44,985 épocas)\npunto: valor; línea: IC95 por bootstrap de personas")
ax.grid(axis="x", color=REJILLA, lw=.6)
limpiar(ax)
fig.legend(handles=[Line2D([], [], color=col, marker=mk, mfc=relleno, mec=col, lw=1.1, label=et) for _, et, col, mk, relleno in SERIES],
           loc="outside lower center", ncol=2, frameon=False)
guardar(fig, "combinaciones")
check("03: el F1 de cada punto, recalculado desde las predicciones, es igual al guardado por el 08 (29 modelo-combinación)",
      dif or "los 29 iguales", not dif)

# =============== 04 Matrices de confusión del modelo seleccionado ===============
cmap = LinearSegmentedColormap.from_list("tablero", ESCALA)


def contraste(c1, c2):
    def L(c):
        c = np.array(to_rgb(c)[:3])
        c = np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
        return float(np.dot(c, [0.2126, 0.7152, 0.0722]))
    a, b = sorted([L(c1), L(c2)], reverse=True)
    return (a + 0.05) / (b + 0.05)


fig, ejes = plt.subplots(1, 2, figsize=(7.2, 3.4), layout="constrained")
CM = {}
for ax, (combo, rot) in zip(ejes, [(E, "(a) Fpz-Cz"), (MEJOR, "(b) Fpz-Cz + EOG + EMG")]):
    d = TEST[combo]
    cm = np.zeros((5, 5), int)
    np.add.at(cm, (d["y_true"], d["probs"].argmax(1)), 1)
    CM[combo] = cm
    pct = 100 * cm / cm.sum(1, keepdims=True)
    im = ax.imshow(pct, cmap=cmap, vmin=0, vmax=100)
    for i in range(5):
        for j in range(5):
            col = max(["white", TINTA], key=lambda t: contraste(cmap(pct[i, j] / 100), t))
            ax.text(j, i - 0.13, f"{pct[i, j]:.0f} %", ha="center", va="center", fontsize=8.5, fontweight="bold", color=col)
            ax.text(j, i + 0.22, f"{cm[i, j]:,}".replace(",", " "), ha="center", va="center", fontsize=8, color=col)
    ax.set_xticks(range(5), CLASES)
    ax.set_yticks(range(5), CLASES)
    ax.set_xticks(np.arange(-.5, 5), minor=True)
    ax.set_yticks(np.arange(-.5, 5), minor=True)
    ax.grid(which="minor", color="white", lw=1.5)
    ax.tick_params(which="both", length=0)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.set_xlabel(f"Estadio predicho\n{rot}")
ejes[0].set_ylabel("Estadio real (experto)")
cb = fig.colorbar(im, ax=ejes, shrink=0.8, pad=0.02)
cb.set_label("% de las épocas reales del estadio (fila)", fontsize=8)
cb.ax.tick_params(labelsize=8)
guardar(fig, "matriz")
ok = {c: np.array_equal(CM[c], np.array(J["combinaciones"][c]["confusion_matrix"])) for c in CM}
check("04: las dos matrices son iguales a las guardadas por el 08 (44,985 épocas cada una)",
      {NOM[c]: (v, int(CM[c].sum())) for c, v in ok.items()}, all(ok.values()) and all(CM[c].sum() == 44985 for c in CM))

# =============== 05 F1 por persona ===============
def f1_persona(dd):
    yy, pp, s = dd["y_true"], dd["probs"].argmax(1), suj_rec[dd["rec_idx"]]
    out = {}
    for q in sorted(set(s)):
        k = s == q
        c = np.zeros((5, 5))
        np.add.at(c, (yy[k], pp[k]), 1)
        tp = np.diag(c)
        den = 2 * tp + (c.sum(0) - tp) + (c.sum(1) - tp)
        f = np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0)
        out[q] = f[c.sum(1) > 0].mean()
    return out


FP = {c: f1_persona(TEST[c]) for c in NOM}
x = pd.read_excel(args.edades)
x = x[x["split"] == "test"].groupby("subject")["age"].first()
EDAD = {f"{int(s):02d}": int(a) for s, a in x.items()}
personas = sorted(FP[E])
corte = float(np.median([EDAD[q] for q in personas]))
joven = {q: EDAD[q] <= corte for q in personas}
fig, ax = plt.subplots(figsize=(7.2, 4.6), layout="constrained")
datos = [[FP[c][q] for q in personas] for c in orden]
estilo = dict(widths=.55, patch_artist=True, showfliers=False, medianprops=dict(color=TINTA, lw=1.4),
              whiskerprops=dict(color=TINTA2), capprops=dict(color=TINTA2), boxprops=dict(facecolor="#e8f0fb", edgecolor=TINTA2))
try:
    ax.boxplot(datos, orientation="horizontal", **estilo)          # matplotlib >= 3.10
except TypeError:
    ax.boxplot(datos, vert=False, **estilo)
rng = np.random.default_rng(0)
for i, c in enumerate(orden, start=1):
    for q in personas:
        yj = i + rng.uniform(-.17, .17)
        if joven[q]:
            ax.plot(FP[c][q], yj, "o", ms=3.6, color=OSCURO, mec=OSCURO, alpha=.9, zorder=3)
        else:
            ax.plot(FP[c][q], yj, "o", ms=3.6, mfc="white", mec=CLARO, mew=1.1, zorder=3)
ax.set_yticks(range(1, 13), [NOM[c] for c in orden])
ax.set_xlabel("F1 macro de la persona (sus dos noches)")
ax.grid(axis="x", color=REJILLA, lw=.6)
ax.axhline(6.5, color=TINTA2, lw=.6)
ax.set_xlim(0.45, None)
ax.text(0.453, 6.6, "↑ sin EMG", ha="left", va="bottom", fontsize=8, color=TINTA2)
ax.text(0.453, 6.4, "↓ con EMG", ha="left", va="top", fontsize=8, color=TINTA2)
limpiar(ax)
edades = np.array([EDAD[q] for q in personas])
fig.legend(handles=[Line2D([], [], ls="", marker="o", color=OSCURO, label=f"≤ {corte:.1f} años (n = {(edades <= corte).sum()}, {edades[edades <= corte].min()}–{edades[edades <= corte].max()})"),
                    Line2D([], [], ls="", marker="o", mfc="white", mec=CLARO, mew=1.1, label=f"> {corte:.1f} años (n = {(edades > corte).sum()}, {edades[edades > corte].min()}–{edades[edades > corte].max()})"),
                    Line2D([], [], color=TINTA, lw=1.4, label="Mediana; caja: cuartiles")],
           loc="outside lower center", frameon=False, ncol=3)
guardar(fig, "personas")
v = np.array([FP[MEJOR][q] for q in personas])
check("05: Fpz-Cz + EOG + EMG 0.758 ± 0.072 por persona, de 0.611 a 0.877 (tabla_por_persona.py y figura_sobreajuste.py); 16 personas por caja",
      f"{v.mean():.3f} ± {v.std(ddof=1):.3f}; {v.min():.3f}–{v.max():.3f}; cajas {[len(z) for z in datos]}",
      (f"{v.mean():.3f}", f"{v.std(ddof=1):.3f}", f"{v.min():.3f}", f"{v.max():.3f}") == ("0.758", "0.072", "0.611", "0.877")
      and all(len(z) == 16 for z in datos))

# =============== 06 Umbrales por clase: cobertura frente a exactitud por estadio ===============
bloque = J["umbrales_validacion"]["por_clase"]["global"]["0.05"]
TAU = {c: bloque["clases"][c]["umbral"] for c in CLASES}
META = {c: bloque["clases"][c]["meta_exactitud"] for c in CLASES}
PT = np.concatenate([TEST[c]["probs"] for c in NOM]); YT = np.concatenate([TEST[c]["y_true"] for c in NOM])
pv, yv = [], []
for c in NOM:
    for k in range(5):
        dd = np.load(next(Path(args.val_probs).rglob(f"{MODELO}_pliegue{k}__{c}.npz")))
        pv.append(dd["probs"]); yv.append(dd["y_true"])
PV, YV = np.concatenate(pv), np.concatenate(yv)
UMB = np.round(np.arange(0, 1, .01), 2)


def curva(probs, yy, k):
    conf, pr = probs.max(1), probs.argmax(1)
    es = pr == k
    cob, ex = [], []
    for t in UMB:
        m = es & (conf >= t)
        if m.sum() < 50:                                     # puntos con muy pocas épocas: se cortan
            break
        cob.append(100 * m.sum() / es.sum()); ex.append(100 * (yy[m] == k).mean())
    return np.array(cob), np.array(ex), es


fig, ejes = plt.subplots(2, 3, figsize=(7.2, 4.6), layout="constrained")
ejes = ejes.ravel()
PUNTO = {}
for i, c in enumerate(CLASES):
    ax = ejes[i]
    for probs, yy, col, ls, nom in [(PV, YV, OSCURO, "--", "val"), (PT, YT, CLARO, "-", "test")]:
        cx, ex, es = curva(probs, yy, i)
        ax.plot(cx, ex, color=col, ls=ls, lw=1.4)
        m = es & (probs.max(1) >= TAU[c])
        PUNTO[(c, nom)] = (m.sum() / es.sum(), (yy[m] == i).mean())
        ax.plot(100 * PUNTO[(c, nom)][0], 100 * PUNTO[(c, nom)][1], "o" if nom == "test" else "D", ms=5,
                mfc="white", mec=col, mew=1.3, zorder=5)
    ax.axhline(100 * META[c], color=TINTA2, lw=.8, ls=":")
    ax.invert_xaxis()
    ax.text(0.04, 0.95, f"{c}  (τ = {TAU[c]:.2f})", transform=ax.transAxes, ha="left", va="top", fontsize=8.5,
            fontweight="bold", color=TINTA)
    ax.grid(color=REJILLA, lw=.6)
    limpiar(ax)
    if i % 3 == 0:
        ax.set_ylabel("Exactitud de lo aceptado (%)")
    if i >= 2:
        ax.set_xlabel("Cobertura del estadio (%)")
ejes[5].axis("off")
ejes[5].legend(handles=[Line2D([], [], color=OSCURO, ls="--", lw=1.4, label="Validación cruzada\n(5 pliegues, 12 combinaciones)"),
                        Line2D([], [], color=CLARO, lw=1.4, label="Prueba\n(16 personas, 12 combinaciones)"),
                        Line2D([], [], ls="", marker="D", mfc="white", mec=OSCURO, label="Umbral τ en validación"),
                        Line2D([], [], ls="", marker="o", mfc="white", mec=CLARO, label="Umbral τ en prueba"),
                        Line2D([], [], color=TINTA2, ls=":", lw=.8, label="Meta (SleepFM − 5 puntos)")],
               loc="center", frameon=False)
guardar(fig, "umbrales")
jt = J["umbral_global_junto"]["por_clase"]["test"]["clases"]
f3 = lambda t: (round(float(t[0]), 3), round(float(t[1]), 3))
check("06: punto de prueba de cada estadio igual a la cobertura y exactitud guardadas por el 08 (tabla_confianza.py)",
      {c: f3(PUNTO[(c, "test")]) for c in CLASES},
      all(f3(PUNTO[(c, "test")]) == f3((jt[c]["cobertura"], jt[c]["exactitud_aceptadas"])) for c in CLASES))
check("06: punto de validación de cada estadio igual al guardado al fijar los umbrales",
      {c: f3(PUNTO[(c, "val")]) for c in CLASES},
      all(f3(PUNTO[(c, "val")]) == f3((bloque["clases"][c]["validacion"]["cobertura"],
                                        bloque["clases"][c]["validacion"]["exactitud_aceptadas"])) for c in CLASES))
check("Letra mínima de 8 pt en las seis figuras", LETRA, min(LETRA.values()) >= 8)
print(f"\nFiguras en {sal}. {'Todo coincide.' if fallos == 0 else f'{fallos} no coinciden: revisar.'}")
