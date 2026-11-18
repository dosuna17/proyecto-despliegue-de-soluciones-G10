"""Matrices de confusión del modelo seleccionado (EMG-MLP) en prueba, y verificación de lo que dice el texto.

Genera la figura desde las predicciones ya guardadas del único test (no vuelve a evaluar nada):
    (a) Fpz-Cz sola   (b) la mejor combinación: Fpz-Cz + EOG + EMG
Cada celda: % de la fila (épocas reales de ese estadio) y, debajo, el número de épocas.
Paleta: la misma escala azul de la matriz de confusión del tablero (app/dashboard/graficas.py, tema claro).

Uso, desde la raíz del repositorio:
    python scripts/verificacion/figura_matrices_confusion.py --probs resultados_test --salida reports/figures
Salida: <salida>/figura_matrices_confusion.png (300 dpi), .pdf y una copia en gris para revisarla.
"""
import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402

CLASES = ["W", "N1", "N2", "N3", "REM"]
PANELES = [("eeg_fpz_cz", "(a) Fpz-Cz"),
           ("eeg_fpz_cz__eog_horizontal__emg_submental", "(b) Fpz-Cz + EOG + EMG")]
ESCALA = ["#cde2fb", "#5598e7", "#104281"]          # graficas.matriz_confusion, tema claro
TINTA = "#000000"                   # negro puro: con #0b0b0b del tablero el contraste mínimo baja a 4.4
# F1 por estadio reportado en prueba (EMG-MLP), para comprobar que la figura sale de los mismos datos
F1_T5 = {"eeg_fpz_cz": (.910, .508, .841, .775, .710),
         "eeg_fpz_cz__eog_horizontal__emg_submental": (.915, .537, .851, .780, .835)}
F1_MACRO_T5 = {"eeg_fpz_cz": .749, "eeg_fpz_cz__eog_horizontal__emg_submental": .784}

p = argparse.ArgumentParser()
p.add_argument("--probs", required=True, help="carpeta con los probs_flexible_emg_mlp__*.npz")
p.add_argument("--salida", default=None, help="carpeta de salida (por defecto, la carpeta de este script)")
args = p.parse_args()
car = Path(args.probs)
sal = Path(args.salida) if args.salida else Path(__file__).resolve().parent

fallos = 0
def check(afirmacion, observado, ok):
    global fallos
    fallos += not ok
    print(f"[{'OK ' if ok else 'NO '}] {afirmacion}\n       observado: {observado}")

def matriz(combo, modelo="flexible_emg_mlp"):
    d = np.load(car / f"probs_{modelo}__{combo}.npz")
    return confusion_matrix(d["y_true"], d["probs"].argmax(1), labels=range(5)), d["y_true"]

CM = {c: matriz(c)[0] for c, _ in PANELES}
PCT = {c: 100 * m / m.sum(1, keepdims=True) for c, m in CM.items()}
F1 = {c: 2 * np.diag(m) / (m.sum(0) + m.sum(1)) for c, m in CM.items()}

# ---------------- Figura ----------------
cmap = LinearSegmentedColormap.from_list("tablero", ESCALA)


def contraste(c1, c2):
    """Razón de contraste WCAG 2 entre dos colores."""
    def L(c):
        c = np.array(to_rgb(c)[:3])
        c = np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
        return float(np.dot(c, [0.2126, 0.7152, 0.0722]))
    a, b = sorted([L(c1), L(c2)], reverse=True)
    return (a + 0.05) / (b + 0.05)


def color_texto(pct):
    """Blanco o tinta, el que más contraste dé sobre el fondo de la celda."""
    fondo = cmap(pct / 100)
    return max(["white", TINTA], key=lambda t: contraste(fondo, t))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
fig, ejes = plt.subplots(1, 2, figsize=(7.2, 3.75), constrained_layout=True)
for ax, (combo, titulo) in zip(ejes, PANELES):
    cm, pct = CM[combo], PCT[combo]
    im = ax.imshow(pct, cmap=cmap, vmin=0, vmax=100)
    for i in range(5):
        for j in range(5):
            color = color_texto(pct[i, j])
            ax.text(j, i - 0.13, f"{pct[i, j]:.0f} %", ha="center", va="center", fontsize=8.5,
                    fontweight="bold", color=color)
            ax.text(j, i + 0.22, f"{cm[i, j]:,}".replace(",", " "), ha="center", va="center", fontsize=8, color=color)
    ax.set_xticks(range(5), CLASES)
    ax.set_yticks(range(5), CLASES)
    ax.set_xticks(np.arange(-.5, 5), minor=True)
    ax.set_yticks(np.arange(-.5, 5), minor=True)
    ax.grid(which="minor", color="white", lw=1.5)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("Estadio predicho por el modelo")
    if ax is ejes[0]:
        ax.set_ylabel("Estadio real (experto)")
    f1 = 2 * np.diag(cm) / (cm.sum(0) + cm.sum(1))
    ax.set_title(f"{titulo}\nF1 macro {f1.mean():.3f}", fontsize=9, loc="left")
cb = fig.colorbar(im, ax=ejes, shrink=0.8, pad=0.02)
cb.set_label("% de las épocas reales del estadio (fila)", fontsize=8)
cb.ax.tick_params(labelsize=8)
png, pdf, gris = sal / "figura_matrices_confusion.png", sal / "figura_matrices_confusion.pdf", sal / "figura_matrices_confusion_gris.png"
fig.savefig(png, dpi=300)
fig.savefig(pdf)
plt.close(fig)
from PIL import Image  # noqa: E402
Image.open(png).convert("L").save(gris, dpi=(300, 300))
print(f"Figura guardada: {png}\n                 {pdf}\n                 {gris} (copia en gris para revisar)\n")

# ---------------- Verificación ----------------
print("== Datos de la figura ==")
_, y = matriz(PANELES[0][0])
iguales = all(np.array_equal(CM[c].sum(1), np.bincount(y, minlength=5)) for c in CM)
check("Las dos matrices usan las mismas 44,985 épocas de prueba (mismas épocas reales por estadio)",
      f"total {[int(m.sum()) for m in CM.values()]}; épocas reales por estadio {dict(zip(CLASES, np.bincount(y).tolist()))}; iguales: {iguales}",
      all(m.sum() == 44985 for m in CM.values()) and iguales)
for c, t in PANELES:
    f1 = 2 * np.diag(CM[c]) / (CM[c].sum(0) + CM[c].sum(1))
    ok = tuple(float(f"{v:.3f}") for v in f1) == F1_T5[c] and float(f"{f1.mean():.3f}") == F1_MACRO_T5[c]
    check(f"{t}: el F1 por estadio y el F1 macro que salen de la matriz son los reportados en prueba",
          " ".join(f"{k} {v:.3f}" for k, v in zip(CLASES, f1)) + f" | macro {f1.mean():.3f}", ok)
check("Cada fila suma 100 % (normalizada por fila)", [np.round(PCT[c].sum(1), 6).tolist() for c in PCT],
      all(np.allclose(PCT[c].sum(1), 100) for c in PCT))

print("\n== Requisitos de la figura ==")
im_png = Image.open(png)
dpi = im_png.info.get("dpi", (0, 0))
check("PNG a 300 dpi, de ancho de doble columna (unos 18 cm)", f"dpi {tuple(round(d) for d in dpi)}, {im_png.size[0]} x {im_png.size[1]} px "
      f"= {im_png.size[0] / 300 * 2.54:.1f} x {im_png.size[1] / 300 * 2.54:.1f} cm", round(dpi[0]) == 300)
lum = lambda h: float(np.dot(to_rgb(h), [0.299, 0.587, 0.114]))
grises = [lum(cmap(v)) for v in np.linspace(0, 1, 101)]
check("Legible en gris: la escala de color va de claro a oscuro sin saltos (cada tono de gris corresponde a un solo %)",
      f"luminancia de 0 % a 100 %: {grises[0]:.2f} -> {grises[50]:.2f} -> {grises[-1]:.2f}; estrictamente decreciente: "
      f"{all(a > b for a, b in zip(grises, grises[1:]))}", all(a > b for a, b in zip(grises, grises[1:])))

peor = min(contraste(cmap(v / 100), color_texto(v)) for v in range(101))
check("El número de cada celda se lee sobre su fondo: texto blanco u oscuro según el fondo, contraste WCAG >= 4.5 en todo el rango 0–100 %",
      f"contraste mínimo {peor:.1f}", peor >= 4.5)

print("\n== Lectura propuesta para el texto ==")
a, b = PCT[PANELES[0][0]], PCT[PANELES[1][0]]
W, N1, N2, N3, REM = range(5)
f = lambda *v: tuple(float(f"{x:.1f}") for x in v)      # redondeo a un decimal, como en el texto
check("N1 tiene el F1 más bajo en las dos, pero no el peor acierto por fila con Fpz-Cz: ahí REM (62.0 %) queda por debajo de N1 (63.3 %); "
      "N1 acierta 63.3 y 61.4 % y sus errores van a N2 (16.6 y 17.5 %) y a W (14.1 % en las dos)",
      f"Fpz-Cz: N1 {a[N1, N1]:.1f} %, a N2 {a[N1, N2]:.1f} %, a W {a[N1, W]:.1f} % | mejor: N1 {b[N1, N1]:.1f} %, a N2 {b[N1, N2]:.1f} %, a W {b[N1, W]:.1f} %",
      np.argmin(F1[PANELES[0][0]]) == N1 and np.argmin(F1[PANELES[1][0]]) == N1
      and np.argmin(np.diag(a)) == REM and np.argmin(np.diag(b)) == N1
      and f(a[N1, N1], b[N1, N1], a[N1, N2], b[N1, N2], a[N1, W], b[N1, W]) == (63.3, 61.4, 16.6, 17.5, 14.1, 14.1))
pa, pb = CM[PANELES[0][0]].sum(0)[N1], CM[PANELES[1][0]].sum(0)[N1]
real = CM[PANELES[0][0]].sum(1)[N1]
check("El modelo predice N1 de más: 7,568 épocas predichas frente a 5,077 reales con Fpz-Cz (1.5 veces) y 6,547 (1.3 veces) con la mejor combinación",
      f"Fpz-Cz {pa:,} predichas / {real:,} reales = {pa / real:.2f}; mejor {pb:,} = {pb / real:.2f}",
      (pa, pb, real) == (7568, 6547, 5077))
check("El cambio grande con la mejor combinación es REM: 62.0 % -> 82.2 % de acierto; REM tomado por N1 baja de 22.5 % a 8.4 %",
      f"REM {a[REM, REM]:.1f} % -> {b[REM, REM]:.1f} %; REM a N1 {a[REM, N1]:.1f} % -> {b[REM, N1]:.1f} %; "
      f"cambios en la diagonal: " + ", ".join(f"{k} {b[i, i] - a[i, i]:+.1f}" for i, k in enumerate(CLASES)),
      f(a[REM, REM], b[REM, REM], a[REM, N1], b[REM, N1]) == (62.0, 82.2, 22.5, 8.4)
      and np.argmax(np.abs(np.diag(b) - np.diag(a))) == REM)
rem = {c: 100 * (m := matriz(c)[0])[REM, REM] / m[REM].sum() for c in
       ["eeg_fpz_cz__eog_horizontal", "eeg_fpz_cz__emg_submental", "eeg_fpz_cz__eeg_pz_oz"]}
check("La mejora de REM viene del EOG: con Fpz-Cz + EOG ya es 82.1 %; con Fpz-Cz + EMG 64.2 % y con Fpz-Cz + Pz-Oz 63.6 %",
      f"Fpz-Cz + EOG {rem['eeg_fpz_cz__eog_horizontal']:.1f} %; Fpz-Cz + EMG {rem['eeg_fpz_cz__emg_submental']:.1f} %; "
      f"Fpz-Cz + Pz-Oz {rem['eeg_fpz_cz__eeg_pz_oz']:.1f} %",
      f(*rem.values()) == (82.1, 64.2, 63.6))
check("W (87.8 y 88.6 %) y N3 (93.3 y 91.7 %) se reconocen bien; N3 se confunde sobre todo con N2 (6.7 y 8.0 %); W y N3 casi nunca se confunden entre sí (0.2 % o menos)",
      f"W {a[W, W]:.1f} / {b[W, W]:.1f} %; N3 {a[N3, N3]:.1f} / {b[N3, N3]:.1f} %; N3 a N2 {a[N3, N2]:.1f} / {b[N3, N2]:.1f} %; "
      f"W a N3 {a[W, N3]:.2f} / {b[W, N3]:.2f} %; N3 a W {a[N3, W]:.2f} / {b[N3, W]:.2f} %",
      f(a[W, W], b[W, W], a[N3, N3], b[N3, N3], a[N3, N2], b[N3, N2]) == (87.8, 88.6, 93.3, 91.7, 6.7, 8.0)
      and max(f(a[W, N3], b[W, N3], a[N3, W], b[N3, W])) <= 0.2)

print(f"\n{'Todo coincide.' if fallos == 0 else f'{fallos} afirmación(es) no coinciden: revisar.'}")
