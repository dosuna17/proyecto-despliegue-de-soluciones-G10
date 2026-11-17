"""Verifica G9: cómo entra el EMG submental y la clase M como limitación.

Parte A. La señal: cabeceras de los EDF crudos de Sleep-EDF (sin librerías externas) y los .npz del EMG.
Parte B. El modelo: la CNN no ve el EMG; la rama recibe los 30 valores originales (necesita torch y el repo).
Parte C. Cuánto aporta el EMG en prueba: EMG-MLP con y sin EMG sobre las mismas épocas, bootstrap pareado
         por persona (no vuelve a evaluar el test: usa las predicciones guardadas).
Parte D. La clase M: cuántas épocas hay en los hipnogramas y que no llegan a los datos evaluados.

Uso, desde la raíz del repositorio:
    python scripts/verificacion/verificar_g9.py --probs resultados_test ^
           --emg-dir "C:\\Users\\albam\\Desktop\\experimento_emg\\datos_emg\\epochs_emg_recortado"
"""
import argparse
import collections
import inspect
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

RAIZ = Path.cwd()
CLASES = ["W", "N1", "N2", "N3", "REM"]
TEST = ["04", "07", "17", "19", "21", "24", "31", "34", "40", "56", "58", "65", "66", "71", "74", "81"]
BASES = ["eeg_fpz_cz", "eeg_pz_oz", "eeg_fpz_cz__eeg_pz_oz", "eeg_fpz_cz__eog_horizontal",
         "eeg_pz_oz__eog_horizontal", "eeg_fpz_cz__eeg_pz_oz__eog_horizontal"]
NOM = {"eeg_fpz_cz": "Fpz-Cz", "eeg_pz_oz": "Pz-Oz", "eeg_fpz_cz__eeg_pz_oz": "Fpz-Cz + Pz-Oz",
       "eeg_fpz_cz__eog_horizontal": "Fpz-Cz + EOG", "eeg_pz_oz__eog_horizontal": "Pz-Oz + EOG",
       "eeg_fpz_cz__eeg_pz_oz__eog_horizontal": "Fpz-Cz + Pz-Oz + EOG"}

p = argparse.ArgumentParser()
p.add_argument("--probs", required=True, help="carpeta con resultados_test.json y los probs_*.npz")
p.add_argument("--emg-dir", default=os.environ.get("SLEEP_EMG_DIR", "data/interim/epochs_emg_recortado"))
p.add_argument("--edf-dir", default="data/sleep-edfx/sleep-cassette")
p.add_argument("--npz3-dir", default="data/interim/epochs_por_sujeto_3canales_recortado")
p.add_argument("--sin-modelo", action="store_true", help="saltar la parte B (si no hay torch)")
args = p.parse_args()

fallos = 0
def check(afirmacion, observado, ok):
    global fallos
    fallos += not ok
    print(f"[{'OK ' if ok else 'NO '}] {afirmacion}\n       observado: {observado}")

r1 = lambda x: float(f"{x:.1f}")

# ---------------- Parte A: la señal ----------------
def cabecera_edf(ruta):
    """Lee la cabecera EDF: etiqueta, unidad física y frecuencia de muestreo de cada señal."""
    with open(ruta, "rb") as f:
        h = f.read(256)
        ns, dur = int(h[252:256]), float(h[244:252])
        s = f.read(ns * 256)
    campo = lambda ini, ancho: [s[ini + i * ancho: ini + (i + 1) * ancho].decode("latin-1").strip() for i in range(ns)]
    etiquetas = campo(0, 16)
    unidades = campo(ns * (16 + 80), 8)
    off = ns * (16 + 80 + 8 + 8 + 8 + 8 + 8 + 80)
    muestras = [int(x) for x in campo(off, 8)]
    return {e: (u, m / dur) for e, u, m in zip(etiquetas, unidades, muestras)}

print("== G9 · Parte A. La señal del EMG ==")
edf = Path(args.edf_dir)
psg = sorted(edf.glob("*-PSG.edf"))
cab = {f.name[:6]: cabecera_edf(f) for f in psg}
emg = {r: c.get("EMG submental") for r, c in cab.items()}
eeg = {r: (c.get("EEG Fpz-Cz"), c.get("EEG Pz-Oz"), c.get("EOG horizontal")) for r, c in cab.items()}
check("En los EDF de Sleep-EDF (153 grabaciones) el EMG submental está a 1 Hz y en µV; EEG y EOG a 100 Hz",
      f"{len(psg)} PSG; EMG: {collections.Counter(emg.values())}; EEG/EOG a 100 Hz en todas: "
      f"{all(all(x and x[1] == 100 for x in v) for v in eeg.values())}",
      len(psg) == 153 and set(emg.values()) == {("uV", 1.0)} and all(all(x and x[1] == 100 for x in v) for v in eeg.values()))

ed = Path(args.emg_dir)
fe = sorted(ed.glob("*_epochs.npz"))
formas, sf, un, iguales, n, neg, tot, por = set(), set(), set(), True, 0, 0, 0, collections.defaultdict(list)
for f in fe:
    d = np.load(f, allow_pickle=True)
    X, y = d["X"], d["y"]
    formas.add(X.shape[1:]); sf.add(float(d["sfreq"])); un.add(str(d["signal_unit"]))
    y3 = np.load(Path(args.npz3_dir) / f.name, allow_pickle=True)["y"]
    iguales &= bool(np.array_equal(y, y3))
    n += len(y); neg += int((X < 0).sum()); tot += X.size
    m = X[:, 0, :].mean(1)
    for c in CLASES:
        por[c].append(m[y == c])
check("Los .npz del EMG: 153 grabaciones, 30 valores por época (1 Hz, µV), mismas épocas y etiquetas que los datos de 3 canales; 195,479 épocas",
      f"{len(fe)} archivos; forma por época {formas}; sfreq {sf}; unidad {un}; etiquetas iguales: {iguales}; {n:,} épocas",
      len(fe) == 153 and formas == {(1, 30)} and sf == {1.0} and un == {"uV"} and iguales and n == 195479)
med = {c: float(np.median(np.concatenate(por[c]))) for c in CLASES}
check("El EMG baja en el sueño y sobre todo en REM (atonía): mediana por época W 3.1 µV, N2 1.9, REM 0.8",
      ", ".join(f"{c} {med[c]:.2f}" for c in CLASES),
      (r1(med["W"]), r1(med["N2"]), r1(med["REM"])) == (3.1, 1.9, 0.8) and min(med, key=med.get) == "REM" and max(med, key=med.get) == "W")
check("Es una envolvente (no la señal cruda): valores de pocos µV y un 4.2 % ligeramente negativos (desplazamiento de la calibración)",
      f"{100 * neg / tot:.1f} % de valores < 0", r1(100 * neg / tot) == 4.2)

# ---------------- Parte B: el modelo ----------------
print("\n== G9 · Parte B. Cómo entra el EMG en el modelo ==")
if args.sin_modelo:
    print("   (omitida con --sin-modelo)")
else:
    import torch
    sys.path.insert(0, str(RAIZ / "src"))
    from sleep_staging.models import tinysleepnet_emg as te
    from sleep_staging.models.tinysleepnet import TinySleepNetConfig
    from sleep_staging.training import emg_datos, canales_flexibles as cf
    torch.manual_seed(0)
    m = te.TinySleepNetEMG(TinySleepNetConfig(use_rnn=True, input_channels=8, seq_length=15), rama="mlp").eval()
    cnn_in = m.cnn.block1.conv.in_channels if hasattr(m.cnn.block1, "conv") else next(
        x for x in m.cnn.modules() if isinstance(x, torch.nn.Conv1d)).in_channels
    check("La CNN recibe 6 canales (Fpz-Cz, Pz-Oz, EOG y sus 3 marcas): el EMG y su marca no entran a la CNN",
          f"canales de entrada de la primera convolución: {cnn_in}; índices que van a la CNN: {te.IDX_CNN}", cnn_in == 6 and 3 not in te.IDX_CNN)
    x = torch.randn(2, 15, 8, 3000)
    vals = torch.rand(2, 15, 30) * 5
    x[:, :, 3] = vals.repeat_interleave(100, dim=-1)
    x[:, :, 7] = 1.0
    capt = {}
    m.cnn.register_forward_hook(lambda mod, i, o: capt.__setitem__("cnn", o.detach().clone()))
    m.emg_branch.register_forward_hook(lambda mod, i, o: capt.__setitem__("rama_in", i[0].detach().clone()))
    with torch.no_grad():
        out1 = m(x)[0]
        f1_ = capt["cnn"]; rin = capt["rama_in"]
        x2 = x.clone(); x2[:, :, 3] = 99.0
        out2 = m(x2)[0]; f2_ = capt["cnn"]
    check("Cambiar el EMG no cambia lo que produce la CNN, pero sí la salida del modelo",
          f"CNN igual: {torch.equal(f1_, f2_)}; salida distinta: {not torch.equal(out1, out2)}",
          torch.equal(f1_, f2_) and not torch.equal(out1, out2))
    check("La rama recibe exactamente los 30 valores originales: el EMG viaja repetido 100 veces en el tensor solo para "
          "tener el mismo largo (3000) y la rama lo recupera con [::100]; no se interpola ni se remuestrea",
          f"iguales a los originales: {torch.equal(rin, vals.reshape(-1, 30))}; forma {tuple(rin.shape)}",
          torch.equal(rin, vals.reshape(-1, 30)))
    n_rama = sum(p_.numel() for p_ in m.emg_branch.parameters())
    capas = [l for l in m.emg_branch.red if isinstance(l, torch.nn.Linear)]
    check("Rama MLP: Linear(31, 32) - ReLU - Linear(32, 16) - ReLU (30 valores + la marca); 1,552 parámetros; su salida (16) se une a la de la CNN antes de la LSTM",
          f"capas {[(l.in_features, l.out_features) for l in capas]}; {n_rama:,} parámetros; entrada de la LSTM {m.lstm.input_size} = {m.cnn_out_dim} + 16",
          [(l.in_features, l.out_features) for l in capas] == [(31, 32), (32, 16)] and n_rama == 1552
          and m.lstm.input_size == m.cnn_out_dim + 16)
    src = inspect.getsource(emg_datos.cargar_con_emg)
    norm = any(t in src for t in (".mean(", ".std(", "zscore", "normaliz"))
    check("El EMG entra en µV, sin normalizar (igual que EEG y EOG)",
          f"normalización en cargar_con_emg: {norm}; repetición: {'np.repeat(E, REPETIR' in src}", not norm)

# ---------------- Parte C: aporte del EMG en prueba ----------------
print("\n== G9 · Parte C. Aporte del EMG en prueba (EMG-MLP con y sin EMG, mismas épocas) ==")
R = json.load(open(Path(args.probs) / "resultados_test.json", encoding="utf-8"))
suj_rec = np.array([str(g)[3:5] for g in R["grabaciones"]])
f1m = lambda y, p_: f1_score(y, p_, average="macro", labels=range(5), zero_division=0)
rng = np.random.default_rng(42)
res = {}
for c in BASES:
    a = np.load(Path(args.probs) / f"probs_flexible_emg_mlp__{c}.npz")
    b = np.load(Path(args.probs) / f"probs_flexible_emg_mlp__{c}__emg_submental.npz")
    assert np.array_equal(a["y_true"], b["y_true"]) and np.array_equal(a["rec_idx"], b["rec_idx"])
    y, pa, pb, s = a["y_true"], a["probs"].argmax(1), b["probs"].argmax(1), suj_rec[a["rec_idx"]]
    P = np.unique(s)
    idx = {q: np.flatnonzero(s == q) for q in P}
    d = 100 * (f1m(y, pb) - f1m(y, pa))
    bo = []
    for _ in range(1000):
        sel = np.concatenate([idx[q] for q in rng.choice(P, size=len(P), replace=True)])
        bo.append(100 * (f1m(y[sel], pb[sel]) - f1m(y[sel], pa[sel])))
    lo, hi = np.percentile(bo, [2.5, 97.5])
    pc = 100 * (f1_score(y, pb, average=None, labels=range(5)) - f1_score(y, pa, average=None, labels=range(5)))
    res[c] = dict(d=d, lo=lo, hi=hi, pc=pc)
    print(f"   {NOM[c]:22s} {d:+.2f} puntos [{lo:+.2f}, {hi:+.2f}] | por estadio " +
          " ".join(f"{k} {v:+.1f}" for k, v in zip(CLASES, pc)))
ds = [res[c]["d"] for c in BASES]
check("Añadir el EMG sube el F1 macro en las 6 combinaciones, de +0.3 a +1.0 puntos",
      f"{min(ds):+.2f} a {max(ds):+.2f}", all(x > 0 for x in ds) and (r1(min(ds)), r1(max(ds))) == (0.3, 1.0))
claras = [NOM[c] for c in BASES if res[c]["lo"] > 0]
check("Con bootstrap pareado por persona, la mejora es clara (IC95 sin el 0) en 4 de 6: Fpz-Cz, Fpz-Cz + Pz-Oz, Fpz-Cz + EOG y Pz-Oz + EOG",
      claras, claras == ["Fpz-Cz", "Fpz-Cz + Pz-Oz", "Fpz-Cz + EOG", "Pz-Oz + EOG"])
rem_max = [NOM[c] for c in BASES if np.argmax(res[c]["pc"]) == 4]
check("Donde más gana es REM (+1.7 a +1.9 puntos de F1) en Fpz-Cz, Pz-Oz, Fpz-Cz + Pz-Oz y Pz-Oz + EOG; con Fpz-Cz + EOG solo +0.6",
      f"REM es el que más gana en: {rem_max}; REM: " + ", ".join(f"{NOM[c]} {res[c]['pc'][4]:+.1f}" for c in BASES),
      rem_max == ["Fpz-Cz", "Pz-Oz", "Fpz-Cz + Pz-Oz", "Pz-Oz + EOG"]
      and (r1(min(res[c]["pc"][4] for c in BASES if NOM[c] in rem_max)), r1(max(res[c]["pc"][4] for c in BASES if NOM[c] in rem_max))) == (1.7, 1.9)
      and r1(res["eeg_fpz_cz__eog_horizontal"]["pc"][4]) == 0.6)

# ---------------- Parte D: la clase M ----------------
print("\n== G9 · Parte D. La clase M (movimiento) ==")
def anotaciones(ruta):
    """Hipnograma EDF+: suma la duración (en épocas de 30 s) de cada anotación."""
    b = open(ruta, "rb").read()
    ns = int(b[252:256])
    txt = b[256 + ns * 256:].decode("latin-1")
    c = collections.Counter()
    for mm in re.finditer(r"\+(\d+(?:\.\d+)?)\x15(\d+(?:\.\d+)?)\x14([^\x14\x00]+)\x14", txt):
        c[mm.group(3)] += float(mm.group(2)) / 30
    return c
hip = {f.name[:6]: anotaciones(f) for f in sorted(edf.glob("*-Hypnogram.edf"))}
tot_c = sum(hip.values(), collections.Counter())
test_c = sum((c for r, c in hip.items() if r[3:5] in TEST), collections.Counter())
m_tot, m_test = round(tot_c["Movement time"]), round(test_c["Movement time"])
con_m = sum(1 for c in hip.values() if c["Movement time"] > 0)
check("Hay 128 épocas M en las 153 grabaciones (en 45 de ellas) y 28 en las 32 de prueba",
      f"{m_tot} en total, en {con_m} grabaciones; {m_test} en prueba", (m_tot, con_m, m_test) == (128, 45, 28))
check("Los conteos de los hipnogramas cuadran con los datos: N1 21,522, N2 69,132, N3 (etapas 3 + 4) 13,039 y REM 25,835",
      f"N1 {tot_c['Sleep stage 1']:.0f}, N2 {tot_c['Sleep stage 2']:.0f}, N3 {tot_c['Sleep stage 3'] + tot_c['Sleep stage 4']:.0f}, "
      f"REM {tot_c['Sleep stage R']:.0f}",
      (round(tot_c["Sleep stage 1"]), round(tot_c["Sleep stage 2"]), round(tot_c["Sleep stage 3"] + tot_c["Sleep stage 4"]),
       round(tot_c["Sleep stage R"])) == (21522, 69132, 13039, 25835))
check("M es el 0.07 % de las épocas con estadio (128 frente a 195,479) y el 0.06 % en prueba (28 frente a 45,207)",
      f"{100 * m_tot / (195479 + m_tot):.2f} % y {100 * m_test / (45207 + m_test):.2f} %",
      (f"{100 * m_tot / (195479 + m_tot):.2f}", f"{100 * m_test / (45207 + m_test):.2f}") == ("0.07", "0.06"))
y_test = np.load(Path(args.probs) / "probs_flexible_emg_mlp__eeg_fpz_cz.npz")["y_true"]
check("M no llega a los datos evaluados: las etiquetas de prueba solo tienen las 5 clases", sorted(set(y_test.tolist())),
      set(y_test.tolist()) <= {0, 1, 2, 3, 4})

print(f"\n{'Todo coincide.' if fallos == 0 else f'{fallos} afirmación(es) no coinciden: revisar.'}")
