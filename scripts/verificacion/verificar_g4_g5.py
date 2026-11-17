"""Verifica G4 (evaluación en prueba) y G5 (ablación sin LSTM) recalculando TODO desde las predicciones por época.

No vuelve a evaluar el test ni carga modelos: lee las probabilidades del único test que se corrió
(probs_<modelo>__<combinación>.npz) y recalcula las métricas con scikit-learn, de forma independiente
de tabla_test.csv. Los intervalos se recalculan con el mismo bootstrap por persona del script 08.

Uso, desde la raíz del repositorio:
    python scripts/verificacion/verificar_g4_g5.py --probs resultados_test
    python scripts/verificacion/verificar_g4_g5.py --probs "..." --mlflow     (además revisa las columnas de VC de la Tabla 6)

La carpeta debe tener resultados_test.json y los 29 probs_*.npz (6 flexible, 12 EMG-MLP, 6 sin LSTM, 5 dedicados).
"""
import argparse
import getpass
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score

E, P, EO, M = "eeg_fpz_cz", "eeg_pz_oz", "eog_horizontal", "emg_submental"
NOM = {E: "Fpz-Cz", P: "Pz-Oz", f"{E}__{P}": "Fpz-Cz + Pz-Oz", f"{E}__{EO}": "Fpz-Cz + EOG", f"{P}__{EO}": "Pz-Oz + EOG",
       f"{E}__{P}__{EO}": "Fpz-Cz + Pz-Oz + EOG", f"{E}__{M}": "Fpz-Cz + EMG", f"{P}__{M}": "Pz-Oz + EMG",
       f"{E}__{P}__{M}": "Fpz-Cz + Pz-Oz + EMG", f"{E}__{EO}__{M}": "Fpz-Cz + EOG + EMG",
       f"{P}__{EO}__{M}": "Pz-Oz + EOG + EMG", f"{E}__{P}__{EO}__{M}": "Las cuatro señales"}
SIN_EMG = list(NOM)[:6]
CTRL = [E, P, f"{E}__{P}", f"{E}__{EO}", f"{E}__{P}__{EO}__{M}"]
ESTADIOS = ["W", "N1", "N2", "N3", "REM"]

# ---- Valores escritos en el documento ----
# Tabla 5: exactitud, F1 macro, IC95 bajo, IC95 alto, exactitud balanceada, kappa, F1 W, N1, N2, N3, REM
T5 = {E: (.799, .749, .717, .778, .776, .726, .910, .508, .841, .775, .710),
      P: (.776, .714, .673, .751, .724, .691, .893, .475, .800, .704, .699),
      f"{E}__{P}": (.806, .756, .723, .787, .780, .735, .923, .513, .839, .772, .736),
      f"{E}__{EO}": (.823, .779, .750, .805, .812, .758, .913, .534, .848, .771, .829),
      f"{P}__{EO}": (.800, .748, .710, .782, .774, .726, .900, .515, .811, .712, .804),
      f"{E}__{P}__{EO}": (.824, .781, .752, .807, .811, .761, .920, .545, .844, .762, .833),
      f"{E}__{M}": (.806, .758, .726, .788, .780, .734, .912, .517, .844, .785, .729),
      f"{P}__{M}": (.782, .720, .679, .757, .723, .699, .893, .479, .809, .702, .716),
      f"{E}__{P}__{M}": (.814, .766, .736, .796, .786, .745, .925, .523, .844, .784, .755),
      f"{E}__{EO}__{M}": (.826, .784, .756, .809, .812, .762, .915, .537, .851, .780, .835),
      f"{P}__{EO}__{M}": (.804, .755, .715, .788, .776, .731, .898, .518, .817, .717, .823),
      f"{E}__{P}__{EO}__{M}": (.827, .784, .756, .808, .810, .764, .922, .543, .847, .770, .836)}
# Tabla 5b: F1 macro [IC95] de flexible, sin LSTM y dedicado (el EMG-MLP ya está en la Tabla 5)
T5B = {"flexible": {E: (.749, .718, .777), P: (.700, .657, .738), f"{E}__{P}": (.767, .738, .792),
                    f"{E}__{EO}": (.764, .734, .792), f"{P}__{EO}": (.735, .704, .762), f"{E}__{P}__{EO}": (.769, .740, .795)},
       "flexible_sin_lstm": {E: (.718, .687, .747), P: (.667, .632, .700), f"{E}__{P}": (.738, .710, .765),
                             f"{E}__{EO}": (.742, .708, .773), f"{P}__{EO}": (.710, .663, .749), f"{E}__{P}__{EO}": (.749, .720, .779)},
       "control": {E: (.764, .734, .791), P: (.720, .686, .754), f"{E}__{P}": (.768, .744, .791),
                   f"{E}__{EO}": (.764, .735, .792), f"{E}__{P}__{EO}__{M}": (.771, .745, .794)}}
# Tabla 5c: diferencia en puntos (dedicado - EMG-MLP), IC95 pareado, personas donde gana el dedicado
T5C = {E: (1.5, 0.5, 2.4, 13), P: (0.6, -0.7, 2.0, 8), f"{E}__{P}": (1.1, -0.2, 2.5, 8),
       f"{E}__{EO}": (-1.5, -2.6, -0.3, 4), f"{E}__{P}__{EO}__{M}": (-1.2, -2.8, 0.2, 4)}
# Tabla 6: VC con, VC sin, delta VC, prueba con, prueba sin, delta prueba
T6 = {E: (.753, .706, .046, .749, .718, .031), P: (.667, .612, .056, .700, .667, .033),
      f"{E}__{P}": (.754, .720, .034, .767, .738, .029), f"{E}__{EO}": (.762, .732, .030, .764, .742, .022),
      f"{P}__{EO}": (.725, .683, .043, .735, .710, .026), f"{E}__{P}__{EO}": (.767, .741, .026, .769, .749, .020)}
T6_MEDIA = (.738, .699, .039, .747, .721, .027)

p = argparse.ArgumentParser()
p.add_argument("--probs", required=True, help="carpeta con resultados_test.json y los 29 probs_*.npz")
p.add_argument("--mlflow", action="store_true", help="revisar también las columnas de validación cruzada de la Tabla 6")
p.add_argument("--n-boot", type=int, default=1000)
args = p.parse_args()
car = Path(args.probs)

fallos = 0
def check(afirmacion, observado, ok):
    global fallos
    fallos += not ok
    print(f"[{'OK ' if ok else 'NO '}] {afirmacion}\n       observado: {observado}")

r3 = lambda x: float(f"{x:.3f}")          # redondeo como en la tabla
r1 = lambda x: float(f"{x:.1f}")

# ---------------- Carga ----------------
R = json.load(open(car / "resultados_test.json", encoding="utf-8"))
grab = [str(g) for g in R["grabaciones"]]
suj_rec = np.array([g[3:5] for g in grab])          # SC4ssN: ss = código de la persona
archivos = {}
for c in SIN_EMG:
    archivos[("flexible", c)] = f"probs_flexible__{c}.npz"
    archivos[("flexible_sin_lstm", c)] = f"probs_flexible_sin_lstm__{c}.npz"
for c in NOM:
    archivos[("flexible_emg_mlp", c)] = f"probs_flexible_emg_mlp__{c}.npz"
for c in CTRL:
    archivos[("control", c)] = f"probs_control_{c}__{c}.npz"
D = {k: np.load(car / f) for k, f in archivos.items()}
ref = D[("flexible_emg_mlp", E)]
y, rec, ep = ref["y_true"], ref["rec_idx"], ref["epoca_idx"]
suj = suj_rec[rec]
pred = {k: d["probs"].argmax(1) for k, d in D.items()}

# ---------------- Métricas independientes ----------------
def metricas(k):
    yp = pred[k]
    f1c = f1_score(y, yp, labels=range(5), average=None, zero_division=0)
    return dict(acc=accuracy_score(y, yp), f1=f1c.mean(), bal=balanced_accuracy_score(y, yp),
                kappa=cohen_kappa_score(y, yp), clase=f1c)

def bootstrap(yp, n_boot=args.n_boot, semilla=42):
    """Igual que intervalo_bootstrap de scripts/08_evaluar_test.py: remuestrea personas con reemplazo."""
    personas = np.unique(suj)
    cms = np.zeros((len(personas), 5, 5))
    for i, s in enumerate(personas):
        m = suj == s
        np.add.at(cms[i], (y[m], yp[m]), 1)
    idx = np.random.default_rng(semilla).integers(0, len(personas), size=(n_boot, len(personas)))
    cm = cms[idx].sum(1)
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    den = 2 * tp + (cm.sum(-2) - tp) + (cm.sum(-1) - tp)
    f1 = np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0).mean(-1)
    return np.percentile(f1, 2.5), np.percentile(f1, 97.5)

MET = {k: metricas(k) for k in D}
IC = {k: bootstrap(pred[k]) for k in D}

# ---------------- G4. Protocolo ----------------
print("== G4 · Protocolo ==")
iguales = all(np.array_equal(d["y_true"], y) and np.array_equal(d["rec_idx"], rec) and np.array_equal(d["epoca_idx"], ep)
              for d in D.values())
check("Los 8 modelos (29 archivos) se evalúan sobre exactamente las mismas épocas",
      f"{len(D)} archivos; mismas etiquetas, grabación y número de época en todos: {iguales}", iguales and len(D) == 29)
n_suj, n_rec = len(np.unique(suj)), len(np.unique(rec))
check("16 personas de prueba y 32 grabaciones (dos noches por persona)",
      f"{n_suj} personas, {n_rec} grabaciones; personas con 2 noches: {sum((suj_rec == s).sum() == 2 for s in np.unique(suj_rec))}",
      n_suj == 16 and n_rec == 32 and len(grab) == 32)
por_rec = np.bincount(rec)
bloques = all(por_rec[i] % 15 == 0 and np.array_equal(ep[rec == i], np.arange(por_rec[i])) for i in range(n_rec))
tot, ev = R["protocolo"]["n_epocas_totales"], R["protocolo"]["n_epocas_evaluadas"]
check("Bloques de 15 épocas sin traslape desde el inicio de cada noche; se evalúan 44,985 de 45,207 (222 menos)",
      f"{len(y):,} épocas evaluadas; cada noche es múltiplo de 15 y contigua desde la época 0: {bloques}; "
      f"JSON: {tot:,} totales, {ev:,} evaluadas, descartadas {tot - ev} (máximo posible 32 x 14 = {32 * 14})",
      len(y) == 44985 == ev and tot == 45207 and bloques and tot - ev <= 32 * 14)
probs_ok = all(np.allclose(d["probs"].sum(1), 1, atol=1e-4) for d in D.values())
check("Solo 5 clases evaluadas (la M no está en los datos) y las probabilidades suman 1",
      f"etiquetas presentes: {sorted(set(y.tolist()))}; probabilidades suman 1 en todos: {probs_ok}",
      set(y.tolist()) <= {0, 1, 2, 3, 4} and probs_ok)
b = R["protocolo"]["bootstrap"]
check("Intervalos: bootstrap por persona, 1,000 remuestreos, semilla 42", b, b["n"] == 1000 and b["semilla"] == 42)

# ---------------- G4. Tabla 5 ----------------
print("\n== G4 · Tabla 5 (EMG-MLP, 12 combinaciones, 11 valores por fila) ==")
for c, esp in T5.items():
    m, ic = MET[("flexible_emg_mlp", c)], IC[("flexible_emg_mlp", c)]
    obs = (m["acc"], m["f1"], ic[0], ic[1], m["bal"], m["kappa"], *m["clase"])
    malos = [n for n, o, e in zip(["exact", "F1", "IC bajo", "IC alto", "exact bal", "kappa"] + ESTADIOS, obs, esp)
             if r3(o) != e]
    check(f"Tabla 5, {NOM[c]}", " ".join(f"{o:.3f}" for o in obs) + (f"  <- difieren: {malos}" if malos else ""), not malos)

# ---------------- G4. Tabla 5b ----------------
print("\n== G4 · Tabla 5b (F1 macro [IC95] de los otros modelos) ==")
for mod, filas in T5B.items():
    for c, esp in filas.items():
        obs = (MET[(mod, c)]["f1"], *IC[(mod, c)])
        ok = tuple(r3(o) for o in obs) == esp
        check(f"Tabla 5b, {mod}, {NOM[c]}: {esp[0]:.3f} [{esp[1]:.3f}, {esp[2]:.3f}]",
              f"{obs[0]:.3f} [{obs[1]:.3f}, {obs[2]:.3f}]", ok)

# ---------------- G4. Lectura ----------------
print("\n== G4 · Lectura ==")
emg = {c: MET[("flexible_emg_mlp", c)] for c in NOM}
mejor = max(emg, key=lambda c: emg[c]["f1"])
m = emg[mejor]
check("La mejor combinación es Fpz-Cz + EOG + EMG con el EMG-MLP (F1 0.784, κ 0.762, exactitud 0.826)",
      f"{NOM[mejor]}: F1 {m['f1']:.4f}, κ {m['kappa']:.4f}, exactitud {m['acc']:.4f}",
      mejor == f"{E}__{EO}__{M}" and (r3(m["f1"]), r3(m["kappa"]), r3(m["acc"])) == (.784, .762, .826))
c4 = emg[f"{E}__{P}__{EO}__{M}"]["f1"]
check("Con las cuatro señales queda igual (0.784)", f"{c4:.4f} (diferencia con la mejor {m['f1'] - c4:.4f})", r3(c4) == .784)
med = {mod: np.mean([MET[(mod, c)]["f1"] for c in SIN_EMG]) for mod in ["flexible_emg_mlp", "flexible", "flexible_sin_lstm"]}
check("F1 macro medio en las 6 combinaciones comunes en prueba: EMG-MLP 0.755, flexible 0.747 "
      "(VC 0.745 y 0.738, verificado en G3); se mantiene el orden del criterio de selección",
      f"EMG-MLP {med['flexible_emg_mlp']:.4f}, flexible {med['flexible']:.4f}, sin LSTM {med['flexible_sin_lstm']:.4f}",
      r3(med["flexible_emg_mlp"]) == .755 and r3(med["flexible"]) == .747 and med["flexible_emg_mlp"] > med["flexible"])
dif = {c: 100 * (MET[("control", c)]["f1"] - emg[c]["f1"]) for c in CTRL}
solo_eeg = [dif[c] for c in CTRL[:3]]
check("Dedicado arriba cuando solo hay EEG: 0.6 a 1.5 puntos (Fpz-Cz, Pz-Oz o ambos)",
      ", ".join(f"{NOM[c]} {dif[c]:+.2f}" for c in CTRL[:3]),
      all(d > 0 for d in solo_eeg) and r1(min(solo_eeg)) == 0.6 and r1(max(solo_eeg)) == 1.5)
check("EMG-MLP arriba cuando hay EOG: 1.5 puntos con Fpz-Cz + EOG y 1.2 con las cuatro señales",
      f"Fpz-Cz + EOG {dif[CTRL[3]]:+.2f}, cuatro señales {dif[CTRL[4]]:+.2f}",
      r1(-dif[CTRL[3]]) == 1.5 and r1(-dif[CTRL[4]]) == 1.2)
solapan = all(max(IC[("control", c)][0], IC[("flexible_emg_mlp", c)][0]) < min(IC[("control", c)][1], IC[("flexible_emg_mlp", c)][1])
              for c in CTRL)
check("Los intervalos de cada modelo (dedicado y EMG-MLP) se solapan en las 5 combinaciones", solapan, solapan)
con_lstm = [MET[k]["clase"] for k in D if k[0] != "flexible_sin_lstm"]
sin_lstm = [MET[k]["clase"] for k in D if k[0] == "flexible_sin_lstm"]
n1c, n1s = [v[1] for v in con_lstm], [v[1] for v in sin_lstm]
w = [MET[k]["clase"][0] for k in D]
dificil = all(np.argmin(MET[k]["clase"]) == 1 for k in D)
facil = all(np.argmax(MET[k]["clase"]) == 0 for k in D)
check("N1 es el estadio más difícil (0.458–0.545 con LSTM; 0.362–0.450 sin LSTM) y W el más fácil (0.870–0.925), en los 29 casos",
      f"N1 con LSTM {min(n1c):.3f}–{max(n1c):.3f}; sin LSTM {min(n1s):.3f}–{max(n1s):.3f}; W {min(w):.3f}–{max(w):.3f}; "
      f"N1 el peor en todos: {dificil}; W el mejor en todos: {facil}",
      dificil and facil and (r3(min(n1c)), r3(max(n1c))) == (.458, .545)
      and (r3(min(n1s)), r3(max(n1s))) == (.362, .450) and (r3(min(w)), r3(max(w))) == (.870, .925))

# ---------------- G4. Tabla 5c (bootstrap pareado) ----------------
print("\n== G4 · Tabla 5c (bootstrap pareado por persona; mismo procedimiento que verificar_g1_comparacion.py) ==")
def f1m(yy, pp):
    return f1_score(yy, pp, average="macro", labels=[0, 1, 2, 3, 4], zero_division=0)

def f1_persona(yy, pp):          # como metricas_por_persona de 08: promedio sobre las clases con soporte
    return f1_score(yy, pp, average="macro", labels=np.unique(yy), zero_division=0)

rng = np.random.default_rng(42)
personas = np.unique(suj)
idx = {s: np.flatnonzero(suj == s) for s in personas}
for c in CTRL:
    pa, pb = pred[("control", c)], pred[("flexible_emg_mlp", c)]
    obs = 100 * (f1m(y, pa) - f1m(y, pb))
    boots = [100 * (f1m(y[s], pa[s]) - f1m(y[s], pb[s]))
             for s in (np.concatenate([idx[q] for q in rng.choice(personas, size=len(personas), replace=True)])
                       for _ in range(args.n_boot))]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    gana = sum(f1_persona(y[idx[s]], pa[idx[s]]) > f1_persona(y[idx[s]], pb[idx[s]]) for s in personas)
    esp = T5C[c]
    ok = (r1(obs), r1(lo), r1(hi), gana) == esp
    lect = "incluye 0" if lo <= 0 <= hi else ("mejor el dedicado" if lo > 0 else "mejor el EMG-MLP")
    check(f"Tabla 5c, {NOM[c]}: {esp[0]:+.1f} [{esp[1]:+.1f}, {esp[2]:+.1f}], dedicado arriba en {esp[3]} de 16",
          f"{obs:+.2f} [{lo:+.2f}, {hi:+.2f}], dedicado arriba en {gana} de 16 personas ({lect})", ok)

# ---------------- G5. Tabla 6 (prueba) y lectura ----------------
print("\n== G5 · Tabla 6, columnas de prueba ==")
dt, dclase = {}, {}
for c in SIN_EMG:
    con, sin = MET[("flexible", c)], MET[("flexible_sin_lstm", c)]
    dt[c] = con["f1"] - sin["f1"]
    dclase[c] = con["clase"] - sin["clase"]
    esp = T6[c]
    ok = (r3(con["f1"]), r3(sin["f1"]), r3(dt[c])) == esp[3:]
    check(f"Tabla 6, {NOM[c]}: prueba {esp[3]:.3f} / {esp[4]:.3f} / {esp[5]:+.3f}",
          f"{con['f1']:.4f} / {sin['f1']:.4f} / {dt[c]:+.4f}", ok)
mc, ms = med["flexible"], med["flexible_sin_lstm"]
check(f"Tabla 6, media de prueba: {T6_MEDIA[3]:.3f} / {T6_MEDIA[4]:.3f} / {T6_MEDIA[5]:+.3f}",
      f"{mc:.4f} / {ms:.4f} / {mc - ms:+.4f}", (r3(mc), r3(ms), r3(mc - ms)) == T6_MEDIA[3:])

print("\n== G5 · Lectura (prueba) ==")
v = list(dt.values())
check("Sin la LSTM el F1 macro baja en las 6 combinaciones, 0.020 a 0.033 en prueba",
      f"{min(v):+.4f} a {max(v):+.4f}", all(x > 0 for x in v) and (r3(min(v)), r3(max(v))) == (.020, .033))
rango = lambda j: (min(d[j] for d in dclase.values()), max(d[j] for d in dclase.values()))
n1, rem, wr, n3 = rango(1), rango(4), rango(0), rango(3)
check("Lo que más se pierde es N1 (0.050 a 0.107 menos) y REM (0.034 a 0.074 menos)",
      f"N1 {n1[0]:+.4f} a {n1[1]:+.4f}; REM {rem[0]:+.4f} a {rem[1]:+.4f} (positivo = mejor con LSTM)",
      (r3(n1[0]), r3(n1[1]), r3(rem[0]), r3(rem[1])) == (.050, .107, .034, .074))
check("W queda igual o ligeramente mejor sin LSTM (hasta 0.027) y N3 algo mejor con LSTM (hasta 0.035)",
      f"W {wr[0]:+.4f} a {wr[1]:+.4f}; N3 {n3[0]:+.4f} a {n3[1]:+.4f} (positivo = mejor con LSTM)",
      r3(wr[0]) == -.027 and r3(wr[1]) <= 0.0005 and r3(n3[1]) == .035 and n3[0] > 0)
solap = [NOM[c] for c in SIN_EMG if IC[("flexible_sin_lstm", c)][1] > IC[("flexible", c)][0]]
check("En prueba los intervalos de confianza (con y sin LSTM) se solapan en parte",
      f"se solapan en {len(solap)} de 6: {solap}", len(solap) > 0)

# ---------------- Opcional: columnas de VC de la Tabla 6 ----------------
if args.mlflow:
    print("\n== G5 · Tabla 6, columnas de validación cruzada (MLflow) ==")
    import mlflow
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        os.environ["MLFLOW_TRACKING_URI"] = input("MLFLOW_TRACKING_URI: ").strip()
        u = input("MLFLOW_TRACKING_USERNAME (vacío si no hace falta): ").strip()
        if u:
            os.environ["MLFLOW_TRACKING_USERNAME"] = u
            os.environ["MLFLOW_TRACKING_PASSWORD"] = getpass.getpass("MLFLOW_TRACKING_PASSWORD: ")
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    tmp = Path(tempfile.mkdtemp(prefix="g5_"))
    cv = {}
    for mod, rid in [("flexible", "029cc677c3734b10959735a6e9246cb8"), ("flexible_sin_lstm", "c424e04708c34f04a5bfec810b705982")]:
        d = Path(mlflow.artifacts.download_artifacts(run_id=rid, artifact_path="resultados", dst_path=str(tmp / mod)))
        pl = [json.load(open(a, encoding="utf-8")) for a in sorted(d.glob(f"{mod}_pliegue[0-9].json"))]
        cv[mod] = {c: np.mean([x["metricas_por_combinacion"][c]["macro_f1"] for x in pl]) for c in SIN_EMG}
    dv = {}
    for c in SIN_EMG:
        con, sin = cv["flexible"][c], cv["flexible_sin_lstm"][c]
        dv[c] = con - sin
        check(f"Tabla 6, {NOM[c]}: VC {T6[c][0]:.3f} / {T6[c][1]:.3f} / {T6[c][2]:+.3f}",
              f"{con:.4f} / {sin:.4f} / {dv[c]:+.4f}", (r3(con), r3(sin), r3(dv[c])) == T6[c][:3])
    mc, ms = np.mean(list(cv["flexible"].values())), np.mean(list(cv["flexible_sin_lstm"].values()))
    check(f"Tabla 6, media de VC: {T6_MEDIA[0]:.3f} / {T6_MEDIA[1]:.3f} / {T6_MEDIA[2]:+.3f}",
          f"{mc:.4f} / {ms:.4f} / {mc - ms:+.4f}", (r3(mc), r3(ms), r3(mc - ms)) == T6_MEDIA[:3])
    v = list(dv.values())
    check("Sin la LSTM el F1 macro baja en las 6 combinaciones, 0.026 a 0.056 en VC; misma dirección en VC y prueba",
          f"{min(v):+.4f} a {max(v):+.4f}", all(x > 0 for x in v) and (r3(min(v)), r3(max(v))) == (.026, .056))
else:
    print("\n(Columnas de VC de la Tabla 6 omitidas: añade --mlflow para revisarlas.)")

print(f"\n{'Todo coincide.' if fallos == 0 else f'{fallos} afirmación(es) no coinciden: revisar.'}")
