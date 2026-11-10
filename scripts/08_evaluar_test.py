"""
Evaluacion UNICA en el conjunto de test de los modelos finales del modelo flexible.

Modelos (todos ya entrenados, solo con sujetos de train; ver scripts/06_train_flexible.py):
    flexible                        flexible de 3 senales (6 combinaciones)
    flexible_emg_mlp                flexible con EMG en una rama aparte (12 combinaciones)
    control_eeg_fpz_cz              controles: un modelo por combinacion
    control_eeg_pz_oz
    control_eeg_fpz_cz__eeg_pz_oz
    control_eeg_fpz_cz__eog_horizontal

Para cada modelo y cada combinacion de senales:
    - exactitud, F1 macro, exactitud balanceada, kappa, F1 por etapa y matriz de confusion
    - intervalo de confianza del 95 % (bootstrap por sujeto) de F1 macro y kappa
    - curva cobertura vs exactitud segun la confianza (probabilidad maxima) de la prediccion
      y el umbral de confianza mas bajo con exactitud aceptada >= 82.6 %
    - probabilidades por epoca guardadas (.npz), para volver a calcular sin reentrenar

Modos:
    --modo prueba   (por defecto) usa 3 sujetos de TRAIN para comprobar que todo corre. No toca el test.
    --modo test     usa los sujetos de test, solo con --confirmo-test-unico, y deja un archivo
                    TEST_YA_EJECUTADO.json que impide repetir la evaluacion.

Carpeta de modelos (--modelos): debe tener, para cada modelo, <run>_final.pth y <run>_final.json.

Uso:
    python scripts/08_evaluar_test.py --modelos modelos_finales                       # prueba
    python scripts/08_evaluar_test.py --modelos modelos_finales --modo test --confirmo-test-unico

Carpetas de datos: SLEEP_NPZ_DIR (3 canales) y SLEEP_EMG_DIR (EMG, solo para el modelo con EMG).

Nota: el umbral de confianza se fija sobre el propio test, asi que su exactitud es optimista;
hay que declararlo asi en el articulo.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from data import dataset as ds_mod  # noqa: E402
from sleep_staging.models.tinysleepnet import TinySleepNet, TinySleepNetConfig  # noqa: E402
from sleep_staging.training import train_tinysleepnet as base  # noqa: E402
from sleep_staging.training.canales_flexibles import (  # noqa: E402
    POSICIONES,
    POSICIONES_EMG,
    FlexibleSequenceDataset,
    combinacion_por_nombre,
    combinaciones_validas,
    n_canales_entrada,
)

MODELOS = [
    "flexible",
    "flexible_emg_mlp",
    "control_eeg_fpz_cz",
    "control_eeg_pz_oz",
    "control_eeg_fpz_cz__eeg_pz_oz",
    "control_eeg_fpz_cz__eog_horizontal",
]
CLASES = base.CLASS_NAMES
OBJETIVO_EXACTITUD = 0.826          # acuerdo entre expertos (Rosenberg & Van Hout 2013)
N_BOOTSTRAP = 1000
SEMILLA_BOOTSTRAP = 42
UMBRALES = np.round(np.arange(0.0, 1.0, 0.01), 2)


# ---------------------------------------------------------------------
# Carga de modelos
# ---------------------------------------------------------------------
def md5(ruta: Path) -> str:
    h = hashlib.md5()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def cargar_modelo(carpeta: Path, run: str):
    """Reconstruye el modelo desde su JSON y carga los pesos (estricto). Devuelve (modelo, info, posiciones)."""
    ruta_json, ruta_pth = carpeta / f"{run}_final.json", carpeta / f"{run}_final.pth"
    if not ruta_json.exists() or not ruta_pth.exists():
        raise FileNotFoundError(f"Falta {ruta_json.name} o {ruta_pth.name} en {carpeta}")
    info = json.load(open(ruta_json, encoding="utf-8"))
    emg = info.get("emg")                                    # None en los modelos sin EMG
    posiciones = POSICIONES_EMG if emg else POSICIONES
    if info["n_canales_entrada"] != n_canales_entrada(posiciones):
        raise ValueError(f"{run}: n_canales_entrada={info['n_canales_entrada']} no coincide con las posiciones.")
    config = TinySleepNetConfig(use_rnn=True, seq_length=ds_mod.DEFAULT_SEQ_LENGTH,
                                input_channels=info["n_canales_entrada"])
    if emg:
        from sleep_staging.models.tinysleepnet_emg import TinySleepNetEMG
        modelo = TinySleepNetEMG(config, rama=emg)
    else:
        modelo = TinySleepNet(config)
    modelo.load_state_dict(torch.load(ruta_pth, map_location="cpu"), strict=True)
    return modelo.to(base.DEVICE).eval(), info, posiciones


# ---------------------------------------------------------------------
# Prediccion por combinacion
# ---------------------------------------------------------------------
@torch.no_grad()
def predecir(modelo, recs, combinacion, posiciones) -> Dict[str, np.ndarray]:
    """
    Probabilidades por epoca, en el mismo orden y con los mismos bloques de 15 epocas
    (sin solape, descartando el resto) que usan el entrenamiento y la validacion.
    """
    dataset = FlexibleSequenceDataset(recs, [combinacion], seq_length=ds_mod.DEFAULT_SEQ_LENGTH,
                                      posiciones=posiciones)
    loader = DataLoader(dataset, batch_size=ds_mod.DEFAULT_BATCH_SIZE, shuffle=False)
    probs, ys = [], []
    for x, y in loader:
        logits, _ = modelo(x.to(base.DEVICE))
        probs.append(torch.softmax(logits, dim=-1).cpu().numpy().reshape(-1, len(CLASES)))
        ys.append(y.numpy().reshape(-1))
    probs, y_true = np.concatenate(probs).astype(np.float32), np.concatenate(ys)

    # a que grabacion y a que epoca pertenece cada prediccion
    rec_idx, epoca_idx = [], []
    for i, r in enumerate(recs):
        n = (r.X.shape[0] // ds_mod.DEFAULT_SEQ_LENGTH) * ds_mod.DEFAULT_SEQ_LENGTH
        rec_idx.append(np.full(n, i, dtype=np.int32))
        epoca_idx.append(np.arange(n, dtype=np.int32))
    rec_idx, epoca_idx = np.concatenate(rec_idx), np.concatenate(epoca_idx)
    if not (len(rec_idx) == len(y_true) == len(probs)):
        raise RuntimeError("El orden de las predicciones no coincide con las grabaciones.")
    return {"probs": probs, "y_true": y_true, "rec_idx": rec_idx, "epoca_idx": epoca_idx}


# ---------------------------------------------------------------------
# Metricas, intervalos de confianza y curva de cobertura
# ---------------------------------------------------------------------
def _f1_kappa_desde_cm(cm: np.ndarray):
    """cm: (..., 5, 5) con filas = real, columnas = predicho. Devuelve F1 macro y kappa."""
    cm = cm.astype(np.float64)
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    fp = cm.sum(axis=-2) - tp
    fn = cm.sum(axis=-1) - tp
    den = 2 * tp + fp + fn
    f1 = np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0.0)
    n = cm.sum(axis=(-2, -1))
    po = tp.sum(axis=-1) / n
    pe = (cm.sum(axis=-2) * cm.sum(axis=-1)).sum(axis=-1) / n ** 2
    kappa = (po - pe) / (1 - pe)
    return f1.mean(axis=-1), kappa


def intervalo_bootstrap(y_true, y_pred, sujeto_de_epoca, n_boot=N_BOOTSTRAP, semilla=SEMILLA_BOOTSTRAP) -> Dict:
    """IC del 95 % de F1 macro y kappa, remuestreando SUJETOS (no epocas) con reemplazo."""
    sujetos = np.unique(sujeto_de_epoca)
    cms = np.zeros((len(sujetos), 5, 5))
    for i, s in enumerate(sujetos):
        m = sujeto_de_epoca == s
        np.add.at(cms[i], (y_true[m], y_pred[m]), 1)
    rng = np.random.default_rng(semilla)
    idx = rng.integers(0, len(sujetos), size=(n_boot, len(sujetos)))
    f1, kappa = _f1_kappa_desde_cm(cms[idx].sum(axis=1))
    return {"macro_f1_ic95": [float(np.percentile(f1, 2.5)), float(np.percentile(f1, 97.5))],
            "kappa_ic95": [float(np.percentile(kappa, 2.5)), float(np.percentile(kappa, 97.5))],
            "n_sujetos": int(len(sujetos)), "n_bootstrap": n_boot}


def curva_cobertura(probs: np.ndarray, y_true: np.ndarray, objetivo: float = OBJETIVO_EXACTITUD) -> Dict:
    """Para cada umbral t: fraccion de epocas con confianza >= t y exactitud dentro de ellas."""
    confianza, y_pred = probs.max(axis=1), probs.argmax(axis=1)
    acierto = (y_pred == y_true)
    filas = []
    for t in UMBRALES:
        m = confianza >= t
        if m.sum() == 0:
            break
        filas.append({"umbral": float(t), "cobertura": float(m.mean()), "exactitud": float(acierto[m].mean()),
                      "n_aceptadas": int(m.sum())})
    cumple = [f for f in filas if f["exactitud"] >= objetivo]
    return {"objetivo_exactitud": objetivo, "curva": filas,
            "umbral_minimo": cumple[0] if cumple else None}


def evaluar_combinacion(pred: Dict[str, np.ndarray], sujeto_de_rec: np.ndarray) -> Dict:
    y_true, probs = pred["y_true"], pred["probs"]
    y_pred = probs.argmax(axis=1)
    metricas = base.compute_metrics(y_true, y_pred)
    metricas.update(intervalo_bootstrap(y_true, y_pred, sujeto_de_rec[pred["rec_idx"]]))
    metricas["n_epocas"] = int(len(y_true))
    metricas["cobertura_vs_exactitud"] = curva_cobertura(probs, y_true)
    return metricas


# ---------------------------------------------------------------------
# Figuras (PDF)
# ---------------------------------------------------------------------
def figura_matrices(modelo: str, resultados: Dict[str, Dict], ruta: Path, etiqueta: str = "test") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nombres = list(resultados)
    n = len(nombres)
    cols = min(n, 4)
    filas = int(np.ceil(n / cols))
    fig, ejes = plt.subplots(filas, cols, figsize=(4.2 * cols, 4.0 * filas), squeeze=False)
    for ax in ejes.ravel():
        ax.axis("off")
    for ax, nombre in zip(ejes.ravel(), nombres):
        ax.axis("on")
        cm = np.array(resultados[nombre]["confusion_matrix"], dtype=float)
        norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(norm, vmin=0, vmax=1, cmap="Blues")
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{norm[i, j] * 100:.0f}\n({int(cm[i, j])})", ha="center", va="center",
                        fontsize=6, color="white" if norm[i, j] > 0.5 else "black")
        ax.set_xticks(range(5), CLASES, fontsize=7)
        ax.set_yticks(range(5), CLASES, fontsize=7)
        ax.set_xlabel("Predicho", fontsize=7)
        ax.set_ylabel("Real", fontsize=7)
        ax.set_title(f"{nombre}\nF1 macro {resultados[nombre]['macro_f1']:.3f}", fontsize=7)
    fig.suptitle(f"Matrices de confusion ({etiqueta}): {modelo} (% por fila, entre parentesis el conteo)", fontsize=9)
    fig.tight_layout()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta)
    plt.close(fig)


def figura_cobertura(modelo: str, resultados: Dict[str, Dict], ruta: Path, etiqueta: str = "test") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for nombre, r in resultados.items():
        cv = r["cobertura_vs_exactitud"]
        curva = cv["curva"]
        linea, = ax.plot([f["cobertura"] * 100 for f in curva], [f["exactitud"] * 100 for f in curva],
                         label=nombre, lw=1.2)
        u = cv["umbral_minimo"]
        if u is not None:   # punto: umbral mas bajo cuya exactitud de aceptadas ya llega al objetivo
            ax.plot(u["cobertura"] * 100, u["exactitud"] * 100, "o", color=linea.get_color(), ms=5)
    ax.axhline(OBJETIVO_EXACTITUD * 100, color="gray", ls="--", lw=0.8,
               label=f"objetivo {OBJETIVO_EXACTITUD * 100:.1f} % (acuerdo entre expertos)")
    ax.set_xlabel("Épocas aceptadas (%)")
    ax.set_ylabel("Exactitud de las aceptadas (%)")
    ax.set_title(f"Cobertura vs exactitud ({etiqueta}): {modelo}", fontsize=9)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=6, loc="lower right")
    ax.invert_xaxis()
    fig.tight_layout()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta)
    plt.close(fig)


# ---------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> Dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modelos", required=True, help="Carpeta con <run>_final.pth y <run>_final.json de los 6 modelos.")
    p.add_argument("--modo", choices=["prueba", "test"], default="prueba")
    p.add_argument("--confirmo-test-unico", action="store_true",
                   help="Obligatorio con --modo test: confirma que el test se usa una sola vez.")
    p.add_argument("--salida", default=None, help="Por defecto resultados_test (test) o resultados_test_PRUEBA (prueba).")
    p.add_argument("--figuras", default=None, help="Por defecto figures/ (test) o dentro de --salida (prueba).")
    p.add_argument("--sujetos-prueba", type=int, default=3, help="Sujetos de train usados en el modo prueba.")
    args = p.parse_args(argv)

    es_test = args.modo == "test"
    salida = Path(args.salida or ("resultados_test" if es_test else "resultados_test_PRUEBA"))
    figuras = Path(args.figuras or ("figures" if es_test else salida / "figuras"))
    candado = salida / "TEST_YA_EJECUTADO.json"
    etiqueta_fig = "test" if es_test else "ENSAYO: sujetos de train, no es rendimiento"
    carpeta = Path(args.modelos)

    # 1) comprobaciones que no tocan ningun dato
    if es_test:
        if not args.confirmo_test_unico:
            raise SystemExit("Falta --confirmo-test-unico: el test se usa UNA sola vez.")
        if candado.exists():
            raise SystemExit(f"El test ya se evaluo ({candado}). No se repite.")
    faltan = [m for m in MODELOS if not ((carpeta / f"{m}_final.pth").exists() and (carpeta / f"{m}_final.json").exists())]
    if faltan:
        raise SystemExit(f"Faltan archivos de estos modelos en {carpeta}: {faltan}")
    modelos = {m: cargar_modelo(carpeta, m) for m in MODELOS}          # falla aqui si los pesos no encajan
    print(f"Modelos cargados: {list(modelos)} | dispositivo: {base.DEVICE}")

    # 2) sujetos
    train_codes, test_codes = ds_mod.get_train_test_subject_codes()
    codigos = test_codes if es_test else train_codes[: args.sujetos_prueba]
    if es_test:
        assert not set(codigos) & set(train_codes), "Un sujeto de test aparece en train."
    print(f"Modo {args.modo.upper()}: {len(codigos)} sujetos"
          + ("" if es_test else " de TRAIN (el test no se toca)"))

    recs3 = ds_mod.load_npz_recordings(subject_codes=codigos, max_epochs_per_recording=None)   # noches completas
    recsE = None
    if any(info.get("emg") for _, info, _ in modelos.values()):
        from sleep_staging.training.emg_datos import cargar_con_emg
        recsE = cargar_con_emg(subject_codes=codigos, max_epochs_per_recording=None)
        assert [r.recording_id for r in recsE] == [r.recording_id for r in recs3], "Grabaciones distintas entre 3 canales y EMG."
    sujeto_de_rec = np.array([r.subject_code for r in recs3])
    print(f"{len(recs3)} grabaciones, {sum(r.X.shape[0] for r in recs3)} epocas")

    # 3) evaluacion
    salida.mkdir(parents=True, exist_ok=True)
    resumen: Dict = {"modo": args.modo, "sujetos": list(codigos), "n_grabaciones": len(recs3),
                     "grabaciones": [r.recording_id for r in recs3], "modelos": {}}
    for nombre, (modelo, info, posiciones) in modelos.items():
        recs = recsE if info.get("emg") else recs3
        combos = [combinacion_por_nombre(c, posiciones) for c in info["combinaciones"]]
        resultados: Dict[str, Dict] = {}
        t0 = time.time()
        for combo in combos:
            pred = predecir(modelo, recs, combo, posiciones)
            np.savez_compressed(salida / f"probs_{nombre}__{combo[0]}.npz", **pred)
            resultados[combo[0]] = evaluar_combinacion(pred, sujeto_de_rec)
            r = resultados[combo[0]]
            print(f"  {nombre:36s} {combo[0]:52s} F1 macro {r['macro_f1']:.4f} "
                  f"[{r['macro_f1_ic95'][0]:.3f}, {r['macro_f1_ic95'][1]:.3f}] | kappa {r['cohens_kappa']:.4f}")
        resumen["modelos"][nombre] = {"n_epocas_entrenamiento": info["n_epocas"], "emg": info.get("emg"),
                                      "combinaciones": resultados}
        figura_matrices(nombre, resultados, figuras / f"test_matriz_confusion_{nombre}.pdf", etiqueta_fig)
        figura_cobertura(nombre, resultados, figuras / f"test_cobertura_{nombre}.pdf", etiqueta_fig)
        print(f"  [{nombre}] listo en {time.time() - t0:.0f} s")

    with open(salida / "resultados_test.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)

    # tabla plana para el articulo
    with open(salida / "tabla_test.csv", "w", encoding="utf-8") as f:
        f.write("modelo,combinacion,accuracy,macro_f1,macro_f1_ic95_bajo,macro_f1_ic95_alto,balanced_accuracy,kappa,"
                + ",".join(f"f1_{c}" for c in CLASES) + ",umbral_82_6,cobertura_en_umbral,n_epocas\n")
        for m, d in resumen["modelos"].items():
            for c, r in d["combinaciones"].items():
                u = r["cobertura_vs_exactitud"]["umbral_minimo"]
                f.write(f"{m},{c},{r['accuracy']:.4f},{r['macro_f1']:.4f},{r['macro_f1_ic95'][0]:.4f},"
                        f"{r['macro_f1_ic95'][1]:.4f},{r['balanced_accuracy']:.4f},{r['cohens_kappa']:.4f},"
                        + ",".join(f"{r['f1_per_class'][k]:.4f}" for k in CLASES)
                        + f",{u['umbral'] if u else ''},{u['cobertura'] if u else ''},{r['n_epocas']}\n")

    # 4) candado (solo en modo test)
    if es_test:
        try:
            commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=RAIZ).stdout.strip()
        except Exception:
            commit = ""
        with open(candado, "w", encoding="utf-8") as f:
            json.dump({"fecha_utc": datetime.now(timezone.utc).isoformat(), "commit": commit,
                       "n_sujetos_test": len(codigos),
                       "md5_pesos": {m: md5(carpeta / f"{m}_final.pth") for m in MODELOS}}, f, indent=2)
        print(f"\nTest evaluado y bloqueado: {candado}")
    print(f"Resultados en {salida}")
    return resumen


if __name__ == "__main__":
    main()
