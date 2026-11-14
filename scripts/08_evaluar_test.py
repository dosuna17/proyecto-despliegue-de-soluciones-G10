"""
Evaluacion UNICA en el conjunto de test de los modelos finales del modelo flexible.

Modelos (todos ya entrenados, solo con sujetos de train; ver scripts/06_train_flexible.py):
    flexible                        flexible de 3 senales (6 combinaciones)
    flexible_emg_mlp                flexible con EMG en una rama aparte (12 combinaciones)
    flexible_sin_lstm               ablacion: el flexible de 3 senales sin la LSTM (6 combinaciones)
    control_eeg_fpz_cz              controles: un modelo por combinacion
    control_eeg_pz_oz
    control_eeg_fpz_cz__eeg_pz_oz
    control_eeg_fpz_cz__eog_horizontal
    control_eeg_fpz_cz__eeg_pz_oz__eog_horizontal__emg_submental   (control con las 4 senales)

Para cada modelo y cada combinacion de senales:
    - exactitud, F1 macro, exactitud balanceada, kappa, F1 por etapa y matriz de confusion
    - intervalo de confianza del 95 % (bootstrap por sujeto) de F1 macro y kappa
    - F1 macro y exactitud por persona (las dos noches de cada sujeto juntas): media, desviacion, min y max
    - curva cobertura vs exactitud segun la confianza (probabilidad maxima) de la prediccion
    - umbrales de confianza FIJADOS CON LA VALIDACION (no con el test): ver abajo
    - probabilidades por epoca guardadas (.npz), para volver a calcular sin reentrenar

Umbrales de confianza (objetivo: exactitud de las epocas aceptadas >= 82.6 %):
    Se calculan ANTES de tocar el test, con las probabilidades de validacion que guarda la validacion
    cruzada (<salida de 06>/probs_validacion/, argumento --probs-validacion). Para cada modelo:
      - umbral por combinacion: el menor umbral cuya exactitud de aceptadas en validacion llega al objetivo;
      - umbral global del modelo: igual, con las probabilidades de todas sus combinaciones juntas
        (es el unico umbral que usaria un tablero).
    El test solo comprueba cobertura y exactitud con esos umbrales ya fijados (IC95 por sujeto).
    La curva completa y el menor umbral segun el test se guardan solo como referencia descriptiva
    (su exactitud seria optimista porque se ajusta sobre los mismos datos).

Umbrales POR CLASE (idea del profesor Mario; metas por clase tomadas de SleepFM, Thapa et al., Nat. Med. 2026):
    Una epoca se acepta solo si su clase predicha c tiene confianza >= tau_c; si no, va a revision.
    - Meta de la clase c:  meta_c = referencia_c - delta.  referencia_c es la SENSIBILIDAD por clase REAL
      (fila normalizada de la matriz de confusion, Fig. 1d, SHHS) que reporta SleepFM. NO es la misma metrica
      que la nuestra: nosotros pedimos la EXACTITUD de lo aceptado dentro de cada clase PREDICHA (precision).
      La referencia se usa solo como orden de dificultad de las clases (N1 mucho mas dificil que REM), no como
      un valor equivalente.
    - tau_c = menor umbral (rejilla 0.00-0.99) cuya precision de las epocas aceptadas predichas como c, medida
      en la VALIDACION (probabilidades de los 5 pliegues de la CV), llega a meta_c, con al menos
      MIN_ACEPTADAS_VALIDACION epocas aceptadas (para que no lo decida un puñado de epocas).
    - delta = 0, 5 y 10 puntos se reportan SIEMPRE en validacion (cobertura y exactitud por clase) y se escribe
      cual se usa (--delta, por defecto 5) ANTES del test. El test solo verifica el delta elegido.
    - Clase INALCANZABLE (ningun umbral llega a su meta con el soporte minimo): tau_c = None y TODAS las epocas
      predichas como esa clase van a revision (cobertura de la clase = 0). Queda marcada en el JSON y en la tabla.
    - El umbral global de 82.6 % (acuerdo entre expertos) se mantiene como criterio secundario, sin cambios.
    - Se reporta la exactitud REAL de lo aceptado por clase en validacion y en test (en N1 la meta queda ~51 %:
      casi la mitad de lo aceptado puede estar mal; es lo que pide la meta y se muestra tal cual).
    Para fijarlos y revisarlos sin tocar el test: --solo-umbrales (no carga modelos ni datos).

Epocas evaluadas: cada noche completa se parte en bloques de 15 epocas sin traslape desde la epoca 0 y se
descarta el resto del final. rec_idx y epoca_idx en cada .npz permiten alinear otros modelos con estas mismas epocas.

Modos:
    --modo prueba   (por defecto) usa 3 sujetos de TRAIN para comprobar que todo corre. No toca el test.
    --modo test     usa los sujetos de test, solo con --confirmo-test-unico. Crea el candado
                    TEST_YA_EJECUTADO.json AL EMPEZAR (estado "en curso", en una ruta fija fuera de --salida)
                    y lo marca "completo" al terminar. Si se corta, se completa con --reanudar (reutiliza las
                    predicciones ya guardadas, con los mismos pesos); no se puede repetir un test completo.

Carpeta de modelos (--modelos): debe tener, para cada modelo, <run>_final.pth y <run>_final.json.

Uso:
    python scripts/08_evaluar_test.py --modelos modelos_finales                       # prueba
    python scripts/08_evaluar_test.py --modelos modelos_finales --probs-validacion res_flexible res_emg ...
    python scripts/08_evaluar_test.py --modelos modelos_finales --modo test --confirmo-test-unico \\
        --probs-validacion res_flexible res_emg res_controles

    python scripts/08_evaluar_test.py --modelos modelos_finales --solo-umbrales \\
        --probs-validacion res_flexible res_emg res_controles          # umbrales por clase, solo validacion

Carpetas de datos: SLEEP_NPZ_DIR (3 canales) y SLEEP_EMG_DIR (EMG, solo para los modelos con EMG).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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
    n_canales_entrada,
)

CONTROL_4_SENALES = "control_eeg_fpz_cz__eeg_pz_oz__eog_horizontal__emg_submental"
MODELOS = [
    "flexible",
    "flexible_emg_mlp",
    "flexible_sin_lstm",
    "control_eeg_fpz_cz",
    "control_eeg_pz_oz",
    "control_eeg_fpz_cz__eeg_pz_oz",
    "control_eeg_fpz_cz__eog_horizontal",
    CONTROL_4_SENALES,
]
CLASES = base.CLASS_NAMES
OBJETIVO_EXACTITUD = 0.826          # acuerdo entre expertos (Rosenberg & Van Hout 2013); referencia aproximada (Sleep-EDF es R&K)
N_BOOTSTRAP = 1000
SEMILLA_BOOTSTRAP = 42
UMBRALES = np.round(np.arange(0.0, 1.0, 0.01), 2)
N_PLIEGUES = 5
# Umbrales por clase. Referencia: SleepFM (Thapa et al., Nat. Med. 2026), Fig. 1d, SHHS, matriz normalizada por FILA:
# es SENSIBILIDAD (de las epocas realmente de la clase c, que fraccion se predice c). Nuestra meta es otra metrica:
# PRECISION de lo aceptado en cada clase predicha. Se usa solo como referencia del orden de dificultad.
REFERENCIA_SLEEPFM_SENSIBILIDAD = {"W": 0.875, "N1": 0.560, "N2": 0.808, "N3": 0.862, "REM": 0.937}
DELTAS = (0.0, 0.05, 0.10)          # puntos que se restan a la referencia, todos reportados en validacion
DELTA_POR_DEFECTO = 0.05
MIN_ACEPTADAS_VALIDACION = 100      # un umbral por clase solo vale si deja al menos tantas epocas aceptadas en validacion
JUSTIFICACION_DELTA_POR_DEFECTO = (
    "delta = 5 puntos, propuesto por el equipo: relaja la referencia de otra metrica (sensibilidad de SleepFM) "
    "sin volverla trivial; se fija con la validacion antes del test y el test solo verifica.")
REGLA_INALCANZABLE = ("Si ninguna confianza (0.00-0.99) alcanza la meta de una clase con al menos "
                      f"{MIN_ACEPTADAS_VALIDACION} epocas aceptadas en validacion, esa clase queda 'inalcanzable': "
                      "umbral = None y TODAS las epocas predichas como esa clase van a revision.")
NOTA_METRICA = ("La referencia de SleepFM es sensibilidad por clase real (fila); la meta de este modelo es la "
                "exactitud (precision) de las epocas aceptadas por clase predicha. No son la misma metrica.")
RUTA_CANDADO_POR_DEFECTO = os.environ.get("TEST_LOCK_PATH") or str(RAIZ / "TEST_YA_EJECUTADO.json")


# ---------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------
def md5(ruta: Path) -> str:
    h = hashlib.md5()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def escribir_json(ruta: Path, obj) -> None:
    """Escritura atomica: nunca queda un candado a medias."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tmp = ruta.with_name(ruta.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    tmp.replace(ruta)


def fmt(x, dec: int = 3) -> str:
    return "" if x is None else f"{x:.{dec}f}"


# ---------------------------------------------------------------------
# Carga de modelos
# ---------------------------------------------------------------------
def cargar_modelo(carpeta: Path, run: str):
    """Reconstruye el modelo desde su JSON y carga los pesos (estricto). Devuelve (modelo, info, posiciones)."""
    ruta_json, ruta_pth = carpeta / f"{run}_final.json", carpeta / f"{run}_final.pth"
    if not ruta_json.exists() or not ruta_pth.exists():
        raise FileNotFoundError(f"Falta {ruta_json.name} o {ruta_pth.name} en {carpeta}")
    info = json.load(open(ruta_json, encoding="utf-8"))
    emg = info.get("emg")                                    # None en los modelos sin EMG
    sin_lstm = bool(info.get("sin_lstm"))                    # ausente en los JSON anteriores
    if emg and sin_lstm:
        raise ValueError(f"{run}: no existe un modelo con EMG y sin LSTM.")
    posiciones = POSICIONES_EMG if emg else POSICIONES
    if info["n_canales_entrada"] != n_canales_entrada(posiciones):
        raise ValueError(f"{run}: n_canales_entrada={info['n_canales_entrada']} no coincide con las posiciones.")
    config = TinySleepNetConfig(use_rnn=not sin_lstm, seq_length=ds_mod.DEFAULT_SEQ_LENGTH,
                                input_channels=info["n_canales_entrada"])
    if emg:
        from sleep_staging.models.tinysleepnet_emg import TinySleepNetEMG
        modelo = TinySleepNetEMG(config, rama=emg)
    elif sin_lstm:
        from sleep_staging.models.tinysleepnet_sin_lstm import TinySleepNetSinLSTM
        modelo = TinySleepNetSinLSTM(config)
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


def metricas_por_persona(y_true, y_pred, sujeto_de_epoca) -> Dict:
    """
    F1 macro y exactitud de cada persona (sus dos noches juntas). El F1 macro de una persona se promedia
    solo sobre las clases que aparecen en sus etiquetas reales (una clase sin ninguna epoca no se puntua
    como 0). Devuelve la lista por persona y media, desviacion (n-1), minimo y maximo entre personas.
    """
    filas = []
    for s in sorted(np.unique(sujeto_de_epoca)):
        m = sujeto_de_epoca == s
        cm = np.zeros((5, 5))
        np.add.at(cm, (y_true[m], y_pred[m]), 1)
        tp = np.diag(cm)
        den = 2 * tp + (cm.sum(axis=0) - tp) + (cm.sum(axis=1) - tp)
        f1 = np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0.0)
        con_soporte = cm.sum(axis=1) > 0
        filas.append({"sujeto": str(s), "n_epocas": int(m.sum()),
                      "accuracy": float(tp.sum() / cm.sum()),
                      "macro_f1": float(f1[con_soporte].mean()),
                      "clases_con_soporte": int(con_soporte.sum())})
    resumen = {}
    for k in ("macro_f1", "accuracy"):
        v = np.array([f[k] for f in filas])
        resumen[k] = {"media": float(v.mean()), "desviacion": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                      "minimo": float(v.min()), "maximo": float(v.max())}
    return {"personas": filas, "resumen": resumen}


def curva_cobertura(probs: np.ndarray, y_true: np.ndarray, objetivo: float = OBJETIVO_EXACTITUD) -> Dict:
    """Para cada umbral t: fraccion de epocas con confianza >= t y exactitud dentro de ellas."""
    confianza, y_pred = probs.max(axis=1).astype(np.float64), probs.argmax(axis=1)
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


# ---------------------------------------------------------------------
# Umbrales fijados con la validacion
# ---------------------------------------------------------------------
def _archivos_validacion(dirs: Sequence[Path], modelo: str, combo: str) -> Dict[int, Path]:
    """<modelo>_pliegue<k>__<combo>.npz en cualquiera de las carpetas (coincidencia exacta del nombre)."""
    patron = re.compile(rf"^{re.escape(modelo)}_pliegue(\d+)__{re.escape(combo)}$")
    por_pliegue: Dict[int, Path] = {}
    for d in dirs:
        for ruta in Path(d).rglob("*.npz"):
            m = patron.match(ruta.stem)
            if m:
                k = int(m.group(1))
                if k in por_pliegue and por_pliegue[k].resolve() != ruta.resolve():
                    raise ValueError(f"Probabilidades de validacion duplicadas para {modelo}/{combo}, pliegue {k}: "
                                     f"{por_pliegue[k]} y {ruta}")
                por_pliegue[k] = ruta
    return por_pliegue


def cargar_probs_validacion(dirs: Sequence[Path], modelo: str, combo: str, exigir_pliegues: bool):
    por_pliegue = _archivos_validacion(dirs, modelo, combo)
    if exigir_pliegues and sorted(por_pliegue) != list(range(N_PLIEGUES)):
        raise SystemExit(f"{modelo}/{combo}: se esperaban los {N_PLIEGUES} pliegues de validacion y hay "
                         f"{sorted(por_pliegue)}. Un umbral con pliegues faltantes quedaria sesgado.")
    if not por_pliegue:
        return None
    datos = [np.load(por_pliegue[k]) for k in sorted(por_pliegue)]
    return (np.concatenate([d["probs"] for d in datos]), np.concatenate([d["y_true"] for d in datos]),
            sorted(por_pliegue))


def umbrales_desde_validacion(dirs: Sequence[Path], modelo: str, combos: Sequence[str], exigir_pliegues: bool) -> Dict:
    """Umbral por combinacion y umbral global del modelo, con probabilidades de validacion de los 5 pliegues."""
    por_comb, pool_p, pool_y = {}, [], []
    for c in combos:
        carga = cargar_probs_validacion(dirs, modelo, c, exigir_pliegues)
        if carga is None:
            por_comb[c] = None
            continue
        p, y, pliegues = carga
        pool_p.append(p)
        pool_y.append(y)
        u = curva_cobertura(p, y)["umbral_minimo"]
        por_comb[c] = {"umbral": u["umbral"] if u else None,
                       "validacion": {"cobertura": u["cobertura"], "exactitud": u["exactitud"]} if u else None,
                       "pliegues": pliegues, "n_epocas_validacion": int(len(y))}
    glob = None
    if pool_p and len(pool_p) == len(combos):
        p, y = np.concatenate(pool_p), np.concatenate(pool_y)
        u = curva_cobertura(p, y)["umbral_minimo"]
        glob = {"umbral": u["umbral"] if u else None,
                "validacion": {"cobertura": u["cobertura"], "exactitud": u["exactitud"]} if u else None,
                "n_epocas_validacion": int(len(y))}
    return {"combinaciones": por_comb, "global": glob}


# ---------------------------------------------------------------------
# Umbrales por clase (fijados con la validacion)
# ---------------------------------------------------------------------
def _clave_delta(d: float) -> str:
    return f"{d:.2f}"


def curvas_por_clase(probs: np.ndarray, y_true: np.ndarray) -> Dict[str, Dict]:
    """Por clase predicha c y para cada umbral de UMBRALES: n de epocas predichas c con confianza >= umbral y cuantas
    aciertan (y_true == c). Independiente de delta: se calcula una vez."""
    conf = probs.max(axis=1).astype(np.float64)
    pred = probs.argmax(axis=1)
    out = {}
    for k, c in enumerate(CLASES):
        m = pred == k
        cf, ok = conf[m], (y_true[m] == k)
        orden = np.argsort(cf, kind="stable")
        cf_s, ok_s = cf[orden], ok[orden]
        resto_ok = np.concatenate([np.cumsum(ok_s[::-1])[::-1], [0]])      # aciertos con confianza >= cf_s[i]
        pos = np.searchsorted(cf_s, UMBRALES, side="left")
        out[c] = {"n_predichas": int(m.sum()), "n_aceptadas": (len(cf_s) - pos).astype(np.int64),
                  "n_aciertos": resto_ok[pos].astype(np.int64)}
    return out


def bloque_umbrales_clase(curvas: Dict[str, Dict], n_total: int, delta: float) -> Dict:
    """Umbral por clase para un delta: menor umbral con precision >= referencia - delta y soporte minimo."""
    clases, acc_tot, ok_tot = {}, 0, 0
    for c in CLASES:
        cv = curvas[c]
        meta = round(REFERENCIA_SLEEPFM_SENSIBILIDAD[c] - delta, 4)
        n_acc, n_ok = cv["n_aceptadas"], cv["n_aciertos"]
        valido = n_acc >= MIN_ACEPTADAS_VALIDACION
        prec = np.where(n_acc > 0, n_ok / np.maximum(n_acc, 1), 0.0)
        cumple = np.flatnonzero(valido & (prec >= meta))
        if len(cumple):
            i = int(cumple[0])
            tau = float(UMBRALES[i])
            val = {"n_predichas": cv["n_predichas"], "n_aceptadas": int(n_acc[i]),
                   "cobertura": float(n_acc[i] / cv["n_predichas"]), "exactitud_aceptadas": float(prec[i])}
            acc_tot += int(n_acc[i]); ok_tot += int(n_ok[i])
            clases[c] = {"referencia_sleepfm_sensibilidad": REFERENCIA_SLEEPFM_SENSIBILIDAD[c], "meta_exactitud": meta,
                         "umbral": tau, "inalcanzable": False, "validacion": val}
        else:
            mejor = float(prec[valido].max()) if valido.any() else None
            clases[c] = {"referencia_sleepfm_sensibilidad": REFERENCIA_SLEEPFM_SENSIBILIDAD[c], "meta_exactitud": meta,
                         "umbral": None, "inalcanzable": True,
                         "mejor_exactitud_con_soporte": mejor,
                         "validacion": {"n_predichas": cv["n_predichas"], "n_aceptadas": 0, "cobertura": 0.0,
                                        "exactitud_aceptadas": None}}
    return {"delta": delta, "clases": clases,
            "validacion_global": {"cobertura": acc_tot / n_total,
                                  "exactitud_aceptadas": (ok_tot / acc_tot) if acc_tot else None,
                                  "n_aceptadas": acc_tot, "n_epocas_validacion": int(n_total)}}


def bloques_por_delta(probs: np.ndarray, y_true: np.ndarray) -> Dict[str, Dict]:
    curvas = curvas_por_clase(probs, y_true)
    return {_clave_delta(d): bloque_umbrales_clase(curvas, len(y_true), d) for d in DELTAS}


def umbrales_clase_desde_validacion(dirs: Sequence[Path], modelo: str, combos: Sequence[str],
                                    exigir_pliegues: bool) -> Dict:
    """Umbrales por clase (todos los delta) por combinacion y globales del modelo (con todas sus combinaciones juntas)."""
    por_comb, pool_p, pool_y = {}, [], []
    for c in combos:
        carga = cargar_probs_validacion(dirs, modelo, c, exigir_pliegues)
        if carga is None:
            por_comb[c] = None
            continue
        p, y, _ = carga
        pool_p.append(p)
        pool_y.append(y)
        por_comb[c] = bloques_por_delta(p, y)
    glob = None
    if pool_p and len(pool_p) == len(combos):
        glob = bloques_por_delta(np.concatenate(pool_p), np.concatenate(pool_y))
    return {"combinaciones": por_comb, "global": glob}


def evaluar_umbrales_clase(pred: Dict[str, np.ndarray], sujeto_de_rec: np.ndarray, taus: Dict[str, Optional[float]],
                           n_boot: int = N_BOOTSTRAP, semilla: int = SEMILLA_BOOTSTRAP) -> Dict:
    """Cobertura y exactitud de lo aceptado POR CLASE PREDICHA en test con umbrales YA FIJADOS (tau None = la clase
    entera va a revision); IC95 por sujeto."""
    probs, y_true = pred["probs"], pred["y_true"]
    suj = sujeto_de_rec[pred["rec_idx"]]
    conf, yp = probs.max(axis=1).astype(np.float64), probs.argmax(axis=1)
    acierto = yp == y_true
    sujetos = np.unique(suj)
    n_tot = np.array([(suj == s).sum() for s in sujetos], dtype=float)
    pred_c, acc_c, ok_c = [], [], []
    for k, c in enumerate(CLASES):
        es_c = yp == k
        t = taus.get(c)
        acepta = es_c & (conf >= t) if t is not None else np.zeros_like(es_c)
        pred_c.append([(es_c & (suj == s)).sum() for s in sujetos])
        acc_c.append([(acepta & (suj == s)).sum() for s in sujetos])
        ok_c.append([(acepta & acierto & (suj == s)).sum() for s in sujetos])
    pred_c, acc_c, ok_c = (np.array(a, dtype=float) for a in (pred_c, acc_c, ok_c))      # (clase, sujeto)
    idx = np.random.default_rng(semilla).integers(0, len(sujetos), size=(n_boot, len(sujetos)))

    def resumen(n_pred, n_acc, n_ok, n_all):
        cob = n_acc.sum() / n_pred.sum() if n_pred.sum() > 0 else None
        ex = n_ok.sum() / n_acc.sum() if n_acc.sum() > 0 else None
        pb, ab, kb = n_pred[idx].sum(axis=1), n_acc[idx].sum(axis=1), n_ok[idx].sum(axis=1)
        cob_b = np.where(pb > 0, ab / np.maximum(pb, 1), np.nan)
        ex_b = np.where(ab > 0, kb / np.maximum(ab, 1), np.nan)
        ic = lambda v: ([float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] if np.isfinite(v).any() else None)
        return {"n_predichas": int(n_pred.sum()), "n_aceptadas": int(n_acc.sum()),
                "cobertura": None if cob is None else float(cob), "cobertura_ic95": ic(cob_b),
                "exactitud_aceptadas": None if ex is None else float(ex), "exactitud_ic95": ic(ex_b)}

    clases = {c: {"umbral": taus.get(c), **resumen(pred_c[k], acc_c[k], ok_c[k], n_tot)} for k, c in enumerate(CLASES)}
    glob = resumen(pred_c.sum(axis=0), acc_c.sum(axis=0), ok_c.sum(axis=0), n_tot)
    glob["error_global"] = 1.0 - float(acierto.mean())
    return {"clases": clases, "global": glob}


def evaluar_umbral(pred: Dict[str, np.ndarray], sujeto_de_rec: np.ndarray, umbral: Optional[float],
                   n_boot: int = N_BOOTSTRAP, semilla: int = SEMILLA_BOOTSTRAP) -> Optional[Dict]:
    """Cobertura y exactitud de las aceptadas en test con un umbral YA FIJADO; IC95 por sujeto."""
    if umbral is None:
        return None
    probs, y_true = pred["probs"], pred["y_true"]
    suj = sujeto_de_rec[pred["rec_idx"]]
    acierto = probs.argmax(axis=1) == y_true
    acepta = probs.max(axis=1).astype(np.float64) >= umbral
    sujetos = np.unique(suj)
    n_tot = np.array([(suj == s).sum() for s in sujetos], dtype=float)
    n_acc = np.array([(acepta & (suj == s)).sum() for s in sujetos], dtype=float)
    n_ok = np.array([(acepta & acierto & (suj == s)).sum() for s in sujetos], dtype=float)
    cobertura = float(n_acc.sum() / n_tot.sum())
    exactitud = float(n_ok.sum() / n_acc.sum()) if n_acc.sum() > 0 else None
    idx = np.random.default_rng(semilla).integers(0, len(sujetos), size=(n_boot, len(sujetos)))
    cob_b = n_acc[idx].sum(axis=1) / n_tot[idx].sum(axis=1)
    ex_b = n_ok[idx].sum(axis=1) / np.maximum(n_acc[idx].sum(axis=1), 1)
    err_global = 1.0 - float(acierto.mean())
    return {"umbral": float(umbral), "cobertura": cobertura, "exactitud_aceptadas": exactitud,
            "n_aceptadas": int(n_acc.sum()),
            "cobertura_ic95": [float(np.percentile(cob_b, 2.5)), float(np.percentile(cob_b, 97.5))],
            "exactitud_ic95": [float(np.percentile(ex_b, 2.5)), float(np.percentile(ex_b, 97.5))],
            # si se aceptaran epocas al azar con la misma cobertura, el error seria el error global
            "error_global": err_global,
            "reduccion_relativa_del_error": (1.0 - (1.0 - exactitud) / err_global)
            if exactitud is not None and err_global > 0 else None}


def evaluar_combinacion(pred: Dict[str, np.ndarray], sujeto_de_rec: np.ndarray,
                        umbrales_val: Optional[Dict] = None, umbrales_clase: Optional[Dict] = None) -> Dict:
    """umbrales_val: {"combinacion": {...}|None, "global": {...}|None} (ya fijados con la validacion).
    umbrales_clase: {"combinacion": bloque|None, "global": bloque|None}: bloque del delta elegido (ver bloque_umbrales_clase)."""
    y_true, probs = pred["y_true"], pred["probs"]
    y_pred = probs.argmax(axis=1)
    suj_epoca = sujeto_de_rec[pred["rec_idx"]]
    metricas = base.compute_metrics(y_true, y_pred)
    metricas.update(intervalo_bootstrap(y_true, y_pred, suj_epoca))
    metricas["n_epocas"] = int(len(y_true))
    metricas["por_persona"] = metricas_por_persona(y_true, y_pred, suj_epoca)
    cv = curva_cobertura(probs, y_true)
    cv["referencia_umbral_en_test"] = cv.pop("umbral_minimo")    # solo descriptivo: se ajusta sobre el propio test
    metricas["cobertura_vs_exactitud"] = cv
    metricas["umbral_validado"] = {}
    for tipo in ("combinacion", "global"):
        fijado = (umbrales_val or {}).get(tipo)
        t = fijado["umbral"] if fijado else None
        metricas["umbral_validado"][tipo] = (
            {"validacion": fijado["validacion"], "test": evaluar_umbral(pred, sujeto_de_rec, t)} if fijado and t is not None
            else None)
    metricas["umbral_por_clase"] = {}
    for tipo in ("combinacion", "global"):
        b = (umbrales_clase or {}).get(tipo)
        metricas["umbral_por_clase"][tipo] = (
            {"delta": b["delta"], "validacion": b,
             "test": evaluar_umbrales_clase(pred, sujeto_de_rec, {c: v["umbral"] for c, v in b["clases"].items()})}
            if b else None)
    return metricas


# ---------------------------------------------------------------------
# Figuras (PNG a 300 dpi y PDF)
# ---------------------------------------------------------------------
def _guardar(fig, ruta_base: Path) -> None:
    ruta_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta_base.with_suffix(".png"), dpi=300)
    fig.savefig(ruta_base.with_suffix(".pdf"))


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
    _guardar(fig, ruta)
    plt.close(fig)


def figura_cobertura(modelo: str, resultados: Dict[str, Dict], ruta: Path, etiqueta: str = "test") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    estilos = ["-", "--", "-.", ":"]
    marcas = ["o", "s", "^", "D", "v", "P", "X", "*"]
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for i, (nombre, r) in enumerate(resultados.items()):
        curva = r["cobertura_vs_exactitud"]["curva"]
        linea, = ax.plot([f["cobertura"] * 100 for f in curva], [f["exactitud"] * 100 for f in curva],
                         label=nombre, lw=1.2, ls=estilos[i % len(estilos)])
        uv = (r["umbral_validado"].get("combinacion") or {}).get("test")
        if uv is not None and uv["exactitud_aceptadas"] is not None:   # punto: umbral fijado en validacion, medido en test
            ax.plot(uv["cobertura"] * 100, uv["exactitud_aceptadas"] * 100, marcas[i % len(marcas)],
                    color=linea.get_color(), ms=6, mfc="none", mew=1.4)
    ax.axhline(OBJETIVO_EXACTITUD * 100, color="gray", ls="--", lw=0.8,
               label=f"objetivo {OBJETIVO_EXACTITUD * 100:.1f} % (acuerdo entre expertos)")
    ax.set_xlabel("Épocas aceptadas (%)")
    ax.set_ylabel("Exactitud de las aceptadas (%)")
    ax.set_title(f"Cobertura vs exactitud ({etiqueta}): {modelo}\nmarcador: umbral fijado en validacion", fontsize=9)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=6, loc="lower right")
    ax.invert_xaxis()
    fig.tight_layout()
    _guardar(fig, ruta)
    plt.close(fig)


def figura_por_persona(modelo: str, resultados: Dict[str, Dict], ruta: Path, etiqueta: str = "test") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nombres = list(resultados)
    datos = [[p["macro_f1"] for p in resultados[n]["por_persona"]["personas"]] for n in nombres]
    fig, ax = plt.subplots(figsize=(max(5.0, 0.7 * len(nombres) + 3), 4.2))
    ax.boxplot(datos, showmeans=True, widths=0.5)
    rng = np.random.default_rng(0)
    for i, v in enumerate(datos, start=1):
        ax.plot(i + rng.uniform(-0.12, 0.12, len(v)), v, "k.", ms=3, alpha=0.6)
    ax.set_xticks(range(1, len(nombres) + 1), nombres, rotation=60, ha="right", fontsize=6)
    ax.set_ylabel("F1 macro por persona")
    ax.set_title(f"Variabilidad entre personas ({etiqueta}): {modelo}", fontsize=9)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    _guardar(fig, ruta)
    plt.close(fig)


# ---------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------

def config_clases(args) -> Dict:
    return {"delta_elegido": args.delta, "justificacion_delta": args.justificacion_delta, "deltas_reportados": list(DELTAS),
            "referencia_sleepfm_sensibilidad": REFERENCIA_SLEEPFM_SENSIBILIDAD, "nota_metrica": NOTA_METRICA,
            "regla_clase_inalcanzable": REGLA_INALCANZABLE, "min_aceptadas_validacion": MIN_ACEPTADAS_VALIDACION,
            "objetivo_global_secundario": OBJETIVO_EXACTITUD}


def _bloque_elegido(por_delta: Optional[Dict], delta: float) -> Optional[Dict]:
    return None if not por_delta else por_delta[_clave_delta(delta)]


def _imprimir_clases(modelo: str, glob: Optional[Dict[str, Dict]], delta: float, todos: bool = False) -> None:
    if not glob:
        print(f"  umbrales por clase {modelo}: sin probabilidades de validacion")
        return
    for d in (DELTAS if todos else (delta,)):
        b = glob[_clave_delta(d)]
        partes = []
        for c in CLASES:
            x = b["clases"][c]
            partes.append(f"{c} {'INALCANZABLE' if x['inalcanzable'] else format(x['umbral'], '.2f')}"
                          + ("" if x["inalcanzable"] else f" (cob {x['validacion']['cobertura']:.2f}, exac {x['validacion']['exactitud_aceptadas']:.3f})"))
        g = b["validacion_global"]
        print(f"  por clase {modelo:36s} delta {d:.2f} meta {[round(REFERENCIA_SLEEPFM_SENSIBILIDAD[c] - d, 3) for c in CLASES]}\n"
              f"      " + " | ".join(partes) + f"\n      global: cobertura {g['cobertura']:.3f}, exactitud aceptadas "
              + ("-" if g["exactitud_aceptadas"] is None else f"{g['exactitud_aceptadas']:.3f}"))


def _filas_clases_csv(f, modelo: str, combo: str, por_delta: Dict[str, Dict], delta_elegido: float, test: Optional[Dict]) -> None:
    for kd, b in por_delta.items():
        for c in CLASES:
            x, v = b["clases"][c], b["clases"][c]["validacion"]
            t = (test or {}).get(c) if abs(float(kd) - delta_elegido) < 1e-9 and test else None
            f.write(f"{modelo},{combo},{kd},{'si' if abs(float(kd) - delta_elegido) < 1e-9 else 'no'},{c},"
                    f"{fmt(x['referencia_sleepfm_sensibilidad'])},{fmt(x['meta_exactitud'])},{fmt(x['umbral'], 2)},"
                    f"{'si' if x['inalcanzable'] else 'no'},{v['n_predichas']},{v['n_aceptadas']},{fmt(v['cobertura'])},"
                    f"{fmt(v['exactitud_aceptadas'])},"
                    + (f"{t['n_predichas']},{t['n_aceptadas']},{fmt(t['cobertura'])},{fmt(t['exactitud_aceptadas'])},"
                       f"{fmt(t['exactitud_ic95'] and t['exactitud_ic95'][0])},{fmt(t['exactitud_ic95'] and t['exactitud_ic95'][1])}\n"
                       if t else ",,,,,\n"))


ENCABEZADO_CLASES_CSV = ("modelo,combinacion,delta,delta_elegido,clase,referencia_sleepfm_sensibilidad,meta_exactitud,umbral,"
                         "inalcanzable,val_n_predichas,val_n_aceptadas,val_cobertura,val_exactitud_aceptadas,"
                         "test_n_predichas,test_n_aceptadas,test_cobertura,test_exactitud_aceptadas,"
                         "test_exactitud_ic95_bajo,test_exactitud_ic95_alto\n")


def solo_umbrales(carpeta: Path, dirs_val: Sequence[Path], salida: Path, args) -> Dict:
    """Umbrales (global y por clase, delta 0/5/10) con la validacion, sin modelos ni datos. Escribe el JSON y el CSV
    que se revisan y se dejan escritos ANTES de correr el test."""
    salida.mkdir(parents=True, exist_ok=True)
    out = {"fecha_utc": datetime.now(timezone.utc).isoformat(), "configuracion": config_clases(args), "modelos": {}}
    for m in MODELOS:
        ruta = carpeta / f"{m}_final.json"
        if not ruta.exists():
            raise SystemExit(f"Falta {ruta.name} en {carpeta} (se usa solo para saber las combinaciones del modelo).")
        combos = json.load(open(ruta, encoding="utf-8"))["combinaciones"]
        u = umbrales_desde_validacion(dirs_val, m, combos, exigir_pliegues=True)
        u["por_clase"] = umbrales_clase_desde_validacion(dirs_val, m, combos, exigir_pliegues=True)
        out["modelos"][m] = u
        g = u["global"]
        print(f"\n{m}: umbral global 82.6 %: " + (f"{g['umbral']}" if g and g["umbral"] is not None else "no alcanza el objetivo"))
        _imprimir_clases(m, u["por_clase"]["global"], args.delta, todos=True)
    escribir_json(salida / "umbrales_validacion.json", out)
    with open(salida / "umbrales_clase_validacion.csv", "w", encoding="utf-8-sig") as f:
        f.write(ENCABEZADO_CLASES_CSV)
        for m, u in out["modelos"].items():
            if u["por_clase"]["global"]:
                _filas_clases_csv(f, m, "GLOBAL", u["por_clase"]["global"], args.delta, None)
            for c, por_delta in u["por_clase"]["combinaciones"].items():
                if por_delta:
                    _filas_clases_csv(f, m, c, por_delta, args.delta, None)
    print(f"\nDelta elegido: {args.delta:.2f}. {args.justificacion_delta}\nEscrito en {salida}/umbrales_validacion.json y umbrales_clase_validacion.csv "
          f"(no se toco el test).")
    return out


def main(argv: Optional[Sequence[str]] = None) -> Dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modelos", required=True, help="Carpeta con <run>_final.pth y <run>_final.json de los modelos.")
    p.add_argument("--modo", choices=["prueba", "test"], default="prueba")
    p.add_argument("--confirmo-test-unico", action="store_true",
                   help="Obligatorio con --modo test: confirma que el test se usa una sola vez.")
    p.add_argument("--reanudar", action="store_true",
                   help="Solo --modo test: completa una evaluacion que quedo 'en curso' reutilizando las "
                        "predicciones ya guardadas en --salida (mismos pesos y mismos umbrales).")
    p.add_argument("--candado", default=RUTA_CANDADO_POR_DEFECTO,
                   help="Ruta FIJA del candado del test, fuera de --salida (por defecto TEST_YA_EJECUTADO.json en la "
                        "raiz del repo, o la variable TEST_LOCK_PATH; en Colab, una ruta de Drive).")
    p.add_argument("--probs-validacion", nargs="+", default=None,
                   help="Carpetas con las probabilidades de validacion de la CV (probs_validacion/). Fijan los "
                        "umbrales de confianza sin usar el test. Obligatorio con --modo test.")
    p.add_argument("--delta", type=float, default=DELTA_POR_DEFECTO, choices=list(DELTAS),
                   help="Puntos que se restan a la referencia por clase para fijar la meta (0, 0.05 o 0.10). Se escribe en "
                        "el candado antes del test; el test solo verifica este delta.")
    p.add_argument("--justificacion-delta", default=JUSTIFICACION_DELTA_POR_DEFECTO,
                   help="Texto que queda escrito (JSON y candado) con el delta elegido.")
    p.add_argument("--solo-umbrales", action="store_true",
                   help="Solo calcula y reporta los umbrales (global y por clase, delta 0/5/10) con la validacion y "
                        "termina: no carga modelos ni datos y no toca el test. Necesita --probs-validacion.")
    p.add_argument("--solo", nargs="+", default=None, choices=MODELOS,
                   help="Solo en --modo prueba: evalua unicamente estos modelos.")
    p.add_argument("--salida", default=None, help="Por defecto resultados_test (test) o resultados_test_PRUEBA (prueba).")
    p.add_argument("--figuras", default=None, help="Por defecto figures/ (test) o dentro de --salida (prueba).")
    p.add_argument("--sujetos-prueba", type=int, default=3, help="Sujetos de train usados en el modo prueba.")
    args = p.parse_args(argv)

    es_test = args.modo == "test"
    salida = Path(args.salida or ("resultados_test" if es_test else "resultados_test_PRUEBA"))
    figuras = Path(args.figuras or ("figures" if es_test else salida / "figuras"))
    candado = Path(args.candado)
    etiqueta_fig = "test" if es_test else "ENSAYO: sujetos de train, no es rendimiento"
    carpeta = Path(args.modelos)
    dirs_val = [Path(d) for d in args.probs_validacion] if args.probs_validacion else []
    modelos_a_evaluar = list(MODELOS)

    if args.solo_umbrales:
        if args.modo == "test" or args.confirmo_test_unico or args.reanudar:
            raise SystemExit("--solo-umbrales no se combina con --modo test, --confirmo-test-unico ni --reanudar.")
        if not dirs_val:
            raise SystemExit("--solo-umbrales necesita --probs-validacion.")
        return solo_umbrales(carpeta, dirs_val, Path(args.salida or "umbrales_validacion"), args)

    # 1) comprobaciones que no tocan ningun dato
    if es_test:
        if not args.confirmo_test_unico:
            raise SystemExit("Falta --confirmo-test-unico: el test se usa UNA sola vez.")
        if args.solo:
            raise SystemExit("--solo solo se usa en modo prueba: el test se evalua con todos los modelos.")
        if not dirs_val:
            raise SystemExit("Falta --probs-validacion: los umbrales se fijan con la validacion antes de tocar el test.")
        if candado.exists():
            estado = json.load(open(candado, encoding="utf-8")).get("estado", "completo")
            if estado == "completo":
                raise SystemExit(f"El test ya se evaluo ({candado}). No se repite.")
            if not args.reanudar:
                raise SystemExit(f"Hay una evaluacion de test 'en curso' ({candado}). Para completarla, "
                                 f"vuelve a correr con --reanudar (usa las predicciones ya guardadas).")
        elif args.reanudar:
            raise SystemExit(f"--reanudar necesita un candado 'en curso' y no existe {candado}.")
    else:
        if args.reanudar:
            raise SystemExit("--reanudar solo existe en modo test.")
        if args.solo:
            modelos_a_evaluar = [m for m in MODELOS if m in args.solo]
    faltan = [m for m in modelos_a_evaluar if not ((carpeta / f"{m}_final.pth").exists() and (carpeta / f"{m}_final.json").exists())]
    if faltan:
        raise SystemExit(f"Faltan archivos de estos modelos en {carpeta}: {faltan}")
    modelos = {m: cargar_modelo(carpeta, m) for m in modelos_a_evaluar}   # falla aqui si los pesos no encajan
    print(f"Modelos cargados: {list(modelos)} | dispositivo: {base.DEVICE}")

    # 2) umbrales con la VALIDACION (antes de tocar el test)
    umbrales: Dict[str, Dict] = {}
    if dirs_val:
        for m, (_, info, _) in modelos.items():
            umbrales[m] = umbrales_desde_validacion(dirs_val, m, info["combinaciones"], exigir_pliegues=es_test)
            g = umbrales[m]["global"]
            print(f"  umbral de validacion {m:36s} global: "
                  + (f"{g['umbral']}" if g and g["umbral"] is not None else "no alcanza el objetivo"
                     if g else "sin probabilidades de validacion"))
        for m, (_, info, _) in modelos.items():
            umbrales[m]["por_clase"] = umbrales_clase_desde_validacion(dirs_val, m, info["combinaciones"],
                                                                       exigir_pliegues=es_test)
            _imprimir_clases(m, umbrales[m]["por_clase"]["global"], args.delta)
        if es_test:
            sin = [m for m, u in umbrales.items() if u["global"] is None or u["por_clase"]["global"] is None]
            if sin:
                raise SystemExit(f"Faltan probabilidades de validacion de: {sin}")
    else:
        print("[AVISO] Sin --probs-validacion: no se calculan umbrales fijados en validacion (solo en prueba).")

    # 3) sujetos
    train_codes, test_codes = ds_mod.get_train_test_subject_codes()
    codigos = test_codes if es_test else train_codes[: args.sujetos_prueba]
    if es_test:
        assert not set(codigos) & set(train_codes), "Un sujeto de test aparece en train."
    print(f"Modo {args.modo.upper()}: {len(codigos)} sujetos"
          + ("" if es_test else " de TRAIN (el test no se toca)"))

    # 4) candado AL EMPEZAR (solo modo test)
    salida.mkdir(parents=True, exist_ok=True)
    if es_test:
        try:
            commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=RAIZ).stdout.strip()
        except Exception:
            commit = ""
        pesos_md5 = {m: md5(carpeta / f"{m}_final.pth") for m in MODELOS}
        if args.reanudar:
            previo = json.load(open(candado, encoding="utf-8"))
            if previo.get("md5_pesos") != pesos_md5:
                raise SystemExit("Los pesos de los modelos cambiaron desde que empezo el test: no se puede reanudar.")
            if previo.get("umbrales_por_clase_config") != json.loads(json.dumps(config_clases(args))):
                raise SystemExit("El delta o la configuracion de umbrales por clase cambio desde que empezo el test: no se puede reanudar.")
            if previo.get("umbrales_fijados_en_validacion") != json.loads(json.dumps(umbrales)):
                raise SystemExit("Los umbrales de validacion cambiaron desde que empezo el test: no se puede reanudar.")
            lock = previo
            lock["reanudado_utc"] = lock.get("reanudado_utc", []) + [datetime.now(timezone.utc).isoformat()]
        else:
            lock = {"estado": "en curso", "fecha_inicio_utc": datetime.now(timezone.utc).isoformat(),
                    "commit": commit, "n_sujetos_test": len(codigos), "modelos": list(MODELOS),
                    "md5_pesos": pesos_md5, "salida": str(salida.resolve()),
                    "SLEEP_NPZ_DIR": os.environ.get("SLEEP_NPZ_DIR"), "SLEEP_EMG_DIR": os.environ.get("SLEEP_EMG_DIR"),
                    "probs_validacion": [str(d) for d in dirs_val], "argumentos": list(argv if argv is not None else sys.argv[1:]),
                    "umbrales_fijados_en_validacion": json.loads(json.dumps(umbrales)),
                    "umbrales_por_clase_config": config_clases(args)}
        escribir_json(candado, lock)
        print(f"Candado del test ({lock['estado']}): {candado}")

    recs3 = ds_mod.load_npz_recordings(subject_codes=codigos, max_epochs_per_recording=None)   # noches completas
    recsE = None
    if any(info.get("emg") for _, info, _ in modelos.values()):
        from sleep_staging.training.emg_datos import cargar_con_emg
        recsE = cargar_con_emg(subject_codes=codigos, max_epochs_per_recording=None)
        assert [r.recording_id for r in recsE] == [r.recording_id for r in recs3], "Grabaciones distintas entre 3 canales y EMG."
    sujeto_de_rec = np.array([r.subject_code for r in recs3])
    n_totales = int(sum(r.X.shape[0] for r in recs3))
    n_esperadas = int(sum((r.X.shape[0] // ds_mod.DEFAULT_SEQ_LENGTH) * ds_mod.DEFAULT_SEQ_LENGTH for r in recs3))
    print(f"{len(recs3)} grabaciones, {n_totales} epocas, {n_esperadas} evaluadas "
          f"(bloques de {ds_mod.DEFAULT_SEQ_LENGTH} sin traslape; se descarta el resto de cada noche)")

    # 5) evaluacion
    resumen: Dict = {"modo": args.modo, "sujetos": list(codigos), "n_grabaciones": len(recs3),
                     "grabaciones": [r.recording_id for r in recs3],
                     "protocolo": {"bloques_de_epocas": ds_mod.DEFAULT_SEQ_LENGTH, "traslape": False,
                                   "n_epocas_totales": n_totales, "n_epocas_evaluadas": n_esperadas,
                                   "bootstrap": {"n": N_BOOTSTRAP, "semilla": SEMILLA_BOOTSTRAP, "unidad": "sujeto"},
                                   "objetivo_exactitud": OBJETIVO_EXACTITUD,
                                   "umbrales": "fijados con las probabilidades de validacion de la CV",
                                   "umbrales_por_clase": config_clases(args)},
                     "modelos": {}}
    for nombre, (modelo, info, posiciones) in modelos.items():
        recs = recsE if info.get("emg") else recs3
        combos = [combinacion_por_nombre(c, posiciones) for c in info["combinaciones"]]
        resultados: Dict[str, Dict] = {}
        t0 = time.time()
        for combo in combos:
            ruta_npz = salida / f"probs_{nombre}__{combo[0]}.npz"
            if args.reanudar and ruta_npz.exists():
                pred = {k: v for k, v in np.load(ruta_npz).items()}
                if len(pred["y_true"]) != n_esperadas:
                    raise SystemExit(f"{ruta_npz.name} no tiene {n_esperadas} epocas: borra solo ese archivo y reanuda.")
                print(f"  (reutiliza {ruta_npz.name})")
            else:
                pred = predecir(modelo, recs, combo, posiciones)
                tmp = ruta_npz.with_name(ruta_npz.stem + ".tmp.npz")
                np.savez_compressed(tmp, **pred)
                tmp.replace(ruta_npz)
            umb = umbrales.get(nombre)
            umb_c = {"combinacion": (umb["combinaciones"].get(combo[0]) if umb else None),
                     "global": (umb["global"] if umb else None)}
            uc = umb["por_clase"] if umb else None
            cl_c = {"combinacion": _bloque_elegido(uc and uc["combinaciones"].get(combo[0]), args.delta),
                    "global": _bloque_elegido(uc and uc["global"], args.delta)}
            resultados[combo[0]] = evaluar_combinacion(pred, sujeto_de_rec, umb_c, cl_c)
            r = resultados[combo[0]]
            pp = r["por_persona"]["resumen"]["macro_f1"]
            print(f"  {nombre:36s} {combo[0]:52s} F1 macro {r['macro_f1']:.4f} "
                  f"[{r['macro_f1_ic95'][0]:.3f}, {r['macro_f1_ic95'][1]:.3f}] | kappa {r['cohens_kappa']:.4f} "
                  f"| por persona {pp['media']:.3f} ± {pp['desviacion']:.3f}")
        resumen["modelos"][nombre] = {"n_epocas_entrenamiento": info["n_epocas"], "emg": info.get("emg"),
                                      "sin_lstm": bool(info.get("sin_lstm")),
                                      "noche_completa": info.get("noche_completa"),
                                      "epocas_cv": info.get("epocas_cv"),
                                      "umbrales_validacion": umbrales.get(nombre),
                                      "combinaciones": resultados}
        figura_matrices(nombre, resultados, figuras / f"test_matriz_confusion_{nombre}", etiqueta_fig)
        figura_cobertura(nombre, resultados, figuras / f"test_cobertura_{nombre}", etiqueta_fig)
        figura_por_persona(nombre, resultados, figuras / f"test_por_persona_{nombre}", etiqueta_fig)
        print(f"  [{nombre}] listo en {time.time() - t0:.0f} s")

    ruta_json = salida / "resultados_test.json"
    escribir_json(ruta_json, resumen)

    # tablas planas para el articulo (3 decimales; F1 de la clase M: la clase no se entrena, va con guion)
    with open(salida / "tabla_test.csv", "w", encoding="utf-8-sig") as f:
        f.write("modelo,combinacion,accuracy,macro_f1,macro_f1_ic95_bajo,macro_f1_ic95_alto,balanced_accuracy,kappa,"
                + ",".join(f"f1_{c}" for c in CLASES) + ",f1_M,"
                "f1_persona_media,f1_persona_desv,f1_persona_min,f1_persona_max,"
                "umbral_val_combinacion,cobertura_test,exactitud_aceptadas_test,"
                "umbral_val_global,cobertura_test_global,exactitud_aceptadas_test_global,n_epocas\n")
        for m, d in resumen["modelos"].items():
            for c, r in d["combinaciones"].items():
                pp = r["por_persona"]["resumen"]["macro_f1"]
                uc = (r["umbral_validado"]["combinacion"] or {}).get("test")
                ug = (r["umbral_validado"]["global"] or {}).get("test")
                f.write(f"{m},{c},{fmt(r['accuracy'])},{fmt(r['macro_f1'])},{fmt(r['macro_f1_ic95'][0])},"
                        f"{fmt(r['macro_f1_ic95'][1])},{fmt(r['balanced_accuracy'])},{fmt(r['cohens_kappa'])},"
                        + ",".join(fmt(r["f1_per_class"][k]) for k in CLASES) + ",—,"
                        + f"{fmt(pp['media'])},{fmt(pp['desviacion'])},{fmt(pp['minimo'])},{fmt(pp['maximo'])},"
                        + f"{fmt(uc and uc['umbral'], 2)},{fmt(uc and uc['cobertura'])},{fmt(uc and uc['exactitud_aceptadas'])},"
                        + f"{fmt(ug and ug['umbral'], 2)},{fmt(ug and ug['cobertura'])},{fmt(ug and ug['exactitud_aceptadas'])},"
                        + f"{r['n_epocas']}\n")
    if umbrales:
        with open(salida / "umbrales_clase.csv", "w", encoding="utf-8-sig") as f:
            f.write(ENCABEZADO_CLASES_CSV)
            for m, d in resumen["modelos"].items():
                u = umbrales[m]["por_clase"]
                if u["global"]:
                    # mismo test para todas las combinaciones: se reporta el global como el de la primera combinacion
                    primera = next(iter(d["combinaciones"].values()))
                    tg = ((primera["umbral_por_clase"] or {}).get("global") or {}).get("test")
                    _filas_clases_csv(f, m, "GLOBAL", u["global"], args.delta, tg and tg["clases"])
                for c, por_delta in u["combinaciones"].items():
                    if por_delta:
                        tc = ((d["combinaciones"][c]["umbral_por_clase"] or {}).get("combinacion") or {}).get("test")
                        _filas_clases_csv(f, m, c, por_delta, args.delta, tc and tc["clases"])
    with open(salida / "por_persona.csv", "w", encoding="utf-8-sig") as f:
        f.write("modelo,combinacion,sujeto,n_epocas,accuracy,macro_f1,clases_con_soporte\n")
        for m, d in resumen["modelos"].items():
            for c, r in d["combinaciones"].items():
                for q in r["por_persona"]["personas"]:
                    f.write(f"{m},{c},{q['sujeto']},{q['n_epocas']},{fmt(q['accuracy'])},{fmt(q['macro_f1'])},"
                            f"{q['clases_con_soporte']}\n")

    # 6) candado: completo
    if es_test:
        lock["estado"] = "completo"
        lock["fecha_fin_utc"] = datetime.now(timezone.utc).isoformat()
        lock["n_epocas_totales"], lock["n_epocas_evaluadas"] = n_totales, n_esperadas
        lock["md5_resultados_test_json"] = md5(ruta_json)
        escribir_json(candado, lock)
        print(f"\nTest evaluado y bloqueado: {candado}")
    print(f"Resultados en {salida}")
    return resumen


if __name__ == "__main__":
    main()
