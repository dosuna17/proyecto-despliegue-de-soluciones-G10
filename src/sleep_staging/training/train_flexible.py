"""
Validacion cruzada de TinySleepNet con canales flexibles (sin MLflow por ahora).

Reutiliza de train_tinysleepnet.py (version base, sin tocarla): run_epoch,
compute_metrics, set_seed y los hiperparametros del yaml. Lo nuevo:
  - Entrada de 2 * n_posiciones canales (senales con ceros + marcas de
    presencia), ver canales_flexibles.py.
  - Entrenamiento con todas las combinaciones (modelo flexible) o con una sola
    (modelo de control dedicado).
  - Validacion por combinacion; la parada temprana y el mejor modelo se guian
    por el promedio del F1 macro de las combinaciones.
  - Mismos pliegues por sujeto que la validacion cruzada de la version base.
  - Tiempos por epoca y por pliegue, y un JSON por pliegue (asi un corte de
    Colab no pierde los pliegues ya terminados).

El conjunto de test no se usa aqui: se reserva para comprobar la version base.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import f1_score

sys.path.append(str(Path(__file__).resolve().parents[2]))  # carpeta src/
from data.dataset import (  # noqa: E402
    DEFAULT_SEQ_LENGTH,
    compute_class_weights,
    get_train_test_subject_codes,
    load_npz_recordings,
    make_subject_kfold_splits,
)
from sleep_staging.models.tinysleepnet import (  # noqa: E402
    TinySleepNet,
    TinySleepNetConfig,
    build_class_weighted_loss,
)
from sleep_staging.training import train_tinysleepnet as base  # noqa: E402
from sleep_staging.training.canales_flexibles import (  # noqa: E402
    POSICIONES,
    POSICIONES_EMG,
    Combinacion,
    build_flexible_dataloader,
    combinacion_por_nombre,
    combinaciones_validas,
    n_canales_entrada,
)


def _contexto(emg: Optional[str], sin_lstm: bool = False):
    """
    emg=None  -> modelo flexible de 3 senales de 100 Hz (6 combinaciones).
    emg='mlp' o 'conv' -> TinySleepNetEMG con el EMG en una rama aparte (12 combinaciones).
    sin_lstm=True -> ablacion: el flexible de 3 senales sin la LSTM (cada epoca por separado).
    Devuelve (posiciones, funcion de carga, constructor del modelo, sufijo del nombre del run).
    """
    if sin_lstm:
        if emg is not None:
            raise ValueError("La ablacion sin LSTM se entrena sobre el flexible de 3 senales: no combines --sin-lstm con --emg.")
        from sleep_staging.models.tinysleepnet_sin_lstm import TinySleepNetSinLSTM
        return POSICIONES, load_npz_recordings, (lambda cfg: TinySleepNetSinLSTM(cfg)), "_sin_lstm"
    if emg is None:
        return POSICIONES, load_npz_recordings, (lambda cfg: TinySleepNet(cfg)), ""
    from sleep_staging.models.tinysleepnet_emg import RAMAS, TinySleepNetEMG
    from sleep_staging.training.emg_datos import cargar_con_emg
    if emg not in RAMAS:
        raise ValueError(f"emg debe ser None o una de {RAMAS}; llego '{emg}'.")
    return POSICIONES_EMG, cargar_con_emg, (lambda cfg: TinySleepNetEMG(cfg, rama=emg)), f"_emg_{emg}"


def _comprobar_control(control: Optional[str], emg: Optional[str], sin_lstm: bool) -> None:
    if sin_lstm and control is not None:
        raise ValueError("La ablacion sin LSTM es del modelo flexible: no combines --sin-lstm con --control.")
    if control is not None and emg is not None and "emg_submental" not in control:
        raise ValueError("Con --emg, el control debe incluir el EMG en su nombre (p. ej. "
                         "eeg_fpz_cz__eeg_pz_oz__eog_horizontal__emg_submental); si no, su nombre "
                         "chocaria con el de un control sin EMG.")
    if control is not None and emg is None and "emg_submental" in control:
        raise ValueError("Un control con EMG necesita --emg (mlp o conv).")


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"No serializable: {type(o)}")


def probs_validacion(model, recs, combo: Combinacion, posiciones: Sequence[str]) -> Dict[str, np.ndarray]:
    """
    Probabilidades por epoca de `recs` con una combinacion, en el mismo orden y con los mismos
    bloques de 15 epocas que la validacion. Sirven para fijar el umbral de confianza con datos
    de validacion (el test solo se usa para comprobarlo).
    """
    loader = build_flexible_dataloader(recs, [combo], shuffle=False, posiciones=posiciones)
    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            logits, _ = model(x.to(base.DEVICE))
            probs.append(torch.softmax(logits, dim=-1).cpu().numpy().reshape(-1, 5))
            ys.append(y.numpy().reshape(-1))
    rec_idx, epoca_idx = [], []
    for i, r in enumerate(recs):
        n = (r.X.shape[0] // DEFAULT_SEQ_LENGTH) * DEFAULT_SEQ_LENGTH
        rec_idx.append(np.full(n, i, dtype=np.int32))
        epoca_idx.append(np.arange(n, dtype=np.int32))
    pred = {
        "probs": np.concatenate(probs).astype(np.float32),
        "y_true": np.concatenate(ys).astype(np.int8),
        "rec_idx": np.concatenate(rec_idx),
        "epoca_idx": np.concatenate(epoca_idx),
        "recording_ids": np.array([r.recording_id for r in recs]),
        "subject_codes": np.array([r.subject_code for r in recs]),
    }
    if not (len(pred["probs"]) == len(pred["y_true"]) == len(pred["rec_idx"])):
        raise RuntimeError("El orden de las predicciones no coincide con las grabaciones.")
    return pred


def entrenar_pliegue(
    fold_idx: int,
    train_codes: Sequence[str],
    val_codes: Sequence[str],
    combos_train: Sequence[Combinacion],
    combos_val: Sequence[Combinacion],
    max_epochs: int,
    patience: int,
    emg: Optional[str] = None,
    sin_lstm: bool = False,
    nombre_run: str = "run",
    salida_probs: Optional[Path] = None,
) -> Dict:
    posiciones, cargar, construir_modelo, _ = _contexto(emg, sin_lstm)
    t_pliegue = time.time()
    train_recs = cargar(
        subject_codes=list(train_codes), max_epochs_per_recording=base.MAX_EPOCHS_PER_RECORDING)
    val_recs = cargar(
        subject_codes=list(val_codes), max_epochs_per_recording=base.MAX_EPOCHS_PER_RECORDING)
    t_carga = time.time() - t_pliegue

    class_weights = compute_class_weights(train_recs)  # SOLO desde el train del pliegue
    train_loader = build_flexible_dataloader(train_recs, combos_train, shuffle=True, posiciones=posiciones)
    val_loaders = {
        nombre: build_flexible_dataloader(val_recs, [(nombre, presentes)], shuffle=False, posiciones=posiciones)
        for nombre, presentes in combos_val
    }

    config = TinySleepNetConfig(
        use_rnn=True, seq_length=DEFAULT_SEQ_LENGTH, class_weights=class_weights,
        input_channels=n_canales_entrada(posiciones),
    )
    model = construir_modelo(config).to(base.DEVICE)
    criterion = build_class_weighted_loss(class_weights, device=base.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=base.LEARNING_RATE, betas=base.ADAM_BETAS)

    print(f"  [pliegue {fold_idx}] {len(train_recs)} grabaciones train / {len(val_recs)} val | "
          f"{len(train_loader.dataset)} secuencias de entrenamiento por epoca | "
          f"carga de datos {t_carga:.0f} s")

    mejor_f1, mejor_epoca, mejores_metricas, sin_mejora = -1.0, 0, None, 0
    mejor_estado = None          # pesos de la mejor epoca (solo si se guardan las probabilidades)
    historial: List[Dict] = []

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        train_loss, y_true_tr, y_pred_tr = base.run_epoch(model, train_loader, criterion, optimizer)
        t_train = time.time() - t0

        t1 = time.time()
        metricas: Dict[str, Dict] = {}
        val_loss = 0.0
        for nombre, loader in val_loaders.items():
            loss, y_true, y_pred = base.run_epoch(model, loader, criterion, optimizer=None)
            metricas[nombre] = base.compute_metrics(y_true, y_pred)
            val_loss += loss / len(val_loaders)
        t_val = time.time() - t1

        f1_medio = float(np.mean([m["macro_f1"] for m in metricas.values()]))
        train_f1 = f1_score(y_true_tr, y_pred_tr, average="macro", zero_division=0)
        print(f"  [pliegue {fold_idx}] epoca {epoch}/{max_epochs} | train_loss={train_loss:.4f} "
              f"train_F1={train_f1:.4f} | val_loss={val_loss:.4f} val_F1_medio={f1_medio:.4f} | "
              f"entrenar {t_train:.0f} s, validar {t_val:.0f} s")
        historial.append({"epoca": epoch, "train_loss": train_loss, "train_macro_f1": train_f1,
                          "val_loss": val_loss, "val_macro_f1_medio": f1_medio,
                          "segundos_entrenar": t_train, "segundos_validar": t_val})

        if f1_medio > mejor_f1:
            mejor_f1, mejor_epoca, mejores_metricas, sin_mejora = f1_medio, epoch, metricas, 0
            if salida_probs is not None:
                mejor_estado = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            sin_mejora += 1
            if sin_mejora >= patience:
                print(f"  [pliegue {fold_idx}] parada temprana en la epoca {epoch} "
                      f"(sin mejora en {patience} epocas)")
                break

    dif_max = None
    if salida_probs is not None and mejor_estado is not None:
        # Probabilidades de validacion de la MEJOR epoca, una por combinacion. Se comprueba que
        # reproducen el F1 macro de esa epoca (si no, no corresponden al modelo que se eligio).
        model.load_state_dict(mejor_estado)
        salida_probs.mkdir(parents=True, exist_ok=True)
        dif_max = 0.0
        for nombre, presentes in combos_val:
            pred = probs_validacion(model, val_recs, (nombre, presentes), posiciones)
            np.savez_compressed(salida_probs / f"{nombre_run}_pliegue{fold_idx}__{nombre}.npz", **pred)
            f1 = f1_score(pred["y_true"], pred["probs"].argmax(axis=1), average="macro", zero_division=0)
            dif_max = max(dif_max, abs(f1 - mejores_metricas[nombre]["macro_f1"]))
        print(f"  [pliegue {fold_idx}] probabilidades de validacion guardadas en {salida_probs} "
              f"(diferencia maxima de F1 macro frente a la validacion: {dif_max:.2e})")
        if dif_max > 1e-4:
            print("  [AVISO] Las probabilidades guardadas no reproducen el F1 macro de la mejor epoca.")

    return {
        "pliegue": fold_idx,
        "mejor_epoca": mejor_epoca,
        "val_macro_f1_medio": mejor_f1,
        "metricas_por_combinacion": mejores_metricas,
        "historial": historial,
        "segundos_pliegue": time.time() - t_pliegue,
        "probs_validacion_dif_max": dif_max,
    }


def resumir(salida: Path, nombre_run: str) -> Optional[Dict]:
    """Promedio y desviacion entre pliegues, por combinacion, con los JSON que haya."""
    archivos = sorted(salida.glob(f"{nombre_run}_pliegue*.json"))
    if not archivos:
        return None
    pliegues = [json.load(open(a, encoding="utf-8")) for a in archivos]
    nombres = list(pliegues[0]["metricas_por_combinacion"])
    resumen: Dict = {"run": nombre_run, "n_pliegues": len(pliegues), "combinaciones": {}}
    for n in nombres:
        f1s = [p["metricas_por_combinacion"][n]["macro_f1"] for p in pliegues]
        kap = [p["metricas_por_combinacion"][n]["cohens_kappa"] for p in pliegues]
        resumen["combinaciones"][n] = {
            "macro_f1_media": float(np.mean(f1s)), "macro_f1_desv": float(np.std(f1s)),
            "kappa_media": float(np.mean(kap)), "kappa_desv": float(np.std(kap)),
        }
    medios = [p["val_macro_f1_medio"] for p in pliegues]
    resumen["macro_f1_medio_global"] = {"media": float(np.mean(medios)), "desv": float(np.std(medios))}
    resumen["segundos_totales"] = float(sum(p["segundos_pliegue"] for p in pliegues))
    return resumen


def run_cv_flexible(
    control: Optional[str] = None,
    salida: str = "resultados_flexible",
    max_epochs: Optional[int] = None,
    pliegues: Optional[Sequence[int]] = None,
    paciencia: Optional[int] = None,
    emg: Optional[str] = None,
    sin_lstm: bool = False,
    guardar_probs: bool = True,
) -> Dict:
    """
    control=None  -> modelo flexible, entrenado y validado con todas las combinaciones.
    control="eeg_fpz_cz" (u otra) -> modelo de control con una sola combinacion.
    pliegues: lista de pliegues a correr (por defecto todos).
    emg: None (3 senales) o 'mlp' / 'conv' (modelo con la rama EMG; 12 combinaciones).
         Con control, el control debe incluir el EMG (p. ej. 'eeg_fpz_cz__eeg_pz_oz__eog_horizontal__emg_submental').
    sin_lstm: ablacion, el flexible de 3 senales sin la LSTM.
    guardar_probs: guarda las probabilidades de validacion de la mejor epoca de cada pliegue
         (carpeta probs_validacion/ dentro de `salida`).
    """
    _comprobar_control(control, emg, sin_lstm)
    posiciones, _, _, sufijo = _contexto(emg, sin_lstm)
    base.set_seed(base.CONFIG["seed"])
    max_epochs = max_epochs or base.MAX_EPOCHS
    patience = paciencia or base.EARLY_STOPPING_PATIENCE

    if control is None:
        combos = combinaciones_validas(posiciones)
        nombre_run = "flexible" + sufijo
    else:
        combos = [combinacion_por_nombre(control, posiciones)]
        nombre_run = f"control_{control}"

    salida_dir = Path(salida)
    salida_dir.mkdir(parents=True, exist_ok=True)
    salida_probs = salida_dir / "probs_validacion" if guardar_probs else None

    train_codes, _ = get_train_test_subject_codes()
    folds = make_subject_kfold_splits(train_codes)
    a_correr = list(range(len(folds))) if pliegues is None else list(pliegues)

    print(f"Run: {nombre_run} | combinaciones: {[c[0] for c in combos]}")
    print(f"Dispositivo: {base.DEVICE} | pliegues: {a_correr} | max epocas: {max_epochs} | paciencia: {patience}")

    t_total = time.time()
    for k in a_correr:
        fold_train, fold_val = folds[k]
        print(f"\n=== PLIEGUE {k} ===")
        resultado = entrenar_pliegue(k, fold_train, fold_val, combos, combos, max_epochs, patience, emg=emg,
                                     sin_lstm=sin_lstm, nombre_run=nombre_run, salida_probs=salida_probs)
        with open(salida_dir / f"{nombre_run}_pliegue{k}.json", "w", encoding="utf-8") as f:
            json.dump(resultado, f, indent=2, default=_json_default)
        print(f"  [pliegue {k}] listo en {resultado['segundos_pliegue'] / 60:.1f} min | "
              f"mejor epoca {resultado['mejor_epoca']} | val_F1_medio {resultado['val_macro_f1_medio']:.4f}")

    resumen = resumir(salida_dir, nombre_run)
    with open(salida_dir / f"{nombre_run}_resumen.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, default=_json_default)

    print(f"\n=== RESUMEN {nombre_run} ({resumen['n_pliegues']} pliegues) ===")
    for n, m in resumen["combinaciones"].items():
        print(f"  {n:40s} F1 macro {m['macro_f1_media']:.4f} +/- {m['macro_f1_desv']:.4f} | "
              f"kappa {m['kappa_media']:.4f} +/- {m['kappa_desv']:.4f}")
    g = resumen["macro_f1_medio_global"]
    print(f"  Promedio de combinaciones: {g['media']:.4f} +/- {g['desv']:.4f}")
    print(f"Tiempo de esta corrida: {(time.time() - t_total) / 60:.1f} min")
    return resumen


def epocas_desde_cv(salida: str, nombre_run: str) -> int:
    """Numero de epocas para el entrenamiento final: promedio (redondeado) de la mejor
    epoca de cada pliegue de la validacion cruzada ya hecha."""
    archivos = sorted(Path(salida).glob(f"{nombre_run}_pliegue*.json"))
    if not archivos:
        raise FileNotFoundError(f"No hay JSON de CV de '{nombre_run}' en {salida}; "
                                f"corre antes la validacion cruzada o pasa --epocas-final.")
    mejores = [json.load(open(a, encoding="utf-8"))["mejor_epoca"] for a in archivos]
    return int(round(float(np.mean(mejores))))


def run_final_flexible(
    control: Optional[str] = None,
    salida: str = "resultados_flexible",
    n_epocas: Optional[int] = None,
    emg: Optional[str] = None,
    sin_lstm: bool = False,
    noche_completa: bool = False,
    escalar_epocas: bool = False,
) -> Dict:
    """
    Modelo final: entrena con TODOS los sujetos de train (los mismos que usa la CV; el test
    no se toca), un numero fijo de epocas y sin parada temprana. Guarda los pesos (.pth) y
    un JSON con todo lo que hace falta para volver a construir y usar el modelo.

    n_epocas=None -> promedio de las mejores epocas de la CV guardada en `salida`.
    noche_completa=True -> usa cada grabacion completa en vez de la ventana continua de
        `max_epochs_per_recording` (la CV sigue usando la ventana). El modelo ve ~2,5 veces mas
        secuencias por epoca.
    escalar_epocas=True (solo con noche_completa) -> divide las epocas de la CV por ese cociente,
        para que el numero de pasos de optimizacion se parezca al de la CV. Por defecto NO se
        escala (mismas epocas que la CV, que es lo que exige el criterio "mismas epocas").
    No hay metrica de rendimiento aqui: el rendimiento del modelo es el de la CV.
    """
    base.set_seed(base.CONFIG["seed"])
    _comprobar_control(control, emg, sin_lstm)
    if escalar_epocas and not noche_completa:
        raise ValueError("--escalar-epocas solo tiene sentido con --noche-completa.")
    posiciones, cargar, construir_modelo, sufijo = _contexto(emg, sin_lstm)
    if control is None:
        combos, nombre_run = combinaciones_validas(posiciones), "flexible" + sufijo
    else:
        combos, nombre_run = [combinacion_por_nombre(control, posiciones)], f"control_{control}"
    epocas_cv = n_epocas or epocas_desde_cv(salida, nombre_run)
    n_epocas = epocas_cv
    ventana = None if noche_completa else base.MAX_EPOCHS_PER_RECORDING

    salida_dir = Path(salida)
    salida_dir.mkdir(parents=True, exist_ok=True)
    train_codes, _ = get_train_test_subject_codes()

    t0 = time.time()
    recs = cargar(subject_codes=list(train_codes), max_epochs_per_recording=ventana)
    class_weights = compute_class_weights(recs)
    loader = build_flexible_dataloader(recs, combos, shuffle=True, posiciones=posiciones)
    factor_escala = None
    if escalar_epocas:
        # Cociente (epocas usadas en la CV) / (epocas de noche completa), por grabacion.
        tope = base.MAX_EPOCHS_PER_RECORDING
        factor_escala = float(sum(min(r.X.shape[0], tope) for r in recs) / sum(r.X.shape[0] for r in recs))
        n_epocas = max(1, int(round(epocas_cv * factor_escala)))

    config = TinySleepNetConfig(
        use_rnn=True, seq_length=DEFAULT_SEQ_LENGTH, class_weights=class_weights,
        input_channels=n_canales_entrada(posiciones),
    )
    model = construir_modelo(config).to(base.DEVICE)
    criterion = build_class_weighted_loss(class_weights, device=base.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=base.LEARNING_RATE, betas=base.ADAM_BETAS)

    print(f"Modelo final '{nombre_run}': {len(recs)} grabaciones "
          f"({'noche completa' if noche_completa else f'ventana de {ventana} epocas'}) | "
          f"{len(loader.dataset)} secuencias por epoca | {n_epocas} epocas "
          f"(CV: {epocas_cv}{'' if factor_escala is None else f', escala {factor_escala:.3f}'}) | "
          f"combinaciones: {[c[0] for c in combos]}")

    historial: List[Dict] = []
    for epoch in range(1, n_epocas + 1):
        t1 = time.time()
        loss, y_true, y_pred = base.run_epoch(model, loader, criterion, optimizer)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        print(f"  epoca {epoch}/{n_epocas} | train_loss={loss:.4f} train_F1={f1:.4f} | "
              f"{time.time() - t1:.0f} s")
        historial.append({"epoca": epoch, "train_loss": loss, "train_macro_f1": f1,
                          "segundos": time.time() - t1})

    ruta_pesos = salida_dir / f"{nombre_run}_final.pth"
    torch.save(model.state_dict(), ruta_pesos)
    info = {
        "run": nombre_run,
        "n_epocas": n_epocas,
        "epocas_cv": epocas_cv,
        "noche_completa": noche_completa,
        "max_epochs_per_recording": ventana,
        "factor_escala_epocas": factor_escala,
        "n_epocas_totales_entrenamiento": int(sum(r.X.shape[0] for r in recs)),
        "sin_lstm": sin_lstm,
        "n_grabaciones": len(recs),
        "combinaciones": [c[0] for c in combos],
        "n_canales_entrada": n_canales_entrada(posiciones),
        "emg": emg,   # None, 'mlp' o 'conv': decide la clase del modelo al cargar los pesos
        "seq_length": DEFAULT_SEQ_LENGTH,
        "clases": ["W", "N1", "N2", "N3", "REM"],
        "class_weights": class_weights,
        "pesos": ruta_pesos.name,
        "historial": historial,
        "segundos_totales": time.time() - t0,
    }
    with open(salida_dir / f"{nombre_run}_final.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, default=_json_default)
    print(f"\nPesos guardados en {ruta_pesos} | tiempo total {(time.time() - t0) / 60:.1f} min")
    return info
