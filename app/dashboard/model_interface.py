# Capa de conexión entre el dashboard y los datos/modelo.
#
# COMMIT 1 de 2: reemplaza la carga de datos simulados (mock_data.py) por
# datos reales del dataset Sleep-EDFx (señal real + etiquetas expertas
# reales, vía loader.py + segmentation.py de Santiago).
#
# La predicción todavía NO llama a la API real -- se usa temporalmente
# la etiqueta experta como "predicción" (confianza fija) para poder
# probar el flujo de datos reales de forma aislada. Esto se reemplaza
# en el commit 2 por la llamada real a /predict o /predict-xgboost.

import os
import sys
import requests
from pathlib import Path

# app/dashboard/model_interface.py -> parents[2] = raíz del repo.
# Mismo patrón que ya usa train_best_xgboost_mlflow.py para poder
# importar src.* sin depender de desde dónde se lance el script.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import find_subject_pairs, load_subject
from src.data.segmentation import extract_epochs,trim_wake_edges, CLASS_NAMES,  EpochDataset

DATA_ROOT = os.getenv("SLEEP_STAGING_DATA_ROOT")
if DATA_ROOT is None:
    raise RuntimeError(
        "Falta configurar la variable de entorno SLEEP_STAGING_DATA_ROOT "
        "con la ruta local donde tengas descargado el dataset Sleep-EDFx."
    )

# Placeholder de confianza mientras el commit 2 conecta la API real.
_CONFIANZA_PROVISIONAL = 0.99

# Se asume que los estudios de Sleep-Cassette empiezan por "SC" y los de Sleep-Telemetry por "ST".
def _carpeta_para_estudio(estudio_id: str) -> str:
    if estudio_id.startswith("SC"):
        return os.path.join(DATA_ROOT, "sleep-cassette")
    if estudio_id.startswith("ST"):
        return os.path.join(DATA_ROOT, "sleep-telemetry")
    raise ValueError(f"Estudio no reconocido (debe empezar por SC o ST): {estudio_id}")

# Carga un estudio real del dataset Sleep-EDFx y lo segmenta en épocas.
def cargar_estudio(estudio_id, canales_activos):
    carpeta = _carpeta_para_estudio(estudio_id)
    pares = find_subject_pairs(carpeta)
    # Busca el par PSG/Hypnogram cuyo nombre de archivo empieza con los primeros 6 caracteres del estudio_id.
    par = next(
        ((psg, hypno) for psg, hypno in pares if os.path.basename(psg).startswith(estudio_id[:6])),
        None,
    )
    if par is None:
        raise FileNotFoundError(
            f"No se encontró el par PSG/Hypnogram para '{estudio_id}' en {carpeta}"
        )

    psg_path, hypno_path = par
    sujeto = load_subject(psg_path, hypno_path, subject_id=estudio_id)
    dataset = extract_epochs(
        sujeto.raw,
        sujeto.annotations,
        recording_id=estudio_id,
        channel_picks=canales_activos,
    )
    dataset = trim_wake_edges(dataset)  # mismo recorte de vigilia usado en el entrenamiento
    return dataset

# COMMIT 1 de 2: reemplaza la predicción simulada por la predicción real vía API.
API_BASE_URL = os.getenv("SLEEP_STAGING_API_URL", "http://localhost:8000")
BATCH_SIZE = 15  # /predict exige exactamente 15 épocas por llamada

# Recorta un EpochDataset a las primeras n_epocas, preservando su forma.
def _recortar_dataset(dataset, n_epocas):
    return EpochDataset(
        recording_id=dataset.recording_id,
        subject_code=dataset.subject_code,
        night=dataset.night,
        X=dataset.X[:n_epocas],
        y=dataset.y[:n_epocas],
        recording_ids=dataset.recording_ids[:n_epocas],
        subject_codes=dataset.subject_codes[:n_epocas],
        channel_names=dataset.channel_names,
        sfreq=dataset.sfreq,
        signal_unit=dataset.signal_unit,
        dropped_epochs=dataset.dropped_epochs + (len(dataset.y) - n_epocas),
    )

# Post a la API real, traduciendo errores HTTP al mensaje que la API ya da.
def _llamar_api(endpoint, payload, timeout):
    try:
        respuesta = requests.post(f"{API_BASE_URL}{endpoint}", json=payload, timeout=timeout)
        respuesta.raise_for_status()
        return respuesta.json()
    except requests.exceptions.ConnectionError as error:
        raise RuntimeError(
            f"No se pudo conectar a la API en {API_BASE_URL}. "
            "¿Está corriendo `uvicorn app.api.main:app --reload`?"
        ) from error
    except requests.exceptions.HTTPError as error:
        detalle = respuesta.json().get("detail", {})
        mensaje = detalle.get("message", str(error)) if isinstance(detalle, dict) else str(error)
        raise RuntimeError(f"La API respondió con error: {mensaje}") from error

# Llama a /predict en bloques de 15 épocas (TinySleepNet).
def _predecir_avanzado(dataset):
    n_epocas = len(dataset.y)
    n_epocas_usables = (n_epocas // BATCH_SIZE) * BATCH_SIZE
    if n_epocas_usables == 0:
        raise ValueError(
            f"El estudio tiene {n_epocas} épocas, menos que el bloque mínimo "
            f"de {BATCH_SIZE} que exige /predict."
        )

    resultados = []
    for inicio in range(0, n_epocas_usables, BATCH_SIZE):
        bloque = dataset.X[inicio:inicio + BATCH_SIZE]  # (15, n_canales, 3000)
        payload = {
            "channel_names": dataset.channel_names,
            "sampling_rate_hz": int(dataset.sfreq),
            "signal_unit": dataset.signal_unit,
            "epochs": bloque.tolist(),
        }
        cuerpo = _llamar_api("/predict", payload, timeout=30)
        for i, pred in enumerate(cuerpo["predictions"]):
            resultados.append({
                "prediccion": pred["predicted_stage"],
                "confianza": pred["confidence"],
                "probabilidades": pred["probabilities"],
                "etiqueta_experta": dataset.y[inicio + i],
            })

    dataset_usado = _recortar_dataset(dataset, n_epocas_usables)
    return dataset_usado, resultados

# llama a /predict-xgboost con todas las épocas de una sola vez (sin límite de 15).
def _predecir_baseline(dataset):
    payload = {
        "channel_names": dataset.channel_names,
        "sampling_rate_hz": int(dataset.sfreq),
        "signal_unit": dataset.signal_unit,
        "epochs": dataset.X.tolist(),
    }
    cuerpo = _llamar_api("/predict-xgboost", payload, timeout=60)
    resultados = [
        {
            "prediccion": pred["predicted_stage"],
            "confianza": pred["confidence"],
            "probabilidades": pred["probabilities"],
            "etiqueta_experta": dataset.y[indice],
        }
        for indice, pred in enumerate(cuerpo["predictions"])
    ]
    return dataset, resultados  # XGBoost no recorta -- no tiene límite de bloque


def predecir_estudio(dataset, modelo="avanzado"):
    """
    Llama a la API real para predecir todas las épocas de un estudio.

    Devuelve (dataset_usado, resultados). dataset_usado puede tener menos
    épocas que el original si modelo="avanzado" y el total no era múltiplo
    de 15 (la API exige bloques exactos); con modelo="baseline" siempre
    coincide con el dataset original.
    """
    if modelo == "avanzado":
        return _predecir_avanzado(dataset)
    if modelo == "baseline":
        return _predecir_baseline(dataset)
    raise ValueError(f"Modelo no reconocido: {modelo}")

# Lee la predicción ya calculada para una época específica (no llama a la API).
def obtener_prediccion(resultados, indice):
    return resultados[indice]