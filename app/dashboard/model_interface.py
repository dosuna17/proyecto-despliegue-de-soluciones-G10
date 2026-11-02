# Capa de conexión entre el dashboard y los datos/modelo.
#
# Estructura del archivo (en este orden):
#   1. Configuración (variables de entorno)
#   2. Variantes de modelo disponibles  <-- AGREGAR MODELOS NUEVOS AQUÍ
#   3. Carga de datos reales (Sleep-EDFx)
#   4. Predicción (llamadas a la API)
#   5. Lectura de resultados ya calculados

import os
import sys
from pathlib import Path

import numpy as np
import requests

# app/dashboard/model_interface.py -> parents[2] = raíz del repo.
# Mismo patrón que ya usa train_best_xgboost_mlflow.py para poder
# importar src.* sin depender de desde dónde se lance el script.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import find_subject_pairs, load_subject
from src.data.segmentation import extract_epochs, trim_wake_edges, EpochDataset

#-------------------------------------------------------------------------------------------------------------------
# 1. Configuración
#-------------------------------------------------------------------------------------------------------------------
# Carpeta local donde cada quien tenga descargado el dataset Sleep-EDFx.
# Se define por variable de entorno (no hay valor por defecto a propósito,
# para no dejar la ruta personal de nadie quemada en el código):
#   $env:SLEEP_STAGING_DATA_ROOT = "C:\ruta\a\sleep-edf-database-expanded-1.0.0"
DATA_ROOT = os.getenv("SLEEP_STAGING_DATA_ROOT")
if DATA_ROOT is None:
    raise RuntimeError(
        "Falta configurar la variable de entorno SLEEP_STAGING_DATA_ROOT "
        "con la ruta local donde tengas descargado el dataset Sleep-EDFx."
    )

# URL de la API de predicción. Por defecto apunta a una API corriendo local
# (uvicorn app.api.main:app --reload); se puede apuntar a la EC2 real con:
#   $env:SLEEP_STAGING_API_URL = "http://<ip-ec2>:8000"
API_BASE_URL = os.getenv("SLEEP_STAGING_API_URL", "http://localhost:8000")

# /predict-tinysleepnet exige exactamente este número de épocas por llamada.
BATCH_SIZE = 15

#-------------------------------------------------------------------------------------------------------------------
# 2. Modelos disponibles
#-------------------------------------------------------------------------------------------------------------------
# Cada entrada describe un modelo entrenado y los canales que necesita.
# La Pantalla 1 usa esto para habilitar solo los checkboxes de canal que
# algún modelo soporta; la Pantalla 2 usa esto para ofrecer solo los
# modelos que se pueden calcular con los canales que el usuario cargó.
#
# Para agregar un modelo nuevo cuando el equipo lo entregue:
#   1. Agrega una entrada nueva al diccionario de abajo.
#   2. No hace falta tocar ninguna otra función de este archivo ni de app.py.
MODELOS_DISPONIBLES = {
    "baseline": {
        "etiqueta": "XGBoost (línea base)",
        "canales_requeridos": ["EEG Fpz-Cz", "EEG Pz-Oz"],
        "model_artifact": "files/best-xgboost-multicanal.joblib",
    },
    "avanzado": {
        "etiqueta": "TinySleepNet",
        "canales_requeridos": ["EEG Fpz-Cz"],
        "canales_entrada": ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal"],
        "model_artifact": "flexible_final.pth",
    },  # <- se habilita cuando el cnn.py de Gabriel llegue a main
}


# Unión de todos los canales que al menos un modelo disponible necesita hoy,
# más EEG Fpz-Cz (siempre obligatorio). La Pantalla 1 usa esto para
# habilitar/deshabilitar checkboxes de canal automáticamente.
def canales_soportados():
    soportados = {"EEG Fpz-Cz"}
    for info in MODELOS_DISPONIBLES.values():
        soportados.update(info["canales_requeridos"])
    return soportados


# Dado el conjunto de canales que el usuario cargó, devuelve los modelos
# que sí se pueden calcular con esos canales (sus canales_requeridos son
# un subconjunto de lo cargado). El orden respeta MODELOS_DISPONIBLES, así
# que "baseline" queda primero -- es el modelo por defecto en Pantalla 2.
def modelos_disponibles(canales_cargados):
    disponibles = set(canales_cargados)
    return [
        {"modelo": clave, **info}
        for clave, info in MODELOS_DISPONIBLES.items()
        if set(info["canales_requeridos"]).issubset(disponibles)
    ]


# Recorta un EpochDataset a un subconjunto de canales, preservando el orden
# en que el modelo los espera. Se usa cuando el usuario cargó más canales
# de los que un modelo puntual necesita.
def subset_canales(dataset, canales_requeridos):
    indices = [dataset.channel_names.index(nombre) for nombre in canales_requeridos]
    return EpochDataset(
        recording_id=dataset.recording_id,
        subject_code=dataset.subject_code,
        night=dataset.night,
        X=dataset.X[:, indices, :],
        y=dataset.y,
        recording_ids=dataset.recording_ids,
        subject_codes=dataset.subject_codes,
        channel_names=canales_requeridos,
        sfreq=dataset.sfreq,
        signal_unit=dataset.signal_unit,
        dropped_epochs=dataset.dropped_epochs,
    )


def preparar_entrada_tinysleepnet(dataset):
    """Construye 3 señales físicas y 3 marcas de presencia en orden fijo."""
    signal_channels = ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal"]
    presence_channels = [f"presence {name}" for name in signal_channels]
    n_epochs, _, n_samples = dataset.X.shape
    signals, presence = [], []
    for channel_name in signal_channels:
        if channel_name in dataset.channel_names:
            signals.append(dataset.X[:, dataset.channel_names.index(channel_name), :])
            presence.append(np.ones((n_epochs, n_samples), dtype=np.float32))
        else:
            signals.append(np.zeros((n_epochs, n_samples), dtype=np.float32))
            presence.append(np.zeros((n_epochs, n_samples), dtype=np.float32))
    X = np.stack([*signals, *presence], axis=1).astype(np.float32, copy=False)
    if not np.any(X[:, 3:5, 0] == 1.0):
        raise ValueError("TinySleepNet requiere al menos un canal EEG.")
    return EpochDataset(
        recording_id=dataset.recording_id,
        subject_code=dataset.subject_code,
        night=dataset.night,
        X=X,
        y=dataset.y,
        recording_ids=dataset.recording_ids,
        subject_codes=dataset.subject_codes,
        channel_names=signal_channels + presence_channels,
        sfreq=dataset.sfreq,
        signal_unit=dataset.signal_unit,
        dropped_epochs=dataset.dropped_epochs,
    )

#-------------------------------------------------------------------------------------------------------------------
# 3. Carga de datos reales (Sleep-EDFx)
#-------------------------------------------------------------------------------------------------------------------
# Se asume que los estudios de Sleep Cassette empiezan por "SC" y los de
# Sleep Telemetry por "ST" (convención de nombres del propio dataset).
def _carpeta_para_estudio(estudio_id):
    if estudio_id.startswith("SC"):
        return os.path.join(DATA_ROOT, "sleep-cassette")
    if estudio_id.startswith("ST"):
        return os.path.join(DATA_ROOT, "sleep-telemetry")
    raise ValueError(f"Estudio no reconocido (debe empezar por SC o ST): {estudio_id}")


# Carga un estudio real del dataset Sleep-EDFx y lo segmenta en épocas,
# aplicando el mismo recorte de vigilia usado en el entrenamiento.
def cargar_estudio(estudio_id, canales_activos):
    carpeta = _carpeta_para_estudio(estudio_id)
    pares = find_subject_pairs(carpeta)

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
    return trim_wake_edges(dataset)

#-------------------------------------------------------------------------------------------------------------------
# 4. Predicción (llamadas a la API)
#-------------------------------------------------------------------------------------------------------------------
# Recorta un EpochDataset a las primeras n_epocas, preservando su forma.
# Se usa cuando el total de épocas no es múltiplo de BATCH_SIZE.
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


# POST a la API real. Separa el error en dos capas: el detalle técnico completo se imprime en la consola
# del servidor (para que el equipo pueda diagnosticarlo), y lo que se le devuelve a la interfaz es siempre
# un mensaje en leguaje llano, sin rutas de archivos, tracksback ni detalles internos de la API. 
# Esto evita que un usuario final vea información sensible o confusa.
def _llamar_api(endpoint, payload, timeout):
    url = f"{API_BASE_URL}{endpoint}"
    try:
        respuesta = requests.post(url, json=payload, timeout=timeout)
        respuesta.raise_for_status()
        return respuesta.json()
    except requests.exceptions.ConnectionError as error:
        print(f"[model_interface] No se pudo conectar a {url}: {error}")
        raise RuntimeError(
            "No fue posible conectarse al servicio de predicción en este "
            "momento. Intente nuevamente en unos minutos o contacte al "
            "equipo técnico si el problema persiste."
        ) from error
    except requests.exceptions.Timeout as error:
        print(f"[model_interface] Timeout llamando a {url}: {error}")
        raise RuntimeError(
            "El servicio de predicción tardó demasiado en responder. "
            "Intente nuevamente; si el problema persiste, contacte al "
            "equipo técnico."
        ) from error
    except requests.exceptions.HTTPError as error:
        detalle = respuesta.json().get("detail", {})
        mensaje_tecnico = detalle.get("message", str(error)) if isinstance(detalle, dict) else str(error)
        print(f"[model_interface] Error HTTP en {url}: {mensaje_tecnico}")
        raise RuntimeError(
            "El servicio de predicción no pudo procesar la solicitud. "
            "Contacte al equipo técnico si el problema persiste."
        ) from error


# Llama a /predict-tinysleepnet en bloques de exactamente 15 épocas (TinySleepNet).
def _predecir_avanzado(dataset):
    n_epocas = len(dataset.y)
    n_epocas_usables = (n_epocas // BATCH_SIZE) * BATCH_SIZE
    if n_epocas_usables == 0:
        raise ValueError(
            f"El estudio tiene {n_epocas} épocas, menos que el bloque mínimo "
            f"de {BATCH_SIZE} que exige /predict-tinysleepnet."
        )

    resultados = []
    for inicio in range(0, n_epocas_usables, BATCH_SIZE):
        bloque = dataset.X[inicio:inicio + BATCH_SIZE]  # (15, n_canales, 3000)
        payload = {
            "channel_names": dataset.channel_names,
            "sampling_rate_hz": int(dataset.sfreq),
            "signal_unit": dataset.signal_unit,
            "model_artifact": MODELOS_DISPONIBLES["avanzado"]["model_artifact"],
            "epochs": bloque.tolist(),
        }
        cuerpo = _llamar_api("/predict-tinysleepnet", payload, timeout=30)
        for i, pred in enumerate(cuerpo["predictions"]):
            resultados.append({
                "prediccion": pred["predicted_stage"],
                "confianza": pred["confidence"],
                "probabilidades": pred["probabilities"],
                "etiqueta_experta": dataset.y[inicio + i],
            })

    dataset_usado = _recortar_dataset(dataset, n_epocas_usables)
    return dataset_usado, resultados


# Llama a /predict-xgboost con todas las épocas de una sola vez (sin límite de bloque).
def _predecir_baseline(dataset):
    payload = {
        "channel_names": dataset.channel_names,
        "sampling_rate_hz": int(dataset.sfreq),
        "signal_unit": dataset.signal_unit,
        "model_artifact": MODELOS_DISPONIBLES["baseline"]["model_artifact"],
        "epochs": dataset.X.tolist(),
    }
    cuerpo = _llamar_api("/predict-xgboost", payload, timeout=180)
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


# Punto de entrada único para predecir un estudio completo. Devuelve
# (dataset_usado, resultados); dataset_usado puede tener menos épocas que
# el original si modelo="avanzado" y el total no era múltiplo de 15.
def predecir_estudio(dataset, modelo="avanzado"):
    if modelo == "avanzado":
        return _predecir_avanzado(dataset)
    if modelo == "baseline":
        return _predecir_baseline(dataset)
    raise ValueError(f"Modelo no reconocido: {modelo}")

#-------------------------------------------------------------------------------------------------------------------
# 5. Lectura de resultados ya calculados
#-------------------------------------------------------------------------------------------------------------------
# Lee la predicción ya calculada para una época específica (no llama a la API).
def obtener_prediccion(resultados, indice):
    return resultados[indice]
