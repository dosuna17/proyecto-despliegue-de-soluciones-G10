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

from src.data.loader import find_subject_pairs, load_subject
from src.data.segmentation import extract_epochs,trim_wake_edges, CLASS_NAMES

DATA_ROOT = r"C:\Users\mjgar\Escritorio\Maestria\Semestre_4\Ciclo_1\Proyecto_Desarrollo_Soluciones\MicroProyecto\Entrega_1\sleep-edf-database-expanded-1.0.0"

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
def predecir_estudio(dataset, modelo="avanzado"):
    """
    PROVISIONAL (commit 1): usa la etiqueta experta como si fuera la
    predicción del modelo, con confianza fija, solo para poder probar
    el resto del tablero con datos reales mientras se conecta la API
    de verdad en el commit 2.

    El parámetro `modelo` ya está aquí para no tener que cambiar la
    firma de la función cuando se conecte la API real.
    """
    n_epocas = len(dataset.y)
    resto = (1 - _CONFIANZA_PROVISIONAL) / (len(CLASS_NAMES) - 1)
    resultados = []
    for indice in range(n_epocas):
        prediccion = dataset.y[indice]
        probabilidades = {
            clase: (_CONFIANZA_PROVISIONAL if clase == prediccion else resto)
            for clase in CLASS_NAMES
        }
        resultados.append({
            "prediccion": prediccion,
            "confianza": _CONFIANZA_PROVISIONAL,
            "probabilidades": probabilidades,
            "etiqueta_experta": dataset.y[indice],
        })
    return resultados

# Lee la predicción ya calculada para una época específica (no llama a la API).
def obtener_prediccion(resultados, indice):
    return resultados[indice]