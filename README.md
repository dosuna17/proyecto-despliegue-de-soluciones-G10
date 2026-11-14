# Clasificación automática de estadios del sueño

Sistema de apoyo a la revisión de polisomnografías: clasifica cada época de 30 segundos en un estadio del sueño, dibuja el hipnograma y señala las épocas en las que el modelo tiene baja confianza para que el técnico concentre allí su revisión. Funciona con cualquier combinación de señales que incluya al menos un canal de EEG.

Proyecto de grado de la Maestría en Inteligencia Artificial (Universidad de los Andes), sobre una propuesta del Grupo de Ingeniería Biomédica (IBIO).

> **Uso previsto.** Es una herramienta de apoyo a la revisión, no de diagnóstico. Toda predicción debe ser validada por personal entrenado.

## Contenido

1. [Problema y solución](#1-problema-y-solución)
2. [Enlaces](#2-enlaces)
3. [Arquitectura](#3-arquitectura)
4. [Instrucciones de uso](#4-instrucciones-de-uso)
5. [Ejemplos de entrada y salida](#5-ejemplos-de-entrada-y-salida)
6. [Modelos: tamaño, hardware y tiempos de inferencia](#6-modelos-tamaño-hardware-y-tiempos-de-inferencia)
7. [Dependencias y entorno](#7-dependencias-y-entorno)
8. [Ejecutar en local](#8-ejecutar-en-local)
9. [Reproducir el despliegue](#9-reproducir-el-despliegue)
10. [Reproducir el entrenamiento](#10-reproducir-el-entrenamiento)
11. [Parámetros configurables](#11-parámetros-configurables)
12. [Credenciales de ejemplo](#12-credenciales-de-ejemplo)
13. [Estructura del repositorio](#13-estructura-del-repositorio)
14. [Equipo](#14-equipo)

## 1. Problema y solución

**Problema.** La estadificación del sueño se hace a mano, época por época: una noche tiene entre 800 y 1200 épocas de 30 s. Es lenta y variable: dos expertos coinciden en cerca del 83 % de las épocas, y en el estadio N1 apenas en el 63 %.

**Solución.** Un modelo de aprendizaje de máquina asigna a cada época uno de cinco estadios, y una aplicación web muestra el resultado y las épocas que conviene revisar.

**Alcance.**

| Aspecto | Decisión |
|---|---|
| Clases | 5: vigilia (W), N1, N2, N3 y REM. Las etapas 3 y 4 de Rechtschaffen y Kales se unen en N3 (criterio AASM). Las épocas de movimiento (M) y las sin clasificar ("?") se excluyen del entrenamiento del clasificador |
| Datos | Subconjunto Sleep Cassette de [Sleep-EDF Expanded](https://physionet.org/content/sleep-edfx/1.0.0/): 153 registros de 78 personas sanas |
| Señales | Al menos un EEG (Fpz-Cz, Pz-Oz o ambos), con EOG y EMG opcionales: 12 combinaciones |
| Evaluación | Partición por persona (ninguna persona aparece en entrenamiento y prueba) y comparación con el acuerdo entre expertos humanos |

## 2. Enlaces

| Recurso | Enlace |
|---|---|
| Aplicación (tablero) | <!-- PENDIENTE (Daniel): URL pública del tablero --> _Por publicar_ |
| API y documentación interactiva | <!-- PENDIENTE (Daniel): URL de la API + /docs --> _Por publicar_ |
| Conjunto de datos | [Sleep-EDF Expanded en PhysioNet](https://physionet.org/content/sleep-edfx/1.0.0/) |
| Manual de usuario del tablero | [app/dashboard/manuales](app/dashboard/manuales) |

## 3. Arquitectura

<!-- PENDIENTE (María José): diagrama de arquitectura (reports/figures/arquitectura.png) -->

| Componente | Tecnología | Función |
|---|---|---|
| Tablero | Streamlit | Selección del estudio y de las señales; hipnograma, distribución de estadios y revisión por época |
| API | FastAPI | Recibe las épocas y las señales disponibles y devuelve, por época, el estadio predicho, la confianza y las probabilidades |
| Modelos | PyTorch (TinySleepNet) y XGBoost | La API carga desde MLflow el modelo marcado como campeón |
| Contenedores | Docker y Docker Compose | Un servicio para la API y otro para el tablero |
| Nube | AWS EC2 | Aloja los contenedores <!-- PENDIENTE (Daniel): tamaño de la instancia y región --> |
| Experimentos y datos | MLflow y DVC sobre S3 | Registro de experimentos, versionado de datos y almacenamiento de los modelos |

## 4. Instrucciones de uso

1. Abrir el tablero (sección 2).
2. **Pantalla 1:** elegir un estudio de demostración y activar las señales disponibles (al menos un EEG). Solo se ofrecen las combinaciones con un modelo entrenado.
3. Elegir el modelo y ejecutar la clasificación.
4. **Pantalla 2:** revisar el resumen del estudio (eficiencia de sueño y porcentaje por estadio), el hipnograma y la predicción de cada época, con su confianza y sus probabilidades. Las épocas de baja confianza aparecen resaltadas.

<!-- PENDIENTE (María José): capturas de las dos pantallas de la versión desplegada -->

## 5. Ejemplos de entrada y salida

<!-- PENDIENTE (Daniel + María José): ejemplo real de /predict según el contrato acordado de la API -->

**Entrada:** épocas de 30 s a 100 Hz (3000 muestras por canal, en microvoltios) y los nombres de los canales.

**Salida:** por cada época, el estadio predicho, la confianza y la probabilidad de cada clase.

```json
{
  "_comentario": "Ejemplo ilustrativo; se reemplaza por una respuesta real de la API",
  "modelo": "tinysleepnet_fpzcz_eog",
  "canales": ["EEG Fpz-Cz", "EOG horizontal"],
  "predicciones": [
    {"epoca": 0, "estadio": "W",  "confianza": 0.97, "probabilidades": {"W": 0.97, "N1": 0.02, "N2": 0.01, "N3": 0.00, "REM": 0.00}},
    {"epoca": 1, "estadio": "N1", "confianza": 0.58, "probabilidades": {"W": 0.31, "N1": 0.58, "N2": 0.09, "N3": 0.00, "REM": 0.02}}
  ]
}
```

## 6. Modelos: tamaño, hardware y tiempos de inferencia

Métricas en el conjunto de prueba (personas no vistas), 5 clases.

| Modelo | Canales | Exactitud | F1 macro | Kappa | Tamaño | Tiempo por estudio |
|---|---|---|---|---|---|---|
| TinySleepNet | Fpz-Cz + EOG | 0.811 | 0.760 | 0.742 | <!-- PENDIENTE (Daniel) --> | <!-- PENDIENTE (Daniel) --> |
| TinySleepNet | Fpz-Cz | 0.769 | 0.723 | 0.688 | | |
| XGBoost | <!-- PENDIENTE (Santiago): canales --> | 0.728 | 0.671 | 0.633 | | |
| XGBoost | Fpz-Cz | 0.670 | 0.610 | 0.560 | | |

TinySleepNet tiene 1,516,421 parámetros. Hardware de inferencia: CPU, sin GPU. <!-- PENDIENTE (Daniel): memoria en reposo y al predecir, medida en la instancia EC2 -->

## 7. Dependencias y entorno

- Python 3.12
- Dependencias fijadas en [`requirements.txt`](requirements.txt): PyTorch, XGBoost, scikit-learn, MNE, FastAPI, Streamlit, MLflow y DVC, entre otras.
- Docker y Docker Compose para ejecutar la aplicación.

## 8. Ejecutar en local

```bash
git clone <URL del repositorio>
cd <carpeta del repositorio>
cp .env.example .env        # y completar los valores
docker compose up --build
```

- Tablero: http://localhost:8501
- API: http://localhost:8000/docs

<!-- PENDIENTE (Daniel): confirmar los pasos con las imágenes separadas de la API y el tablero -->

## 9. Reproducir el despliegue

<!-- PENDIENTE (Daniel): pasos para crear la instancia EC2, la IP fija, el grupo de seguridad y el .env del servidor -->

## 10. Reproducir el entrenamiento

Cada etapa tiene un comando y su configuración en `configs/`. Cada corrida queda registrada en MLflow.

| Paso | Comando | Responsable |
|---|---|---|
| 1. Descargar los datos | `python scripts/01_download_data.py` | Daniel |
| 2. Construir el conjunto de épocas | <!-- PENDIENTE (Santiago) --> `python scripts/02_build_dataset.py` | Santiago |
| 3. Entrenar los modelos clásicos | <!-- PENDIENTE (Santiago) --> `python scripts/04_train_classical.py` | Santiago |
| 4. Entrenar TinySleepNet | `python scripts/05_train_tinysleepnet.py --modo cv` (validación cruzada) o `--modo final` | Gabriel |
| 5. Entrenar la versión flexible de canales (validación cruzada y modelo final) | `python scripts/06_train_flexible.py` (validación cruzada) y luego `python scripts/06_train_flexible.py --final`. Opciones: `--emg mlp` agrega el EMG; `--control <combinación>` entrena un control (con EMG: `--emg mlp --control eeg_fpz_cz__eeg_pz_oz__eog_horizontal__emg_submental`); `--sin-lstm` entrena la ablación sin LSTM; `--final --noche-completa` entrena el final con las noches completas (la validación cruzada usa la ventana de `max_epochs_per_recording`). La validación cruzada guarda las probabilidades de validación en `<salida>/probs_validacion/` | Gabriel |
| 6. Evaluar en test (una sola vez, con autorización) | `python scripts/08_evaluar_test.py --modelos <carpeta> --probs-validacion <carpetas de la CV> --modo test --confirmo-test-unico` (antes, `--modo prueba` como ensayo). Los umbrales de confianza se fijan con las probabilidades de validación, antes de abrir el test: el global (82,6 %) y, por clase, la meta de SleepFM (sensibilidad, otra métrica) menos δ aplicada a la exactitud de lo aceptado en cada clase predicha. `--solo-umbrales` los calcula y reporta en validación para δ = 0, 5 y 10 pts sin cargar modelos ni datos; el δ elegido (`--delta`, por defecto 5) queda escrito en el candado y el test solo verifica ese δ. Una clase cuya meta no se alcanza con ningún umbral queda «inalcanzable» y todas sus épocas van a revisión. El candado `TEST_YA_EJECUTADO.json` se crea al empezar (ruta `--candado` o `TEST_LOCK_PATH`); si la corrida se corta, se completa con `--reanudar` | Gabriel |
| 7. Subir a MLflow | `scripts/07_log_mlflow_flexible.py`: ver la subsección siguiente | Gabriel |

### Registrar en MLflow y marcar el modelo campeón

La API no recibe archivos: busca en MLflow el último run `tinysleepnet_flexible_cv` con la etiqueta `champion=true` y la carpeta `modelo_final/flexible_final.pth` (ver `configs/models/model_registry.yaml`). Por eso los pesos, las métricas de validación cruzada y las de test viven en **un mismo run**. Cuatro comandos, siempre con `MLFLOW_TRACKING_URI` definida en el entorno:

```bash
# 1. Validación cruzada: crea el run y escribe su run_id
python scripts/07_log_mlflow_flexible.py --carpeta resultados --run flexible

# 2. Modelo final: agrega los pesos y el JSON al mismo run (modelo_final/)
python scripts/07_log_mlflow_flexible.py --carpeta resultados --run flexible --final --cv-run-id <run_id>

# 3. Test (solo si el candado TEST_YA_EJECUTADO.json está "completo"): agrega las métricas test_* y las figuras
python scripts/07_log_mlflow_flexible.py --run flexible --cv-run-id <run_id> --test-carpeta resultados_test --candado <ruta del candado>

# 4. Campeón: etiqueta ese run champion=true y se la quita a todos los demás del experimento
python scripts/07_log_mlflow_flexible.py --run flexible --cv-run-id <run_id> --campeon
```

Los pasos 2 a 4 se pueden pedir en un solo comando. `--run` es `flexible`, `flexible_emg_mlp`, `flexible_sin_lstm` o `control_<combinación>`.

**Criterio del campeón** (se fija antes de mirar el test; el test solo reporta): el modelo flexible con mayor F1 macro medio en la validación cruzada sobre las 6 combinaciones comunes (las que no usan EMG), tomando el run más reciente de `flexible` y de `flexible_emg_mlp`. Solo esos dos pueden ser campeón; los controles y la ablación sin LSTM son comparaciones. `--campeon` imprime el ranking, se detiene si el run elegido no es el primero (salvo `--forzar`, que queda anotado en el run) y guarda el criterio en la etiqueta `criterio_campeon`. Si el campeón no es `flexible`, hay que cambiar `artifact_name` en el registro (y el código de la API, porque `flexible_emg_mlp` usa `TinySleepNetEMG` y recibe el EMG repetido 100 veces); el script avisa en ese caso. Sin `--cv-run-id`, `--final` conserva el comportamiento anterior (un run aparte que la API no usa).

## 11. Parámetros configurables

| Parámetro | Dónde | Para qué |
|---|---|---|
| `SLEEP_STAGING_API_URL` | `.env` | Dirección de la API que usa el tablero |
| `MODEL_PATH` | `.env` | Ruta del modelo dentro de la imagen de la API |
| `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD` | `.env` | Servidor de MLflow y credenciales de acceso |
| Hiperparámetros de TinySleepNet | [`configs/models/tinysleepnet.yaml`](configs/models/tinysleepnet.yaml) | Semilla, tamaño de secuencia y de lote, tasa de aprendizaje, parada temprana |
| Umbral de baja confianza, estudios de demostración, tamaño máximo de carga | <!-- PENDIENTE (María José) --> `configs/app.yaml` | Comportamiento del tablero |

<!-- PENDIENTE (todos): completar con cada parámetro nuevo -->

## 12. Credenciales de ejemplo

Ninguna credencial se guarda en el repositorio. El archivo [`.env.example`](.env.example) lista las variables necesarias con valores de ejemplo; se copia a `.env`, que Git ignora.

## 13. Estructura del repositorio

```text
.
├── app/
│   ├── api/              # API de inferencia (FastAPI)
│   └── dashboard/        # Tablero (Streamlit) y manuales
├── configs/              # Configuración de modelos y aplicación
├── data/                 # Referencias DVC (los datos no se suben a Git)
├── modelos/              # Pesos de los modelos (no se suben a Git)
├── notebooks/            # Exploración de datos
├── reports/              # Tablas y figuras
├── scripts/              # Un comando por etapa
├── src/                  # Código del pipeline y de los modelos
├── tests/                # Pruebas automáticas
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

## 14. Equipo

| Integrante | Rol |
|---|---|
| María José García Bonilla | Líder, interfaz y documentación |
| Daniel Alejandro Osuna Garzón | Ingeniería de implementación y despliegue |
| Joel Santiago Vargas Lindarte | Datos y modelos clásicos |
| Gabriel Antonio Vizcaya Meneses | Modelos de aprendizaje profundo |

Datos: Sleep-EDF Expanded (Kemp et al., 2000), distribuido por PhysioNet bajo licencia Open Data Commons Attribution v1.0.