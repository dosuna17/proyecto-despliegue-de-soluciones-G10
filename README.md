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

**Problema.** La estadificación del sueño se hace a mano, época por época: una noche tiene entre 800 y 1200 épocas de 30 s. Es lenta y variable: dos expertos coinciden en cerca del 82.6 % de las épocas, y en el estadio N1 apenas en el 63 % (Rosenberg y Van Hout, 2013).

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
| Aplicación (tablero) | http://18.223.227.20:8501 (IP elástica en AWS EC2; la instancia se enciende en los horarios de evaluación) |
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
| TinySleepNet flexible con EMG (campeón) | Fpz-Cz + EOG + EMG | 0.826 | 0.784 | 0.762 | 1,558,165 parámetros | <!-- PENDIENTE (Daniel) --> |
| TinySleepNet flexible con EMG (campeón) | Fpz-Cz | 0.799 | 0.749 | 0.726 | | |
| TinySleepNet (versión base) | Fpz-Cz + EOG | 0.811 | 0.760 | 0.742 | <!-- PENDIENTE (Daniel) --> | <!-- PENDIENTE (Daniel) --> |
| TinySleepNet (versión base) | Fpz-Cz | 0.769 | 0.723 | 0.688 | | |
| XGBoost | <!-- PENDIENTE (Santiago): canales --> | 0.728 | 0.671 | 0.633 | | |
| XGBoost | Fpz-Cz | 0.670 | 0.610 | 0.560 | | |

El campeón (un solo modelo para las 12 combinaciones de señales) tiene 1,558,165 parámetros; la versión base de TinySleepNet, 1,516,421. Las métricas del campeón son de las 16 personas de prueba (44,985 épocas); las de la versión base (v0.1.0) se obtuvieron con otra configuración de evaluación, así que no se comparan directamente. Hardware de inferencia: CPU, sin GPU. <!-- PENDIENTE (Daniel): memoria en reposo y al predecir, medida en la instancia EC2 -->

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

La API no recibe archivos. Su fuente del checkpoint se configura con `MODEL_USE_MLFLOW`: en `true`, resuelve el run publicado en MLflow y descarga el artefacto configurado en `MLFLOW_ARTIFACT_PATH`; en `false`, usa el checkpoint descargado indicado por `MODEL_PATH` (por defecto, `/models/flexible_final.pth` en Docker) y no consulta MLflow. La resolución desde MLflow usa el experimento y las etiquetas configuradas en `configs/models/model_registry.yaml`, sin depender del nombre del run. Cuatro comandos para registrar el modelo, siempre con `MLFLOW_TRACKING_URI` definida en el entorno:

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

**Estado actual.** El campeón es `flexible_emg_mlp` (run `95951e2a2cfd4368970636cc6bdea9b5`, `champion=true` desde el 15 de noviembre); el run `flexible` ya no tiene la etiqueta. `configs/models/model_registry.yaml` busca ese run por la etiqueta y la API carga `TinySleepNetEMG`.

### Reproducir los resultados del modelo flexible (artículo)

Comandos exactos de las corridas que se reportan en el artículo. Todos se ejecutan desde la raíz del repositorio, con semilla 42 (`configs/models/tinysleepnet.yaml`) y con estas dos variables de entorno:

```bash
export SLEEP_NPZ_DIR=data/interim/epochs_por_sujeto_3canales_recortado   # EEG Fpz-Cz, EEG Pz-Oz y EOG (100 Hz)
export SLEEP_EMG_DIR=data/interim/epochs_emg_recortado                   # EMG submental (1 Hz, 30 valores por época)
```

**1. Datos** (153 grabaciones; 195,479 épocas después del recorte de vigilia):

```bash
dvc pull                                                                  # o python scripts/01_download_data.py
python scripts/generar_3canales.py
python src/data/aplicar_recorte_vigilia.py data/interim/epochs_por_sujeto_3canales
python scripts/generar_emg.py                                             # comprueba que las etiquetas coincidan con las de 3 canales
```

**2. Validación cruzada** (5 pliegues por persona sobre las 62 personas de entrenamiento; ventana de 500 épocas continuas por noche). Guarda los JSON por pliegue y las probabilidades de validación (`<salida>/probs_validacion/`), que después fijan los umbrales:

| Run | Comando |
|---|---|
| `flexible` | `python scripts/06_train_flexible.py --max-epocas 30 --salida res_flexible` |
| `flexible_emg_mlp` (campeón) | `python scripts/06_train_flexible.py --emg mlp --max-epocas 30 --salida res_emg` |
| `flexible_sin_lstm` (ablación) | `python scripts/06_train_flexible.py --sin-lstm --max-epocas 30 --salida res_sin_lstm` |
| `control_<combinación>` (4 controles sin EMG) | `python scripts/06_train_flexible.py --control <combinación> --max-epocas 240 --paciencia 60 --salida res_controles`, con `<combinación>` = `eeg_fpz_cz`, `eeg_pz_oz`, `eeg_fpz_cz__eeg_pz_oz`, `eeg_fpz_cz__eog_horizontal` |
| `control_…__emg_submental` (control de 4 señales) | `python scripts/06_train_flexible.py --emg mlp --control eeg_fpz_cz__eeg_pz_oz__eog_horizontal__emg_submental --max-epocas 240 --paciencia 60 --salida res_controles` |

Los flexibles usan paciencia 5 (la del yaml) y se detienen por paciencia antes de 30 épocas; los controles tienen más presupuesto porque cada uno ve una sola combinación. Cada pliegue se puede correr por separado con `--pliegues k`.

**3. Modelo final** (noche completa, sin validación; las épocas son la media de las mejores épocas de la validación cruzada): el mismo comando de cada run, sin `--max-epocas` ni `--paciencia`, con `--final --noche-completa`. Por ejemplo, el campeón:

```bash
python scripts/06_train_flexible.py --final --noche-completa --emg mlp --salida res_emg
```

**4. Prueba (una sola vez).** Primero los umbrales, solo con validación; luego el ensayo y el test:

```bash
python scripts/08_evaluar_test.py --modelos modelos_finales --solo-umbrales --probs-validacion res_flexible res_emg res_sin_lstm res_controles
python scripts/08_evaluar_test.py --modelos modelos_finales --modo prueba --probs-validacion res_flexible res_emg res_sin_lstm res_controles --delta 0.05
python scripts/08_evaluar_test.py --modelos modelos_finales --modo test --confirmo-test-unico --probs-validacion res_flexible res_emg res_sin_lstm res_controles --delta 0.05 --salida resultados_test --figuras resultados_test/figuras
```

El test de este proyecto ya se ejecutó (candado `TEST_YA_EJECUTADO.json` en estado completo) y no se repite: los resultados se analizan desde las predicciones guardadas (`resultados_test/probs_<modelo>__<combinación>.npz`).

**5. MLflow** (experimento `sleep-staging-advanced`): para cada run, los pasos 1 a 3 de la subsección siguiente; el campeón se marca con `--campeon`. Runs del artículo:

| Run | run_id |
|---|---|
| `flexible_emg_mlp` (campeón, `champion=true`) | `95951e2a2cfd4368970636cc6bdea9b5` |
| `flexible` | `029cc677c3734b10959735a6e9246cb8` |
| `flexible_sin_lstm` | `c424e04708c34f04a5bfec810b705982` |
| `control_eeg_fpz_cz` | `2bd024695ad44d19b175490e5abbdedb` |
| `control_eeg_pz_oz` | `30877a344b304c5a8a2b5f2e3163805f` |
| `control_eeg_fpz_cz__eeg_pz_oz` | `076282f5e5534691b441514b50e90441` |
| `control_eeg_fpz_cz__eog_horizontal` | `038a654fc695444ab650f4e2aac9056f` |
| `control_eeg_fpz_cz__eeg_pz_oz__eog_horizontal__emg_submental` | `74b16a9d1a854bb5a5589353fec14f91` |

Pesos del campeón: `modelo_final/flexible_emg_mlp_final.pth` (1,558,165 parámetros; md5 `a3fb82ce27e33159d30b38ed3df3850c`).

### Verificar los números del artículo

Cada script recalcula una parte del artículo desde los artefactos guardados (MLflow, `resultados_test/`, `probs_validacion/`) e imprime `[OK ]` o `[NO ]` por afirmación. Ninguno reentrena ni vuelve a evaluar el test.

| Sección | Comando (desde la raíz) |
|---|---|
| Arquitectura y pesos del campeón | `python scripts/verificacion/verificar_arquitectura.py` y `verificar_pesos.py` |
| Entrenamiento con todas las combinaciones y comparación con los dedicados | `python scripts/verificacion/verificar_entrenamiento.py` y `verificar_flexible_vs_dedicados.py --probs resultados_test` |
| Hiperparámetros | `python scripts/verificacion/verificar_hiperparametros.py` y `verificar_tabla_hiperparametros.py` |
| Validación cruzada | `python scripts/verificacion/verificar_validacion_cruzada.py` |
| Prueba por combinación, comparación con los dedicados y ablación sin LSTM | `python scripts/verificacion/verificar_prueba_y_ablacion.py --probs resultados_test --mlflow` |
| Matrices de confusión del campeón | `python scripts/verificacion/figura_matrices_confusion.py --probs resultados_test --salida reports/figures` |
| Curvas de entrenamiento (sobreajuste) y variabilidad entre personas | `python scripts/verificacion/figura_sobreajuste.py --probs resultados_test --salida reports/figures` |
| Umbrales de confianza por estadio | `python scripts/verificacion/figura_umbrales.py --probs resultados_test --val-probs res_emg/probs_validacion --salida reports/figures` |
| EMG y clase M | `python scripts/verificacion/verificar_emg_y_clase_m.py --probs resultados_test --emg-dir data/interim/epochs_emg_recortado` |
| Descripción de la arquitectura frente al código y los pesos del campeón | `python scripts/verificacion/verificar_parrafo_arquitectura.py` |
| Lote y longitud de secuencia de los 8 modelos | `python scripts/verificacion/verificar_lote_secuencia.py` |
| Cómputo (parámetros y tiempos de entrenamiento) | `python scripts/verificacion/verificar_computo.py` |
| Tablas de resultados: prueba por combinación, validación cruzada, comparaciones, confianza, F1 por persona y entrenamiento | `python scripts/verificacion/tabla_test_por_combinacion.py --probs resultados_test` (igual `tabla_comparaciones.py`, `tabla_confianza.py` y `tabla_por_persona.py`); `tabla_validacion_cruzada.py` y `tabla_entrenamiento.py` leen MLflow |
| Figuras de `reports/figures/aprendizaje_profundo/` (01 a 06) | `python scripts/verificacion/generar_figuras_articulo.py --probs resultados_test --val-probs res_emg/probs_validacion` |

Los que leen MLflow piden la URI, el usuario y la clave si no están en el entorno.

## 11. Parámetros configurables

| Parámetro | Dónde | Para qué |
|---|---|---|
| `SLEEP_STAGING_API_URL` | `.env` | Dirección de la API que usa el tablero |
| `MODEL_USE_MLFLOW` | `.env` | `true` busca y descarga el modelo desde MLflow; `false` carga el checkpoint local de `MODEL_PATH` |
| `MODEL_PATH` | `.env` | Ruta del checkpoint local (requerido con `MODEL_USE_MLFLOW=false`) |
| `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD` | `.env` | Servidor de MLflow y credenciales de acceso |
| Hiperparámetros de TinySleepNet | [`configs/models/tinysleepnet.yaml`](configs/models/tinysleepnet.yaml) | Semilla, tamaño de secuencia y de lote, tasa de aprendizaje, parada temprana |
| Tablero: URL de la API, solicitudes en paralelo, recorte de la noche, umbrales de confianza por estadio, estudios de demostración, tamaño máximo de carga | [`configs/app.yaml`](configs/app.yaml) (cada valor se puede cambiar con una variable `TABLERO_*`, ver la guía de despliegue del tablero) | Comportamiento del tablero sin tocar el código |

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
