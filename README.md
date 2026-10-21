# Proyecto de Desarrollo de Soluciones - Grupo 5

Clasificación automática de estadios del sueño mediante aprendizaje de máquina
a partir de señales de polisomnografía (Sleep-EDFx).

## Equipo
- María José García Bonilla
- Joel Santiago Vargas Lindarte
- Gabriel Antonio Vizcaya Meneses
- Daniel Alejandro Osuna Garzón

## Estructura del repositorio

```text
.
├── .dvc/                         # Configuración local de DVC
├── .github/
│   └── CODEOWNERS                # Responsables del código
├── app/
│   ├── api/                      # API de inferencia
│   └── dashboard/                # Interfaz de visualización
├── configs/                      # Configuraciones de entrenamiento y aplicación
├── data/
│   ├── raw/                      # Datos de origen (no se suben a Git)
│   ├── interim/                  # Datos intermedios
│   ├── processed/                # Datos preparados para modelado
│   └── sleep-edfx.dvc            # Referencia DVC al conjunto Sleep-EDFx
├── models/                       # Referencias locales; sin pesos en Git
├── notebooks/
│   └── EDA_microproyecto_G5.ipynb
├── reports/
│   └── figures/                  # Figuras y visualizaciones
├── src/
│   ├── preprocessing/            # Limpieza y transformación de señales
│   ├── features/                 # Ingeniería de características
│   ├── models/                   # Entrenamiento y definición de modelos
│   ├── evaluation/               # Métricas y evaluación
│   └── utils/                    # Utilidades compartidas
├── tests/                        # Pruebas automatizadas
├── .dvcignore
├── .gitignore
├── README.md
└── requirements.txt
```

Los datos se gestionan con DVC. Los artefactos locales de MLflow, datos crudos,
credenciales y pesos de modelos se excluyen de Git; los modelos candidatos y de
producción se deben versionar en MLflow Model Registry.

## Arquitectura de datos y experimentación

```text
                    GitHub
                      │
                 código y metadatos
                      │
                      ▼
             Proyecto G5 (repo)
              │             │
        archivos .dvc     código de entrenamiento
              │             │
              ▼             ▼
             DVC          MLflow Tracking
              │             │
              ▼             ├── Backend store: SQLite (EC2)
 S3 / dvc-proyecto/files/  │
                            ▼
                  S3 / dvc-proyecto/mlflow-artifacts/
```

- **GitHub** conserva el código y los metadatos de DVC, pero no datos crudos,
  pesos de modelos ni credenciales.
- **DVC** versiona los conjuntos de datos almacenados en
  `s3://dvc-proyecto/files/`.
- **MLflow** registra los experimentos en un backend SQLite alojado en EC2 y
  almacena artefactos, métricas y modelos en
  `s3://dvc-proyecto/mlflow-artifacts/`.

## Despliegue de la API

```text
Dashboard
   │
   │ HTTP :8000
   ▼
┌──────────────────────┐
│   Docker + FastAPI   │
│                      │
│ GET /health          │
│ POST /predict        │  ← pendiente
└──────────┬───────────┘
           │
           │ MLFLOW_TRACKING_URI
           ▼
┌──────────────────────┐
│   MLflow :5000       │
│      EC2             │
└──────────────────────┘
```

El contenedor expone FastAPI en el puerto `8000`. El endpoint `GET /health`
permite comprobar la disponibilidad del servicio; `POST /predict` se añadirá
para cargar el modelo registrado desde MLflow y realizar inferencias.
### Endpoint de inferencia

`POST /predict` recibe 15 epocas consecutivas del canal `EEG Fpz-Cz`.
Cada epoca debe contener 3.000 muestras (30 segundos a 100 Hz) en microvoltios
(`uV`). La respuesta incluye una prediccion, confianza y probabilidades para
cada epoca. Coloque `tinysleepnet_final.pth` en `models/` o configure
`MODEL_PATH` con la ruta al checkpoint antes de iniciar la API.
## MLflow Tracking Server

El proyecto utiliza un servidor MLflow desplegado en AWS EC2.

### Tracking URI

Todos los experimentos deben utilizar la siguiente variable de entorno:

```bash
export MLFLOW_TRACKING_URI=https://18.119.196.100
```
