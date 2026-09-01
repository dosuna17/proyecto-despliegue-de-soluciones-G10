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

## MLflow Tracking Server

El proyecto utiliza un servidor MLflow desplegado en AWS EC2.

### Tracking URI

Todos los experimentos deben utilizar la siguiente variable de entorno:

```bash
export MLFLOW_TRACKING_URI=http://32.194.151.154:5000
```
