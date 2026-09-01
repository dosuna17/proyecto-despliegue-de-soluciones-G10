# Proyecto de Desarrollo de Soluciones - Grupo 5

Clasificación automática de estadios del sueño mediante aprendizaje de máquina
a partir de señales de polisomnografía (Sleep-EDFx).

## Equipo
- María José García Bonilla
- Joel Santiago Vargas Lindarte
- Gabriel Antonio Vizcaya Meneses
- Daniel Alejandro Osuna Garzón

## Estructura
- `data/` — datos versionados con DVC
- `.dvc/` — configuración de DVC

## MLflow Tracking Server

El proyecto utiliza un servidor MLflow desplegado en AWS EC2.

### Tracking URI

Todos los experimentos deben utilizar la siguiente variable de entorno:

```bash
export MLFLOW_TRACKING_URI=http://32.194.151.154:5000

```text
