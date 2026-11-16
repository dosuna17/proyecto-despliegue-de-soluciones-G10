# Figuras: aprendizaje profundo

Figuras de TinySleepNet (modelo flexible, rama EMG, controles y ablación sin LSTM), de la
evaluación única en prueba y de los umbrales de confianza por estadio.
Responsable: Gabriel.

## Formato

- **PDF vectorial** (`plt.savefig("....pdf", bbox_inches="tight")`), para que se vean nítidas en el artículo.
- Ancho pensado para una columna del artículo (unos 8.8 cm, `figsize=(3.5, 2.6)`) o para dos columnas (`figsize=(7.2, 3)`).
- Texto de al menos 8 pt y colores que se distingan también impresos en gris.
- **Sin título dentro de la figura**: el título y la explicación van en el pie de figura del artículo.
- Nombre: `NN_descripcion_corta.pdf`, en minúsculas y sin tildes ni espacios. El número indica el orden sugerido.
- Indicar en el script o notebook qué parte genera cada figura.

## Figuras esperadas

| Archivo | Contenido |
|---|---|
| `01_arquitectura_flexible.pdf` | Esquema del modelo flexible: entrada de señales + marcas de presencia y rama EMG |
| `02_curvas_entrenamiento_cv.pdf` | Pérdida y F1 de entrenamiento y validación por pliegue |
| `03_f1_por_combinacion_test.pdf` | F1 macro en prueba con IC 95 % por combinación: campeón, flexible, controles y sin LSTM |
| `04_matriz_confusion_campeon.pdf` | Matriz de confusión normalizada por fila del campeón (mejor combinación y Fpz-Cz) |
| `05_f1_por_persona.pdf` | Distribución del F1 macro por persona en prueba |
| `06_umbrales_por_clase.pdf` | Cobertura y precisión de lo aceptado por estadio frente a la meta (SleepFM − 5) |
