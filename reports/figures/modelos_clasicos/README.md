# Figuras: datos y modelos clásicos

Figuras del análisis exploratorio, los modelos clásicos (árbol de decisión, Random Forest,
XGBoost, LightGBM) y la referencia YASA, generadas desde los notebooks de `notebooks/`.
Responsable: Santiago.

## Formato

- **PDF vectorial** (`plt.savefig("....pdf", bbox_inches="tight")`), para que se vean nítidas en el artículo.
- Ancho pensado para una columna del artículo (unos 8.8 cm, `figsize=(3.5, 2.6)`) o para dos columnas (`figsize=(7.2, 3)`).
- Texto de al menos 8 pt y colores que se distingan también impresos en gris.
- **Sin título dentro de la figura**: el título y la explicación van en el pie de figura del artículo.
- Nombre: `NN_descripcion_corta.pdf`, en minúsculas y sin tildes ni espacios. El número indica el orden sugerido.
- En el notebook, una línea que diga qué celda genera cada figura.

## Figuras esperadas

| Archivo | Contenido |
|---|---|
| `01_distribucion_clases_recorte.pdf` | Épocas por estadio antes y después del recorte de vigilia |
| `02_edad_por_sujeto.pdf` | Edad por número de sujeto y partición entrenamiento/prueba estratificada |
| `03_ejemplo_recorte.pdf` | Hipnograma de un registro con la ventana recortada (±30 min) |
| `04_f1_por_combinacion_clasicos.pdf` | F1 macro en prueba de los cuatro modelos clásicos en las 12 combinaciones |
| `05_matriz_confusion_xgboost.pdf` | Matriz de confusión normalizada por fila del mejor clásico |
| `06_importancia_caracteristicas.pdf` | Importancia de las variables del mejor clásico |
| `07_yasa_vs_modelos.pdf` | YASA frente a los modelos en las mismas épocas de prueba |
