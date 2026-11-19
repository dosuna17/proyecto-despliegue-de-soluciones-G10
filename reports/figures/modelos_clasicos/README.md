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

## Figuras

| Archivo | Contenido | Notebook |
|---|---|---|
| `01_distribucion_clases_recorte.pdf` | Épocas por estadio antes y después del recorte de vigilia (±30 min), con el % de cada estadio sobre cada barra | `01_EDA.ipynb` |
| `02_edad_por_sujeto.pdf` | Edad por número de sujeto y partición entrenamiento/prueba estratificada | `01_EDA.ipynb` |
| `03_ejemplo_recorte.pdf` | Señal de los cuatro canales con el tramo que conserva el recorte: (a) SC4001, sin siestas; (b) SC4331, con siestas | `09_preprocess_base_models_EMG.ipynb` |
| `04_f1_por_combinacion_clasicos.pdf` | F1 macro en prueba de los cuatro modelos clásicos en las 12 combinaciones | `10_train_base_models_multichannel.ipynb` |
| `05_matriz_confusion_xgboost_global.pdf` | Matriz de confusión normalizada por fila del mejor clásico, con las 12 combinaciones juntas | `10_train_base_models_multichannel.ipynb` |
| `05_matriz_confusion_xgboost_4senales.pdf` | Igual, solo con la combinación de las 4 señales | `10_train_base_models_multichannel.ipynb` |
| `06_importancia_caracteristicas.pdf` | Las 15 variables más importantes del mejor clásico | `10_train_base_models_multichannel.ipynb` |
| `07_f1_por_sujeto_xgboost.pdf` | F1 macro del mejor clásico por sujeto de prueba, ordenados por edad: rango y promedio de las 12 combinaciones y valor con las 4 señales | `10_train_base_models_multichannel.ipynb` |
| `07_f1_por_sujeto_xgboost.csv` | Tabla de la figura anterior: sujeto, edad, sexo, noches y F1 macro | `10_train_base_models_multichannel.ipynb` |
| `08_calibracion_umbral_sacadas_eog.pdf` | AUC de REM frente a N1 + N2 y sacadas en N3 según el umbral y el suavizado del detector de sacadas del EOG | `12_eog_features_validation.ipynb` |
