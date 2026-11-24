# Notebooks

Los notebooks se ejecutan en orden numérico desde la carpeta `notebooks/` (todos
usan `Path.cwd().parent` como raíz del proyecto) con el entorno de
`requirements.txt`.

## Resumen de cada notebook

| Notebook | Qué hace |
|---|---|
| `01_EDA.ipynb` | Análisis exploratorio de Sleep-EDF (Sleep Cassette): canales, frecuencias de muestreo, duración de las grabaciones y distribución de las etapas de sueño. |
| `02_data_loader_validation.ipynb` | Valida `src/data/loader.py` y `src/data/segmentation.py`: carga de PSG + hipnograma y segmentación en épocas de 30 s. |
| `03_split_subjects_validation.ipynb` | Analiza `SC-subjects.xls` y genera `data/SC-subjects-split.xlsx` con el split train/test **por sujeto** (las dos noches de un sujeto quedan del mismo lado). |
| `04_data_preprocessing.ipynb` | Ejemplo de cómo convertir las épocas crudas en features espectrales tabulares. |
| `05_model_xgboost.ipynb` | Primer XGBoost sobre `train.csv`/`test.csv`, con búsqueda de hiperparámetros validada con `StratifiedGroupKFold` por `subject_code` y métrica `f1_macro`. |
| `06_best_model_xgboost.ipynb` | Reentrena el mejor XGBoost del 05, calcula métricas y lo registra en MLflow. |
| `07_preprocess_base_models_multichannel.ipynb` | Genera `train_multicanal.csv` y `test_multicanal.csv` con EEG Fpz-Cz, EEG Pz-Oz y EOG horizontal (6 combinaciones de canales, 35 features). |
| `08_train_base_models_multichannel.ipynb` | Entrena XGBoost, LightGBM, Random Forest y Árbol de Decisión sobre los CSV del 07, compara por combinación de canales y registra en MLflow. |
| `09_preprocess_base_models_EMG.ipynb` | Igual que el 07, pero agrega EMG submental: genera `train_multicanal_emg.csv` y `test_multicanal_emg.csv` (12 combinaciones, 40 features). |
| `10_train_base_models_multichannel.ipynb` | Igual que el 08, pero con los CSV del 09. Todos los artefactos y modelos de MLflow llevan el sufijo `emg`, para no sobrescribir los del 08. |
| `11_yasa_validation.ipynb` | Valida el clasificador pre-entrenado de [YASA](https://yasa-sleep.org/) (`yasa.SleepStaging`) sobre el mismo set de test del 10 (44 985 épocas) y reporta sus métricas. 
| `12_eog_features_validation.ipynb` | Valida las features de tiempo del EOG (`p2p_robust`, `hjorth_mobility`, `saccade_count`, `kurtosis`) solo con sujetos de train: calibra el umbral de sacadas (750 µV/s con 100 ms de suavizado, por AUC de REM frente a N1 + N2), compara las features por clase y revisa su redundancia con las espectrales.

### Procesamiento común en el 07 y el 09

- Se recortan los periodos largos de vigilia al inicio y al final de cada
  grabación, dejando 60 épocas (30 min) de margen.
- Cada época se repite una vez por combinación de canales. Las features de un
  canal que no está en la combinación quedan en `NaN`.
- Las columnas `channel_combo`, `channel_count` y `has_*` se guardan solo para
  análisis; los notebooks de entrenamiento no las usan como features.

Tamaños esperados:

| CSV | Filas |
|---|---|
| `test_multicanal.csv` (07) | 269 910 (44 985 épocas × 6 combinaciones) |
| `test_multicanal_emg.csv` (09) | 539 820 (44 985 épocas × 12 combinaciones) |

## Reproducibilidad con DVC (notebooks 07 y 09)

DVC versiona tanto los datos de entrada como los CSV generados, que se guardan
en el remoto `s3remote` (`s3://proyecto-dvc/sleep-edfx`). Git guarda solo los
archivos `.dvc`, que contienen el hash de cada versión. Para reproducir un
dataset hay que usar el mismo commit de código y los mismos datos de entrada.

### 1. Descargar los datos de entrada

Necesitan credenciales de AWS con acceso al bucket, configuradas con
`aws configure` o con variables de entorno.

```bash
dvc pull data/sleep-edfx.dvc data/SC-subjects-split.xlsx.dvc
```

Con eso quedan en su lugar:

- `data/sleep-edfx/sleep-cassette/`: los EDF de PSG y los hipnogramas.
- `data/SC-subjects-split.xlsx`: el split por sujeto que genera el notebook 03.

### 2. Ejecutar el notebook

Pueden correrlo desde Jupyter/VS Code o desde la terminal:

```bash
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 07_preprocess_base_models_multichannel.ipynb
jupyter nbconvert --to notebook --execute --inplace 09_preprocess_base_models_EMG.ipynb
```

### 3. Versionar las salidas

```bash
# Notebook 07
dvc add data/processed/train_multicanal.csv data/processed/test_multicanal.csv

# Notebook 09
dvc add data/processed/train_multicanal_emg.csv data/processed/test_multicanal_emg.csv

git add data/processed/*.dvc data/processed/.gitignore
git commit -m "Actualiza datasets multicanal"
dvc push
```

Hagan `dvc push` antes del `git push`. Si lo hacen al revés, el `.dvc` queda
apuntando a un hash que no existe en S3.

### 4. Usar un dataset ya generado

Si solo van a entrenar (notebooks 08 y 10), no necesitan volver a ejecutar el
preprocesamiento. Basta con traer la versión que corresponde al commit actual:

```bash
dvc pull data/processed/train_multicanal.csv.dvc data/processed/test_multicanal.csv.dvc
dvc pull data/processed/train_multicanal_emg.csv.dvc data/processed/test_multicanal_emg.csv.dvc
```

Los `.dvc` de los CSV `_emg` existirán solo después de hacer el paso 3 con el
notebook 09.

## Variables obtenidas para cada canal

Son 40 features por época de 30 s: 14 por cada canal EEG, 7 del EOG y 5 del
EMG. Los nombres tienen la forma `<canal>__<feature>`, por ejemplo
`EEG_Fpz-Cz__delta_power_rel`. Las señales están en µV.

### Canales EEG: `EEG_Fpz-Cz__*` y `EEG_Pz-Oz__*` (14 cada uno, 28 en total)

Se calculan sobre la densidad espectral de potencia (PSD), estimada con Welch
usando ventanas de 4 s. La banda total es 0.5–30 Hz.

| Feature | Definición |
|---|---|
| `delta_power_abs` | Potencia absoluta en delta (0.5–4 Hz), en µV². Es la señal típica del sueño profundo (N3). |
| `theta_power_abs` | Potencia absoluta en theta (4–8 Hz). Se asocia a N1 y REM. |
| `alpha_power_abs` | Potencia absoluta en alpha (8–12 Hz). Se asocia a la vigilia relajada con ojos cerrados. |
| `sigma_power_abs` | Potencia absoluta en sigma (12–16 Hz). Es la banda de los husos de sueño, característicos de N2. |
| `beta_power_abs` | Potencia absoluta en beta (16–30 Hz). Se asocia a la vigilia activa. |
| `*_power_rel` (5 features: delta, theta, alpha, sigma, beta) | Potencia de la banda dividida por la potencia total (0.5–30 Hz). No depende de la amplitud absoluta de cada registro. |
| `delta_alpha_ratio` | Potencia delta / potencia alpha. Aumenta a medida que el sueño es más profundo. |
| `theta_alpha_ratio` | Potencia theta / potencia alpha. Ayuda a reconocer la transición de vigilia a N1. |
| `sef95` | Frecuencia de borde espectral: la frecuencia bajo la cual cae el 95 % de la potencia. Disminuye en el sueño profundo. |
| `spectral_entropy` | Entropía de Shannon normalizada (0–1) del espectro. Es baja si la potencia está concentrada en pocas frecuencias y alta si está repartida. |

### Canal EOG horizontal: `EOG_horizontal__*` (7)

| Feature | Definición |
|---|---|
| `delta_power_abs` | Potencia del EOG en 0.5–4 Hz. Capta movimientos oculares lentos y parpadeos. |
| `delta_power_rel` | La misma potencia, relativa a la potencia total del EOG. |
| `beta_power_abs` | Potencia del EOG en 16–30 Hz. Refleja la actividad muscular y los artefactos de la vigilia. |
| `p2p_robust` | Amplitud pico a pico robusta: percentil 99 menos percentil 1, en µV. Es alta cuando hay movimientos oculares amplios. |
| `hjorth_mobility` | Movilidad de Hjorth, √(Var(x′)/Var(x)). Es una estimación de la frecuencia media de la señal. |
| `saccade_count` | Número de movimientos oculares rápidos detectados en la época: picos de velocidad (señal suavizada 100 ms) por encima de 750 µV/s, separados al menos 0.3 s. Ayuda a reconocer REM y vigilia. |
| `kurtosis` | Curtosis de Fisher de la señal, que es 0 para una señal gaussiana. Es alta cuando hay eventos aislados de gran amplitud, como movimientos rápidos o parpadeos. |

### Canal EMG submental: `EMG_submental__*` (5)

En Sleep Cassette, este canal ya viene como envolvente muestreada a 1 Hz: la
señal se filtró sobre 16 Hz, se rectificó y se suavizó, y se expresa en µV RMS.
Por eso las features se calculan sobre las 30 muestras originales de cada
época, y no sobre un espectro.

| Feature | Definición |
|---|---|
| `mean` | Tono muscular medio de la época, en µV. Es mínimo en REM (atonía). |
| `std` | Variabilidad del tono dentro de la época. |
| `p10` | Percentil 10: el nivel basal del tono muscular. |
| `p90` | Percentil 90: los picos de activación, por ejemplo por movimientos. |
| `slope` | Pendiente de una recta ajustada al tono a lo largo de la época, en µV/s. Indica si el tono sube o baja, como ocurre en las transiciones de etapa. |
