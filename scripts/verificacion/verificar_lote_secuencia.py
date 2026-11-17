"""Verifica qué tamaño de lote y qué longitud de secuencia se usaron de verdad en los 8 modelos.

Tres fuentes, de la más cercana al código a la más cercana al resultado:
  1. El código del repositorio: constantes y valores por defecto que usan el entrenamiento y el 08.
  2. Un lote real: arma el cargador de entrenamiento del modelo flexible con una grabación sintética
     (no lee datos) y mira la forma del primer lote.
  3. MLflow: los parámetros registrados en cada run y la seq_length guardada en el JSON del modelo final.
No entrena ni evalúa nada.

Uso, desde la raíz del repositorio:
    python scripts/verificacion/verificar_lote_secuencia.py
    (pide el URI, el usuario y la clave de MLflow si no están en el entorno)
"""
import getpass
import inspect
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

RAIZ = Path.cwd()
sys.path.insert(0, str(RAIZ / "src"))
import data.dataset as ds  # noqa: E402
from sleep_staging.models.tinysleepnet import TinySleepNetConfig  # noqa: E402
from sleep_staging.training import canales_flexibles as cf  # noqa: E402

RUNS = {"flexible_emg_mlp": "95951e2a2cfd4368970636cc6bdea9b5", "flexible": "029cc677c3734b10959735a6e9246cb8",
        "flexible_sin_lstm": "c424e04708c34f04a5bfec810b705982",
        "control_eeg_fpz_cz": "2bd024695ad44d19b175490e5abbdedb", "control_eeg_pz_oz": "30877a344b304c5a8a2b5f2e3163805f",
        "control_eeg_fpz_cz__eeg_pz_oz": "076282f5e5534691b441514b50e90441",
        "control_eeg_fpz_cz__eog_horizontal": "038a654fc695444ab650f4e2aac9056f",
        "control_eeg_fpz_cz__eeg_pz_oz__eog_horizontal__emg_submental": "74b16a9d1a854bb5a5589353fec14f91"}

print("== 1. Código ==")
firma = inspect.signature(cf.build_flexible_dataloader).parameters
print(f"   data/dataset.py              DEFAULT_SEQ_LENGTH = {ds.DEFAULT_SEQ_LENGTH}   DEFAULT_BATCH_SIZE = {ds.DEFAULT_BATCH_SIZE}")
print(f"   build_flexible_dataloader    seq_length por defecto = {firma['seq_length'].default}   batch_size por defecto = {firma['batch_size'].default}")
print(f"   TinySleepNetConfig           seq_length por defecto = {TinySleepNetConfig().seq_length}")
fuente = inspect.getsource(sys.modules["sleep_staging.training.train_flexible"]) if "sleep_staging.training.train_flexible" in sys.modules else None
if fuente is None:
    import sleep_staging.training.train_flexible as tf_mod  # noqa: E402
    fuente = inspect.getsource(tf_mod)
llamadas = [l.strip() for l in fuente.splitlines() if "build_flexible_dataloader(" in l]
print("   train_flexible.py arma sus cargadores así (si no pasa batch_size, usa el valor por defecto):")
for l in llamadas:
    print(f"      {l}")
pasa_batch = any("batch_size" in l for l in llamadas)

print("\n== 2. Un lote real del cargador de entrenamiento ==")
rec = ds.SubjectRecording(**{k: v for k, v in dict(
    subject_code="00", recording_id="sintetica", X=np.zeros((600, len(cf.POSICIONES), 3000), np.float32), y=np.zeros(600, np.int64),
    channel_names=list(cf.POSICIONES)).items() if k in inspect.signature(ds.SubjectRecording).parameters})
combos = cf.combinaciones_validas()[:1] if hasattr(cf, "combinaciones_validas") else None
try:
    cargador = cf.build_flexible_dataloader([rec], combos, shuffle=False) if combos else None
    xb, yb = next(iter(cargador))
    print(f"   x del primer lote: {tuple(xb.shape)}  -> (lote, épocas por secuencia, canales, muestras)")
    print(f"   y del primer lote: {tuple(yb.shape)}")
    lote_real, sec_real = xb.shape[0], xb.shape[1]
except Exception as e:                                                 # noqa: BLE001
    print(f"   (no se pudo armar el lote sintético: {e}); se usan solo las otras fuentes")
    lote_real = sec_real = None

print("\n== 3. MLflow ==")
import mlflow  # noqa: E402
if not os.environ.get("MLFLOW_TRACKING_URI"):
    os.environ["MLFLOW_TRACKING_URI"] = input("MLFLOW_TRACKING_URI: ").strip()
    u = input("MLFLOW_TRACKING_USERNAME (vacío si no hace falta): ").strip()
    if u:
        os.environ["MLFLOW_TRACKING_USERNAME"] = u
        os.environ["MLFLOW_TRACKING_PASSWORD"] = getpass.getpass("MLFLOW_TRACKING_PASSWORD: ")
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
cli = mlflow.tracking.MlflowClient()
tmp = Path(tempfile.mkdtemp(prefix="lote_"))
resumen = {}
for m, rid in RUNS.items():
    params = cli.get_run(rid).data.params
    rel = {k: v for k, v in params.items() if any(t in k.lower() for t in ("seq", "batch", "lote", "secuencia"))}
    try:
        d = Path(mlflow.artifacts.download_artifacts(run_id=rid, artifact_path="modelo_final", dst_path=str(tmp / m)))
        jf = json.load(open(next(d.rglob(f"{m}_final.json")), encoding="utf-8"))
        sec_json = jf.get("seq_length")
    except Exception:                                                  # noqa: BLE001
        sec_json = None
    resumen[m] = (rel, sec_json)
    print(f"   {m:<62} parámetros: {rel or '(ninguno con seq/batch en el nombre)'}; seq_length del JSON final: {sec_json}")

print("\n== Resumen ==")
sec_mlflow = {v[1] for v in resumen.values()}
print(f"   Longitud de secuencia: código {ds.DEFAULT_SEQ_LENGTH}; lote real {sec_real}; JSON final de los 8 modelos {sorted(sec_mlflow, key=str)}")
print(f"   Tamaño de lote:        código {ds.DEFAULT_BATCH_SIZE}; train_flexible {'lo fija explícitamente' if pasa_batch else 'usa el valor por defecto'}; "
      f"lote real {lote_real}")
print("   (si algún parámetro de MLflow arriba muestra otro valor de lote o secuencia, ese manda: es lo que se registró al entrenar)")
