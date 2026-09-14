"""
Genera un archivo .npz por sujeto/noche, procesando TODO el dataset
Sleep-EDF Sleep Cassette (153 grabaciones esperadas).

Uso:
    python generar_npz.py

Antes de correr, ajusta DATA_DIR y OUT_DIR si tu estructura de carpetas
es distinta (verifica primero con el "chequeo rápido" de más abajo).

Diseño clave: REANUDABLE. Si el proceso se interrumpe (ej. se cae la
sesión de AWS Academy Lab a mitad de camino), simplemente vuelve a
correr el script -- salta automáticamente los sujetos que ya tienen
su .npz generado y solo procesa los que faltan. No usa --force para
evitar reprocesar por accidente y perder tiempo.
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import numpy as np

# Ajusta esto si tu loader.py/segmentation.py no están en la raíz del repo
sys.path.append(str(Path(__file__).resolve().parent))

from loader import find_subject_pairs, load_subject          # noqa: E402
from segmentation import extract_epochs, class_distribution   # noqa: E402

# ---------------------------------------------------------------------
# CONFIGURACIÓN -- ajusta rutas si hace falta
# ---------------------------------------------------------------------
DATA_DIR = Path("data/sleep-edfx/sleep-cassette")
# Entrega 3 -- Ruta B (TinySleepNet EEG+EOG): carpeta y canales distintos a la
# corrida EEG-solo original, para no chocar con esos .npz ya generados (el
# script es "reanudable" y saltaria todo si el OUT_DIR ya tiene archivos).
OUT_DIR = Path("data/interim/epochs_por_sujeto_eeg_eog")
CHANNEL_PICKS = ["EEG Fpz-Cz", "EOG horizontal"]


def save_epoch_dataset(dataset, out_path: Path) -> None:
    """Guarda un EpochDataset como .npz comprimido con todos sus campos."""
    np.savez_compressed(
        out_path,
        X=dataset.X,
        y=dataset.y,
        recording_ids=dataset.recording_ids,
        subject_codes=dataset.subject_codes,
        channel_names=np.array(dataset.channel_names),
        sfreq=dataset.sfreq,
        signal_unit=dataset.signal_unit,
        dropped_epochs=dataset.dropped_epochs,
        recording_id=dataset.recording_id,
        subject_code=dataset.subject_code,
        night=dataset.night,
    )


def main():
    if not DATA_DIR.exists():
        print(f"ERROR: no existe {DATA_DIR.resolve()}")
        print("Verifica que ya corriste 'dvc pull' y ajusta DATA_DIR si tu estructura es distinta.")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pairs = find_subject_pairs(str(DATA_DIR))
    print(f"Se encontraron {len(pairs)} pares PSG/Hypnogram en {DATA_DIR}")

    if len(pairs) == 0:
        print("ERROR: 0 pares encontrados. Revisa la ruta DATA_DIR (¿dvc pull completo?).")
        sys.exit(1)
    if len(pairs) != 153:
        print(f"AVISO: se esperaban 153 grabaciones (Sleep Cassette), se encontraron {len(pairs)}. "
              f"Puede ser normal si dvc pull aún no terminó, o si hay archivos sueltos.")

    total_class_counts = {"W": 0, "N1": 0, "N2": 0, "N3": 0, "REM": 0}
    ok, skipped, failed = 0, 0, []
    t0 = time.time()

    for i, (psg_path, hypno_path) in enumerate(pairs, start=1):
        recording_id = Path(psg_path).name.split("-")[0][:6]
        out_path = OUT_DIR / f"{recording_id}_epochs.npz"

        if out_path.exists():
            # Entrega 3 -- guarda contra reusar por accidente .npz generados
            # con OTRA combinacion de canales si alguien cambia CHANNEL_PICKS
            # pero se olvida de cambiar tambien OUT_DIR: sin este chequeo, el
            # script "reanudable" saltaria estos archivos en silencio.
            existing_channels = list(np.load(out_path, allow_pickle=True)["channel_names"])
            if existing_channels != CHANNEL_PICKS:
                raise RuntimeError(
                    f"{out_path} ya existe pero fue generado con canales "
                    f"{existing_channels}, no con CHANNEL_PICKS actual "
                    f"{CHANNEL_PICKS}. Cambia OUT_DIR a una carpeta nueva "
                    f"antes de correr con esta combinacion de canales."
                )
            skipped += 1
            continue  # ya procesado en una corrida anterior -> reanudable

        try:
            subject = load_subject(psg_path, hypno_path, subject_id=recording_id)
            dataset = extract_epochs(
                subject.raw,
                subject.annotations,
                recording_id=recording_id,
                channel_picks=CHANNEL_PICKS,
            )
            save_epoch_dataset(dataset, out_path)

            for cls, count in class_distribution(dataset).items():
                total_class_counts[cls] += count

            ok += 1
            elapsed = time.time() - t0
            print(f"[{i}/{len(pairs)}] OK  {recording_id}  "
                  f"({dataset.X.shape[0]} épocas, {dataset.dropped_epochs} descartadas)  "
                  f"[{elapsed:.0f}s transcurridos]")

        except Exception as e:
            failed.append((recording_id, str(e)))
            print(f"[{i}/{len(pairs)}] FALLÓ {recording_id}: {e}")
            traceback.print_exc(limit=1)
            continue  # sigue con el siguiente sujeto, no aborta todo el proceso

    print("\n" + "=" * 60)
    print(f"Procesados en esta corrida: {ok}")
    print(f"Ya existían (saltados):     {skipped}")
    print(f"Fallaron:                   {len(failed)}")
    print(f"Total .npz en {OUT_DIR}:    {len(list(OUT_DIR.glob('*.npz')))}")
    print(f"Distribución de clases (solo lo procesado en esta corrida): {total_class_counts}")
    if failed:
        print("\nSujetos que fallaron (revisar manualmente):")
        for rid, err in failed:
            print(f"  - {rid}: {err}")
    print("=" * 60)


if __name__ == "__main__":
    main()
