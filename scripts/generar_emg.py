"""Genera el EMG submental (1 Hz, 30 valores por epoca) de las 153 grabaciones, ya recortado.

Mismo procedimiento que los datos de 3 canales: epocas de 30 s, 5 clases, recorte de vigilia
(buffer de 60 epocas). Comprueba que las etiquetas de cada grabacion coinciden con las de
3 canales (sin y con recorte). Lee de data/ del repo; escribe en data/interim/epochs_emg_recortado (o en --salida).
El resultado es lo que lee emg_datos.py (variable SLEEP_EMG_DIR).

Uso (desde cualquier carpeta; encuentra solo la raiz del repo):
    python scripts/generar_emg.py
Es reanudable: si interrumpes, vuelve a correrlo y salta las grabaciones ya hechas.
"""
import argparse, os, sys, time
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
REPO_RAIZ = Path(__file__).resolve().parent.parent
p.add_argument("--repo", default=str(REPO_RAIZ), help="Raiz del repo (por defecto, la de este script).")
p.add_argument("--salida", default=None,
               help="Carpeta de salida (por defecto, <repo>/data/interim/epochs_emg_recortado).")
args = p.parse_args()
repo = Path(args.repo)
salida = Path(args.salida) if args.salida else repo / "data" / "interim" / "epochs_emg_recortado"
salida.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(repo / "src" / "data"))      # solo para importar
from loader import find_subject_pairs, load_hypnogram, load_psg                     # noqa: E402
from segmentation import extract_epochs                                              # noqa: E402
from aplicar_recorte_vigilia import trim_wake_edges_arrays                           # noqa: E402

pares = find_subject_pairs(str(repo / "data" / "sleep-edfx" / "sleep-cassette"))
print(f"{len(pares)} grabaciones (esperadas 153). Salida: {salida}")
interim = repo / "data" / "interim"
ok, saltadas, malas, total, clases = 0, 0, [], 0, {}
t0 = time.time()
for i, (psg, hip) in enumerate(pares, 1):
    rid = Path(psg).name.split("-")[0][:6]
    destino = salida / f"{rid}_epochs.npz"
    if destino.exists():
        saltadas += 1
        continue
    try:
        raw = load_psg(psg, canales=["EMG submental"])
        ann = load_hypnogram(hip)
        raw.set_annotations(ann, emit_warning=False)
        ds = extract_epochs(raw, ann, recording_id=rid, channel_picks=["EMG submental"])
        Xr, yr = trim_wake_edges_arrays(ds.X, ds.y)
        y3 = np.load(interim / "epochs_por_sujeto_3canales" / f"{rid}_epochs.npz", allow_pickle=True)["y"]
        y3r = np.load(interim / "epochs_por_sujeto_3canales_recortado" / f"{rid}_epochs.npz", allow_pickle=True)["y"]
        if not (np.array_equal(y3, ds.y) and np.array_equal(y3r, yr)):
            malas.append(rid)
            print(f"[{i}/{len(pares)}] DIFERENTE {rid}: no se guarda")
            continue
        n = len(yr)
        np.savez_compressed(
            destino, X=Xr, y=yr, channel_names=np.array(ds.channel_names), sfreq=ds.sfreq,
            signal_unit=ds.signal_unit, recording_id=rid, subject_code=ds.subject_code,
            night=ds.night, recording_ids=np.full(n, rid, dtype=object),
            subject_codes=np.full(n, ds.subject_code, dtype=object),
        )
        ok += 1
        print(f"[{i}/{len(pares)}] OK {rid}: {n} epocas ({time.time() - t0:.0f} s)")
    except Exception as e:
        malas.append(rid)
        print(f"[{i}/{len(pares)}] FALLO {rid}: {e}")

archivos = sorted(salida.glob("*_epochs.npz"))
for a in archivos:
    y = np.load(a, allow_pickle=True)["y"]
    total += len(y)
    for c, k in zip(*np.unique(y, return_counts=True)):
        clases[str(c)] = clases.get(str(c), 0) + int(k)
print("\n" + "=" * 60)
print(f"Nuevas: {ok} | ya existian: {saltadas} | con problema: {len(malas)} {malas}")
print(f"Archivos en la carpeta: {len(archivos)} | epocas totales: {total:,} (esperado 195,479)")
print("Epocas por clase:", clases)
print("=" * 60)
