"""
Módulo de I/O para el dataset Sleep-EDF (Sleep Cassette).

Responsabilidad única: leer archivos .edf desde disco y devolver
objetos en memoria (señal cruda + anotaciones crudas). No contiene
lógica de ventaneo, filtrado de etiquetas ni construcción de
features/labels — eso vive en segmentation.py.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import mne


@dataclass
class Subject:
    """Contenedor de la señal cruda + anotaciones crudas de un sujeto/noche."""
    subject_id: str
    raw: mne.io.BaseRaw
    annotations: mne.Annotations
    psg_path: str
    hypnogram_path: str


def load_psg(path: str, verbose: bool = False) -> mne.io.BaseRaw:
    """
    Carga un archivo *-PSG.edf como un objeto mne.io.Raw (señal continua).

    No se hace preload aquí (preload=False) para no cargar en RAM
    hasta que segmentation.py realmente necesite los datos.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Archivo PSG no encontrado: {path}")
    return mne.io.read_raw_edf(path, preload=False, verbose=verbose)


def load_hypnogram(path: str) -> mne.Annotations:
    """
    Carga un archivo *-Hypnogram.edf como mne.Annotations.

    Cada anotación es un intervalo (onset, duration, description),
    p. ej. ("Sleep stage W", onset=0.0, duration=30630.0).
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Archivo Hypnogram no encontrado: {path}")
    annotations = mne.read_annotations(path)
    if len(annotations) == 0:
        raise ValueError(f"El archivo de hypnograma no contiene anotaciones: {path}")
    return annotations


def load_subject(
    psg_path: str,
    hypnogram_path: str,
    subject_id: Optional[str] = None,
    verbose: bool = False,
) -> Subject:
    """
    Carga el par PSG + Hypnogram de un sujeto y hace validaciones básicas
    de consistencia (que el hypnograma no exceda la duración del PSG).

    No filtra etiquetas ni corta en épocas — únicamente adjunta las
    anotaciones crudas al objeto Raw para que segmentation.py las use.
    """
    raw = load_psg(psg_path, verbose=verbose)
    annotations = load_hypnogram(hypnogram_path)

    raw.set_annotations(annotations, emit_warning=False)

    psg_duration = raw.n_times / raw.info["sfreq"]
    hypno_end = max(o + d for o, d in zip(annotations.onset, annotations.duration))
    if hypno_end - psg_duration > 30:
        # Es común en Sleep-EDF que la última anotación ('Sleep stage ?')
        # rellene el hypnograma hasta las 24h aunque el PSG real sea más
        # corto. No es un error de alineación -> solo se advierte; la
        # grilla de épocas en segmentation.py nunca excede la señal real.
        print(
            f"[loader] Aviso: el hypnograma termina en {hypno_end:.1f}s pero el PSG "
            f"dura {psg_duration:.1f}s. Probablemente es la cola '?' de relleno; "
            f"se ignorará al extraer épocas ({psg_path})."
        )

    if subject_id is None:
        subject_id = os.path.basename(psg_path).split("-")[0]

    return Subject(
        subject_id=subject_id,
        raw=raw,
        annotations=annotations,
        psg_path=psg_path,
        hypnogram_path=hypnogram_path,
    )


def find_subject_pairs(data_dir: str) -> List[Tuple[str, str]]:
    """
    Escanea un directorio y empareja archivos *-PSG.edf con su
    *-Hypnogram.edf correspondiente.

    Convención de nombres de Sleep-EDF: los primeros 6 caracteres
    identifican sujeto+noche (p. ej. 'SC4001'); el 7mo carácter
    difiere entre PSG ('E0') y Hypnogram ('EC'), por lo que el
    emparejamiento se hace por ese prefijo de 6 caracteres.

    Devuelve una lista de tuplas (psg_path, hypnogram_path).
    """
    psg_files = sorted(glob.glob(os.path.join(data_dir, "*-PSG.edf")))
    hypno_files = sorted(glob.glob(os.path.join(data_dir, "*-Hypnogram.edf")))

    hypno_by_prefix = {os.path.basename(f)[:6]: f for f in hypno_files}

    pairs: List[Tuple[str, str]] = []
    unmatched: List[str] = []
    for psg_path in psg_files:
        prefix = os.path.basename(psg_path)[:6]
        hypno_path = hypno_by_prefix.get(prefix)
        if hypno_path:
            pairs.append((psg_path, hypno_path))
        else:
            unmatched.append(psg_path)

    if unmatched:
        print(f"[loader] Aviso: {len(unmatched)} PSG sin hypnograma emparejado: {unmatched}")

    return pairs


if __name__ == "__main__":
    # Prueba rápida manual contra un solo sujeto
    import sys

    psg_path = sys.argv[1] if len(sys.argv) > 1 else "SC4001E0-PSG.edf"
    hypno_path = sys.argv[2] if len(sys.argv) > 2 else "SC4001EC-Hypnogram.edf"

    subject = load_subject(psg_path, hypno_path)
    print(f"Sujeto: {subject.subject_id}")
    print(f"Canales: {subject.raw.ch_names}")
    print(f"sfreq: {subject.raw.info['sfreq']} Hz")
    print(f"Duración PSG: {subject.raw.n_times / subject.raw.info['sfreq']:.1f} s")
    print(f"N° anotaciones: {len(subject.annotations)}")
