"""Genera un registro EDF sintético (PSG + hipnograma) para probar el tablero.

No son datos fisiológicos: cada estadio se simula con su ritmo característico
(alfa en vigilia, theta en N1, husos en N2, ondas lentas en N3, theta y
movimientos oculares en REM) para que la API simulada pueda "clasificarlo" y
la interfaz se pruebe de principio a fin sin el dataset ni MLflow.

Uso:
    python app/dashboard/tests/datos_sinteticos.py carpeta_salida
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np

FRECUENCIA = 100
EPOCA_S = 30

# Una noche esquemática: vigilia larga al inicio y al final (como en Sleep Cassette)
# y cuatro ciclos de sueño en medio. (estadio, minutos)
SECUENCIA = [("W", 150), ("N1", 6), ("N2", 20), ("N3", 35), ("N2", 15), ("REM", 12),
             ("W", 3), ("N1", 4), ("N2", 30), ("N3", 25), ("N2", 15), ("REM", 20),
             ("N2", 35), ("N3", 10), ("N2", 20), ("REM", 25), ("W", 6),
             ("N1", 5), ("N2", 40), ("REM", 30), ("N1", 4), ("W", 240)]


def estadios_por_epoca() -> np.ndarray:
    return np.array([e for e, minutos in SECUENCIA for _ in range(minutos * 2)], dtype=object)


def _epoca(estadio: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.arange(EPOCA_S * FRECUENCIA) / FRECUENCIA
    ruido = lambda amp: rng.normal(0, amp, t.size)  # noqa: E731
    fase = rng.uniform(0, 2 * np.pi)
    if estadio == "W":
        eeg = 25 * np.sin(2 * np.pi * 10 * t + fase) + ruido(8)
        eog = 60 * np.sign(np.sin(2 * np.pi * 0.3 * t)) * (rng.random() > 0.5) + ruido(10)
    elif estadio == "N1":
        eeg = 30 * np.sin(2 * np.pi * 6 * t + fase) + ruido(8)
        eog = 30 * np.sin(2 * np.pi * 0.2 * t) + ruido(8)
    elif estadio == "N2":
        envolvente = np.exp(-((t % 8 - 4) ** 2) / 0.3)
        eeg = 25 * np.sin(2 * np.pi * 4 * t + fase) + 35 * envolvente * np.sin(2 * np.pi * 13 * t) + ruido(6)
        eog = ruido(8)
    elif estadio == "N3":
        eeg = 90 * np.sin(2 * np.pi * 1.2 * t + fase) + ruido(10)
        eog = 40 * np.sin(2 * np.pi * 1.2 * t + fase) + ruido(8)
    else:  # REM
        eeg = 18 * np.sin(2 * np.pi * 5 * t + fase) + ruido(6)
        saltos = (rng.random(t.size) < 0.002).cumsum() % 2
        eog = 80 * (saltos - 0.5) + ruido(10)
    # Un 6 % de épocas con artefacto para producir baja confianza.
    if rng.random() < 0.06:
        eeg = eeg + rng.normal(0, 60, t.size)
    emg = {"W": 20, "N1": 12, "N2": 10, "N3": 8, "REM": 3}[estadio] + rng.normal(0, 1, EPOCA_S)
    return eeg, eog, emg


def generar(carpeta: str | Path, semilla: int = 7) -> tuple[Path, Path]:
    import pyedflib

    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(semilla)
    estadios = estadios_por_epoca()
    fpz, pz, eog, emg = [], [], [], []
    for estadio in estadios:
        a, o, m = _epoca(estadio, rng)
        fpz.append(a)
        pz.append(0.8 * a + rng.normal(0, 5, a.size))
        eog.append(o)
        emg.append(m)
    # pyedflib escribe en las unidades físicas declaradas; las declaramos en µV.
    senales = [np.concatenate(fpz), np.concatenate(pz), np.concatenate(eog), np.concatenate(emg)]
    nombres = ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal", "EMG submental"]
    frecuencias = [FRECUENCIA, FRECUENCIA, FRECUENCIA, 1]
    inicio = datetime(2026, 3, 14, 20, 50, 0)

    ruta_psg = carpeta / "SINT001E0-PSG.edf"
    escritor = pyedflib.EdfWriter(str(ruta_psg), len(senales), file_type=pyedflib.FILETYPE_EDFPLUS)
    cabeceras = []
    for nombre, senal, frecuencia in zip(nombres, senales, frecuencias):
        maximo = float(np.ceil(np.abs(senal).max() + 1))
        cabeceras.append({
            "label": nombre, "dimension": "uV", "sample_frequency": frecuencia,
            "physical_max": maximo, "physical_min": -maximo, "digital_max": 32767, "digital_min": -32768,
        })
    escritor.setSignalHeaders(cabeceras)
    escritor.setStartdatetime(inicio)
    escritor.writeSamples(senales)
    escritor.close()

    ruta_hipno = carpeta / "SINT001EC-Hypnogram.edf"
    escritor = pyedflib.EdfWriter(str(ruta_hipno), 1, file_type=pyedflib.FILETYPE_EDFPLUS)
    escritor.setSignalHeaders([{"label": "dummy", "dimension": "uV", "sample_frequency": 1, "physical_max": 1,
                                "physical_min": -1, "digital_max": 32767, "digital_min": -32768}])
    escritor.setStartdatetime(inicio)
    nombres_rk = {"W": "Sleep stage W", "N1": "Sleep stage 1", "N2": "Sleep stage 2",
                  "N3": "Sleep stage 3", "REM": "Sleep stage R"}
    onset = 0
    for estadio, minutos in SECUENCIA:
        escritor.writeAnnotation(onset, minutos * 60, nombres_rk[estadio])
        onset += minutos * 60
    escritor.writeSamples([np.zeros(int(onset))])
    escritor.close()
    return ruta_psg, ruta_hipno


if __name__ == "__main__":
    salida = sys.argv[1] if len(sys.argv) > 1 else "datos_sinteticos"
    for ruta in generar(salida):
        print(ruta)
