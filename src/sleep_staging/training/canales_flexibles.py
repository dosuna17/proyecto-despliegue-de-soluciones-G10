"""
Canales flexibles para TinySleepNet: ceros + marca de presencia, con cada
secuencia repetida en todas las combinaciones de canales.

Idea (la misma que usa el XGBoost multicanal de Santiago, notebook 07):
  - El modelo tiene siempre las mismas posiciones de entrada, en orden fijo.
  - Si un canal no esta en la combinacion, su posicion se rellena con ceros y
    se acompana de una marca de presencia (0 = ausente, 1 = presente).
  - Cada secuencia de entrenamiento se usa una vez por combinacion (6 copias
    con las 3 senales de 100 Hz). Las 15 epocas de una misma copia comparten
    la combinacion: en la practica un estudio llega siempre con los mismos
    canales toda la noche, y la LSTM recorre esas 15 epocas juntas.
  - Siempre hay al menos un EEG en cada combinacion.

Entrada del modelo: (secuencia, 2 * n_posiciones, 3000). Con 3 posiciones son
6 canales (3 de senal y 3 de marca), asi que el modelo se crea con
input_channels = n_canales_entrada().

Para agregar el EMG mas adelante: anadir "EMG submental" al final de POSICIONES
(y que los .npz traigan ese canal en esa misma posicion). Las combinaciones
pasan de 6 a 12 solas.

Las senales entran en microvoltios, sin normalizar, igual que la version base.
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.append(str(Path(__file__).resolve().parents[2]))  # carpeta src/
from data.dataset import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    DEFAULT_SEQ_LENGTH,
    SleepSequenceDataset,
    SubjectRecording,
)

# Orden fijo de las posiciones = orden de los canales dentro de los .npz de
# 3 canales (scripts/generar_3canales.py). No cambiarlo sin regenerar los datos.
POSICIONES: List[str] = ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal"]
POSICIONES_EEG: List[str] = ["EEG Fpz-Cz", "EEG Pz-Oz"]
# Con el EMG submental como cuarta posicion (modelo TinySleepNetEMG). El EMG viene a
# 1 Hz (30 valores por epoca): en el tensor viaja repetido REPETIR_EMG veces.
POSICIONES_EMG: List[str] = POSICIONES + ["EMG submental"]
REPETIR_EMG = 100   # 30 valores x 100 = 3000 muestras

Combinacion = Tuple[str, Tuple[bool, ...]]  # (nombre, presente por posicion)


def _slug(canal: str) -> str:
    return canal.lower().replace(" ", "_").replace("-", "_")


def combinaciones_validas(
    posiciones: Sequence[str] = POSICIONES,
    eeg: Sequence[str] = POSICIONES_EEG,
) -> List[Combinacion]:
    """
    Todas las combinaciones con al menos un EEG y cualquier subconjunto de los
    demas canales. Mismos nombres y mismo orden que build_channel_combinations
    del notebook 07 de Santiago (p. ej. "eeg_fpz_cz__eog_horizontal").
    """
    eeg_sub = [list(c) for n in range(1, len(eeg) + 1) for c in combinations(eeg, n)]
    otros = [p for p in posiciones if p not in eeg]
    otros_sub = [list(c) for n in range(len(otros) + 1) for c in combinations(otros, n)]

    resultado: List[Combinacion] = []
    for opcional in otros_sub:
        for e in eeg_sub:
            canales = e + opcional
            nombre = "__".join(_slug(c) for c in canales)
            presentes = tuple(p in canales for p in posiciones)
            resultado.append((nombre, presentes))
    return resultado


def n_canales_entrada(posiciones: Sequence[str] = POSICIONES) -> int:
    """Canales que recibe la primera capa: una senal y una marca por posicion."""
    return 2 * len(posiciones)


def aplicar_combinacion(x: torch.Tensor, presentes: Sequence[bool]) -> torch.Tensor:
    """
    x: (secuencia, n_posiciones, muestras) con todas las senales.
    Devuelve (secuencia, 2 * n_posiciones, muestras): las senales ausentes en
    cero seguidas de las marcas de presencia.
    """
    p = torch.tensor(presentes, dtype=x.dtype).view(1, -1, 1)
    senales = x * p
    marcas = p.expand(x.shape[0], -1, x.shape[2])
    return torch.cat([senales, marcas], dim=1)


# Texto con el que se reconoce cada posicion en el nombre de un canal (sin importar
# mayusculas), igual que hace la API del XGBoost ("fpz-cz", "pz-oz", "eog").
_CLAVES_POSICION = {
    "EEG Fpz-Cz": "fpz-cz",
    "EEG Pz-Oz": "pz-oz",
    "EOG horizontal": "eog",
    "EMG submental": "emg",
}


def preparar_entrada(
    epocas,
    nombres_canales: Sequence[str],
    posiciones: Sequence[str] = POSICIONES,
    emg=None,
) -> Tuple[torch.Tensor, Tuple[bool, ...]]:
    """
    Para la API: arma la entrada del modelo flexible a partir de lo que llega.

    epocas: (n_epocas, n_canales, muestras), canales en el orden de nombres_canales.
    nombres_canales: p. ej. ["EEG Fpz-Cz", "EOG horizontal"], en cualquier orden.
    Devuelve (tensor (n_epocas, 2 * n_posiciones, muestras), presentes por posicion).
    Cada senal va a su posicion fija; las que no llegan quedan en cero con marca 0.
    Falla con un mensaje claro si hay un canal desconocido o repetido, si no
    hay ningun EEG o si el numero de nombres no coincide con los datos.

    EMG (solo con el modelo TinySleepNetEMG, posiciones=POSICIONES_EMG): el EMG
    llega a 1 Hz, asi que NO va en `epocas` sino en `emg`, con forma (n_epocas, 30);
    se repite 100 veces para tener 3000 muestras. Con emg=None el EMG queda
    ausente (ceros y marca 0). Con posiciones=POSICIONES, `emg` debe ser None.

    Ojo: la salida es por epoca. El modelo espera secuencias de 15 epocas
    consecutivas (seq_length), asi que quien llame debe agruparlas de a 15 antes
    de pasarlas a la red; las que sobren al final se descartan, igual que en el
    entrenamiento.
    """
    x = torch.as_tensor(epocas, dtype=torch.float32)
    if x.ndim != 3 or x.shape[1] != len(nombres_canales):
        raise ValueError(
            f"epocas debe tener forma (n_epocas, n_canales, muestras) con n_canales = "
            f"{len(nombres_canales)}; llego {tuple(x.shape)}."
        )
    completo = torch.zeros(x.shape[0], len(posiciones), x.shape[2], dtype=x.dtype)
    presentes = [False] * len(posiciones)
    pos_emg = posiciones.index("EMG submental") if "EMG submental" in posiciones else None
    for i, nombre in enumerate(nombres_canales):
        coincide = [
            k for k, pos in enumerate(posiciones)
            if _CLAVES_POSICION.get(pos, pos.lower()) in nombre.lower()
        ]
        if coincide and coincide[0] == pos_emg:
            raise ValueError("El EMG llega a 1 Hz (30 valores por epoca): pasalo con el argumento "
                             "emg=(n_epocas, 30), no dentro de `epocas`.")
        if len(coincide) != 1:
            raise ValueError(f"No reconozco el canal '{nombre}'. Posiciones validas: {list(posiciones)}.")
        k = coincide[0]
        if presentes[k]:
            raise ValueError(f"El canal '{posiciones[k]}' llego repetido.")
        completo[:, k] = x[:, i]
        presentes[k] = True
    if emg is not None:
        if pos_emg is None:
            raise ValueError("Llego `emg`, pero `posiciones` no incluye el EMG (usa POSICIONES_EMG).")
        e = torch.as_tensor(emg, dtype=torch.float32)
        if e.ndim != 2 or e.shape != (x.shape[0], 30):
            raise ValueError(f"emg debe tener forma (n_epocas, 30) = {(x.shape[0], 30)}; llego {tuple(e.shape)}.")
        completo[:, pos_emg] = e.repeat_interleave(REPETIR_EMG, dim=-1)
        presentes[pos_emg] = True
    if not any(presentes[k] for k, pos in enumerate(posiciones) if pos in POSICIONES_EEG):
        raise ValueError("Se necesita al menos un EEG (EEG Fpz-Cz o EEG Pz-Oz).")
    return aplicar_combinacion(completo, presentes), tuple(presentes)


class FlexibleSequenceDataset(Dataset):
    """
    Envuelve SleepSequenceDataset (mismo corte fijo de secuencias de 15 epocas)
    y entrega cada secuencia una vez por combinacion, aplicando la mascara al
    pedir el elemento (no se guardan copias en memoria).
    El elemento i corresponde a la secuencia i // n_combinaciones con la
    combinacion i % n_combinaciones.
    """

    def __init__(
        self,
        recordings: List[SubjectRecording],
        combinaciones: Sequence[Combinacion],
        seq_length: int = DEFAULT_SEQ_LENGTH,
        posiciones: Sequence[str] = POSICIONES,
    ):
        n_canales = recordings[0].X.shape[1]
        if n_canales != len(posiciones):
            raise ValueError(
                f"Los datos traen {n_canales} canales y POSICIONES tiene "
                f"{len(posiciones)}. Revisa que SLEEP_NPZ_DIR apunte a la carpeta de 3 canales."
            )
        if not combinaciones:
            raise ValueError("Se necesita al menos una combinacion.")
        self.base = SleepSequenceDataset(recordings, seq_length=seq_length)
        self.combinaciones = list(combinaciones)

    def __len__(self) -> int:
        return len(self.base) * len(self.combinaciones)

    def __getitem__(self, idx: int):
        i_base, i_comb = divmod(idx, len(self.combinaciones))
        x, y = self.base[i_base]
        _, presentes = self.combinaciones[i_comb]
        return aplicar_combinacion(x, presentes), y


def build_flexible_dataloader(
    recordings: List[SubjectRecording],
    combinaciones: Sequence[Combinacion],
    seq_length: int = DEFAULT_SEQ_LENGTH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    shuffle: bool = True,
    posiciones: Sequence[str] = POSICIONES,
) -> DataLoader:
    """Igual que build_dataloader de dataset.py, pero con combinaciones de canales."""
    dataset = FlexibleSequenceDataset(recordings, combinaciones, seq_length=seq_length,
                                      posiciones=posiciones)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def combinacion_por_nombre(nombre: str, posiciones: Sequence[str] = POSICIONES) -> Combinacion:
    """Para los modelos de control: una sola combinacion, p. ej. 'eeg_fpz_cz'."""
    for comb in combinaciones_validas(posiciones):
        if comb[0] == nombre:
            return comb
    raise ValueError(f"Combinacion desconocida: {nombre}")


# ---------------------------------------------------------------------
# Prueba rapida con datos sinteticos (no necesita los .npz ni GPU):
#     python src\sleep_staging\training\canales_flexibles.py
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np

    combos = combinaciones_validas()
    print("Combinaciones (3 posiciones):")
    for nombre, presentes in combos:
        print(f"  {nombre:40s} {presentes}")
    assert len(combos) == 6
    assert all(c[1][0] or c[1][1] for c in combos), "toda combinacion debe tener un EEG"
    assert len(combinaciones_validas(POSICIONES + ["EMG submental"])) == 12

    rng = np.random.default_rng(0)
    rec = SubjectRecording(
        recording_id="SCsint", subject_code="99",
        X=rng.normal(size=(65, 3, 3000)).astype(np.float32) * 20,
        y=rng.integers(0, 5, size=65).astype(np.int64),
    )
    ds = FlexibleSequenceDataset([rec], combos)
    n_base = len(ds.base)
    print(f"\nSecuencias base: {n_base} | con combinaciones: {len(ds)}")
    assert len(ds) == n_base * 6

    for i in range(len(ds)):
        x, y = ds[i]
        assert x.shape == (15, n_canales_entrada(), 3000), x.shape
        assert y.shape == (15,)
        _, presentes = combos[i % 6]
        x0, _ = ds.base[i // 6]
        for pos, p in enumerate(presentes):
            marca = x[:, 3 + pos]
            senal = x[:, pos]
            if p:
                assert (marca == 1).all() and torch.equal(senal, x0[:, pos])
            else:
                assert (marca == 0).all() and (senal == 0).all()

    xb, yb = next(iter(build_flexible_dataloader([rec], combos, batch_size=4, shuffle=False)))
    print("Lote de ejemplo:", tuple(xb.shape), tuple(yb.shape))

    from sleep_staging.models.tinysleepnet import TinySleepNet, TinySleepNetConfig
    modelo = TinySleepNet(TinySleepNetConfig(input_channels=n_canales_entrada()))
    logits, _ = modelo(xb)
    print("Salida del modelo:", tuple(logits.shape), "(esperado (4, 15, 5))")
    print("\nTodo OK")
