"""Genera los estudios de demostración del tablero a partir de Sleep Cassette.

Toma registros de personas del conjunto de PRUEBA (columna ``split`` de
data/SC-subjects-split.xlsx), para que la demostración muestre al modelo
clasificando personas que nunca vio durante el entrenamiento.

Por defecto elige 2 personas: la de menor y la de mayor edad del conjunto
de prueba (noche 1), para mostrar el efecto de la edad en el sueño.

Cada estudio se guarda como .npz comprimido con:
- las señales en µV a 100 Hz (float16),
- el hipnograma del experto por época,
- metadatos (edad, sexo, noche) sin ningún dato identificable.

Para reducir el tamaño se conserva el periodo de sueño del experto más un
margen de vigilia (60 min por defecto), mayor que los 30 min que usa el
tablero: así la demostración también ejercita el recorte automático.

Uso (desde la raíz del repositorio):
    python scripts/generar_estudios_demo.py \
        --datos data/sleep-edfx/sleep-cassette \
        --particion data/SC-subjects-split.xlsx
    # sujetos concretos:   --sujetos 4 71
    # incluir EMG:          --incluir-emg
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "app" / "dashboard"))

from config import cargar_config  # noqa: E402
from registro import Registro, guardar_demo, leer_edf  # noqa: E402

ESTADIOS_SUENO = ("N1", "N2", "N3", "REM")


def elegir_sujetos(particion: pd.DataFrame, cantidad: int) -> list[int]:
    prueba = particion[particion["split"] == "test"].drop_duplicates("subject").sort_values("age")
    if prueba.empty:
        raise SystemExit("La partición no tiene sujetos con split = 'test'.")
    posiciones = np.linspace(0, len(prueba) - 1, num=min(cantidad, len(prueba))).round().astype(int)
    return prueba.iloc[posiciones]["subject"].astype(int).tolist()


def buscar_archivos(carpeta: Path, sujeto: int, noche: int) -> tuple[str, str]:
    prefijo = f"SC4{sujeto:02d}{noche}"
    psg = glob.glob(str(carpeta / f"{prefijo}*-PSG.edf"))
    hipno = glob.glob(str(carpeta / f"{prefijo}*-Hypnogram.edf"))
    if not psg or not hipno:
        raise FileNotFoundError(f"No se encontraron PSG e hipnograma para {prefijo} en {carpeta}.")
    return psg[0], hipno[0]


def recortar(registro: Registro, margen_min: int, epoca_s: int) -> Registro:
    etiquetas = registro.etiquetas_experto
    sueno = np.flatnonzero(np.isin(etiquetas, ESTADIOS_SUENO))
    margen = margen_min * 60 // epoca_s
    inicio = max(0, sueno[0] - margen)
    fin = min(len(etiquetas), sueno[-1] + margen + 1)
    muestras = registro.frecuencia_hz * epoca_s
    senales = {c: s[inicio * muestras:fin * muestras] for c, s in registro.senales.items()}
    nuevo_inicio = None
    if registro.inicio is not None:
        nuevo_inicio = registro.inicio + timedelta(seconds=int(inicio) * epoca_s)
    return Registro(
        nombre=registro.nombre, origen="demo", frecuencia_hz=registro.frecuencia_hz, senales=senales,
        inicio=nuevo_inicio, etiquetas_experto=etiquetas[inicio:fin],
        canales_en_archivo=list(senales),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datos", required=True, type=Path, help="Carpeta con los EDF de Sleep Cassette.")
    parser.add_argument("--particion", required=True, type=Path, help="SC-subjects-split.xlsx (con columna split).")
    parser.add_argument("--salida", type=Path, default=None, help="Carpeta de salida (por defecto la de configs/app.yaml).")
    parser.add_argument("--cantidad", type=int, default=2, help="Número de personas si no se indican --sujetos.")
    parser.add_argument("--sujetos", type=int, nargs="*", help="Números de sujeto concretos (deben ser de prueba).")
    parser.add_argument("--noche", type=int, default=1, choices=(1, 2))
    parser.add_argument("--margen-min", type=int, default=60, help="Vigilia conservada antes y después del sueño.")
    parser.add_argument("--incluir-emg", action="store_true", help="Guardar también el EMG submental.")
    args = parser.parse_args()

    config = cargar_config(entorno={})
    salida = args.salida or Path(config["estudios_demo"]["carpeta"])
    salida.mkdir(parents=True, exist_ok=True)
    particion = pd.read_excel(args.particion)
    columnas = {c.lower().split(" ")[0]: c for c in particion.columns}
    particion = particion.rename(columns={columnas["subject"]: "subject", columnas["age"]: "age",
                                          columnas["sex"]: "sex", columnas["split"]: "split"})

    sujetos = args.sujetos or elegir_sujetos(particion, args.cantidad)
    alias = dict(config["senales"]["alias"])
    if not args.incluir_emg:
        alias.pop("EMG submental", None)

    for sujeto in sujetos:
        fila = particion[particion["subject"] == sujeto].iloc[0]
        if fila["split"] != "test":
            raise SystemExit(f"El sujeto {sujeto} es de entrenamiento: no sirve como demostración.")
        psg, hipno = buscar_archivos(args.datos, sujeto, args.noche)
        nombre = os.path.basename(psg).split("-")[0]
        registro = leer_edf(psg, alias, config["senales"]["frecuencia_hz"], config["senales"]["duracion_epoca_s"],
                            hipno, nombre=nombre)
        registro = recortar(registro, args.margen_min, config["senales"]["duracion_epoca_s"])
        sexo = "mujer" if int(fila["sex"]) == 1 else "hombre"
        descripcion = f"{sexo.capitalize()} de {int(fila['age'])} años · noche {args.noche} · persona no vista en el entrenamiento"
        ruta = guardar_demo(registro, salida / f"{nombre}.npz", descripcion,
                            {"edad": str(int(fila["age"])), "sexo": sexo, "noche": str(args.noche)})
        horas = registro.duracion_s / 3600
        print(f"{ruta}  ({horas:.1f} h, {ruta.stat().st_size / 1e6:.1f} MB) · {descripcion}")


if __name__ == "__main__":
    main()
