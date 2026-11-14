"""Configuración del tablero: lee configs/app.yaml y aplica variables de entorno.

Orden de prioridad (de mayor a menor):
1. Variable de entorno (ver ``VARIABLES_ENTORNO``).
2. Valor en configs/app.yaml (o en el archivo indicado por TABLERO_CONFIG).

Así el mismo código corre en local y en la nube cambiando solo el entorno.
"""

from __future__ import annotations

import copy
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import yaml

RAIZ_REPOSITORIO = Path(__file__).resolve().parents[2]
RUTA_CONFIG_POR_DEFECTO = RAIZ_REPOSITORIO / "configs" / "app.yaml"


def _a_booleano(valor: str) -> bool:
    return valor.strip().lower() in {"1", "true", "si", "sí", "yes", "on"}


# variable de entorno -> (ruta dentro del YAML, conversión)
VARIABLES_ENTORNO: Dict[str, Tuple[Tuple[str, ...], Callable[[str], Any]]] = {
    "SLEEP_STAGING_API_URL": (("api", "url"), str),
    "TABLERO_TIMEOUT_INFERENCIA_S": (("api", "timeout_inferencia_s"), float),
    "TABLERO_SOLICITUDES_PARALELAS": (("api", "solicitudes_paralelas"), int),
    "TABLERO_SECUENCIAS_POR_SOLICITUD": (("api", "secuencias_por_solicitud"), int),
    "TABLERO_UMBRAL_CONFIANZA": (("interpretacion", "umbral_confianza"), float),
    "TABLERO_TAMANO_MAXIMO_MB": (("carga", "tamano_maximo_mb"), int),
    "TABLERO_CARPETA_DEMOS": (("estudios_demo", "carpeta"), str),
}


def _asignar(destino: Dict[str, Any], ruta: Tuple[str, ...], valor: Any) -> None:
    nodo = destino
    for clave in ruta[:-1]:
        nodo = nodo.setdefault(clave, {})
    nodo[ruta[-1]] = valor


def cargar_config(ruta: str | Path | None = None, entorno: Dict[str, str] | None = None) -> Dict[str, Any]:
    """Devuelve la configuración combinada (YAML + variables de entorno)."""

    entorno = dict(os.environ) if entorno is None else entorno
    ruta = Path(ruta or entorno.get("TABLERO_CONFIG") or RUTA_CONFIG_POR_DEFECTO)
    if not ruta.is_file():
        raise FileNotFoundError(f"No se encontró el archivo de configuración del tablero: {ruta}")
    with ruta.open(encoding="utf-8") as archivo:
        config = yaml.safe_load(archivo) or {}
    config = copy.deepcopy(config)

    for variable, (ruta_yaml, convertir) in VARIABLES_ENTORNO.items():
        valor = entorno.get(variable)
        if valor not in (None, ""):
            try:
                _asignar(config, ruta_yaml, convertir(valor))
            except ValueError as error:
                raise ValueError(f"La variable {variable}={valor!r} no tiene un formato válido.") from error

    config["api"]["url"] = config["api"]["url"].rstrip("/")
    carpeta = Path(config["estudios_demo"]["carpeta"])
    if not carpeta.is_absolute():
        carpeta = RAIZ_REPOSITORIO / carpeta
    config["estudios_demo"]["carpeta"] = str(carpeta)
    _validar(config)
    return config


def _validar(config: Dict[str, Any]) -> None:
    umbral = config["interpretacion"]["umbral_confianza"]
    if not 0 < umbral < 1:
        raise ValueError("interpretacion.umbral_confianza debe estar entre 0 y 1.")
    if config["api"]["solicitudes_paralelas"] < 1:
        raise ValueError("api.solicitudes_paralelas debe ser al menos 1.")
    if not 1 <= config["api"].get("secuencias_por_solicitud", 1) <= 4:
        raise ValueError("api.secuencias_por_solicitud debe estar entre 1 y 4.")
    if config["recorte"]["margen_min"] < 0:
        raise ValueError("recorte.margen_min no puede ser negativo.")


@lru_cache(maxsize=1)
def obtener_config() -> Dict[str, Any]:
    """Configuración del proceso, leída una sola vez."""

    return cargar_config()
