"""Pruebas de configuración, lectura de archivos y comunicación con la API.

La API se reemplaza por tests/api_simulada.py (mismo contrato que
app/api/main.py), así estas pruebas no necesitan MLflow ni modelos.
"""

import importlib

import numpy as np
import pytest
from fastapi.testclient import TestClient

import datos_sinteticos
from api_client import ClienteAPI, ErrorAPI
from config import RUTA_CONFIG_POR_DEFECTO, cargar_config
from registro import ErrorRegistro, guardar_demo, leer_demo, leer_edf, listar_demos, mapear_canales


@pytest.fixture(scope="module")
def config():
    return cargar_config(RUTA_CONFIG_POR_DEFECTO, entorno={})


@pytest.fixture(scope="module")
def archivos(tmp_path_factory):
    return datos_sinteticos.generar(tmp_path_factory.mktemp("edf"))


@pytest.fixture(scope="module")
def registro(config, archivos):
    psg, hipno = archivos
    return leer_edf(psg, config["senales"]["alias"], 100, 30, hipno)


class _SesionPrueba:
    """Adapta TestClient a la interfaz de requests.Session que usa ClienteAPI."""

    def __init__(self, cliente):
        self.cliente = cliente

    def get(self, url, timeout=None):
        return self.cliente.get(url.replace("http://api", ""))

    def post(self, url, json=None, timeout=None):
        return self.cliente.post(url.replace("http://api", ""), json=json)


def _cliente(config, monkeypatch, con_models=False):
    monkeypatch.setenv("API_SIMULADA_CON_MODELS", "1" if con_models else "0")
    import api_simulada
    importlib.reload(api_simulada)
    cfg = {**config, "api": {**config["api"], "url": "http://api"}}
    cliente = ClienteAPI(cfg)
    cliente.sesion = _SesionPrueba(TestClient(api_simulada.app))
    return cliente


# ---------------------------------------------------------------- configuración
def test_variables_de_entorno_tienen_prioridad(config):
    cfg = cargar_config(RUTA_CONFIG_POR_DEFECTO, entorno={
        "SLEEP_STAGING_API_URL": "http://api:8000/",
        "TABLERO_UMBRALES_CONFIANZA": "N1=0.7, REM=0.8",
    })
    assert cfg["api"]["url"] == "http://api:8000"
    umbrales = cfg["interpretacion"]["umbrales_confianza"]
    assert umbrales["N1"] == 0.7 and umbrales["REM"] == 0.8
    assert umbrales["N3"] == 0.66  # los demás estadios se conservan del YAML


def test_valor_de_entorno_invalido_se_explica():
    with pytest.raises(ValueError, match="TABLERO_UMBRALES_CONFIANZA"):
        cargar_config(RUTA_CONFIG_POR_DEFECTO, entorno={"TABLERO_UMBRALES_CONFIANZA": "alto"})


# ---------------------------------------------------------------- lectura de archivos
def test_alias_de_canales_no_distinguen_mayusculas(config):
    encontrados = mapear_canales(["FPZ-CZ", "EOG", "Resp oro-nasal"], config["senales"]["alias"])
    assert encontrados == {"EEG Fpz-Cz": "FPZ-CZ", "EOG horizontal": "EOG"}


def test_lee_edf_en_microvoltios_con_hipnograma(registro):
    assert set(registro.canales_disponibles) == {"EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal", "EMG submental"}
    assert registro.frecuencia_hz == 100
    assert 5 < np.std(registro.senales["EEG Fpz-Cz"]) < 200  # µV, no voltios
    esperado = datos_sinteticos.estadios_por_epoca()
    assert np.array_equal(registro.etiquetas_experto[: len(esperado)], esperado)


def test_archivo_que_no_es_edf_se_rechaza_con_mensaje(config, tmp_path):
    falso = tmp_path / "x.edf"
    falso.write_bytes(b"no es un edf")
    with pytest.raises(ErrorRegistro, match="EDF válido"):
        leer_edf(falso, config["senales"]["alias"], 100, 30)


def test_estudio_demo_ida_y_vuelta(registro, tmp_path):
    ruta = guardar_demo(registro, tmp_path / "demo.npz", "prueba", {"edad": "33"})
    demos = listar_demos(tmp_path)
    assert demos[0]["descripcion"] == "prueba" and demos[0]["edad"] == "33"
    copia = leer_demo(ruta)
    assert copia.canales_disponibles == registro.canales_disponibles
    assert np.max(np.abs(copia.senales["EEG Fpz-Cz"] - registro.senales["EEG Fpz-Cz"])) < 0.25
    assert np.array_equal(copia.etiquetas_experto, registro.etiquetas_experto)


# ---------------------------------------------------------------- API
def test_sin_endpoint_models_usa_la_configuracion(config, monkeypatch):
    modelo = _cliente(config, monkeypatch).info_modelo()
    assert modelo.origen == "config"
    assert len(modelo.combinaciones) == 12  # tres opciones de EEG por cuatro combinaciones de EOG/EMG


def test_con_endpoint_models_usa_lo_que_dice_la_api(config, monkeypatch):
    modelo = _cliente(config, monkeypatch, con_models=True).info_modelo()
    assert modelo.origen == "api" and "simulado" in modelo.nombre


@pytest.mark.parametrize("canales", [["EEG Fpz-Cz"], ["EEG Pz-Oz", "EOG horizontal"],
                                     ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal"],
                                     ["EEG Fpz-Cz", "EOG horizontal", "EMG submental"]])
def test_clasifica_todas_las_epocas_con_cada_combinacion(config, monkeypatch, registro, canales):
    cliente = _cliente(config, monkeypatch)
    modelo = cliente.info_modelo()
    n = int(registro.duracion_s // 30)
    avance = []
    resultado = cliente.clasificar(registro.senales, canales, modelo, n, lambda h, t: avance.append((h, t)))
    assert not np.any(resultado.estadios == "?")  # ninguna época sin clasificar
    assert resultado.probabilidades.shape == (n, 5)
    assert avance[-1][0] == avance[-1][1]


def test_agrupa_cuatro_secuencias_en_una_solicitud(config, monkeypatch):
    cliente = _cliente(config, monkeypatch)
    modelo = cliente.info_modelo()
    llamadas = []
    post_original = cliente.sesion.post

    def post_contabilizado(url, json=None, timeout=None):
        llamadas.append(np.asarray(json["epochs"]).shape)
        return post_original(url, json=json, timeout=timeout)

    cliente.sesion.post = post_contabilizado
    tiempo = np.arange(60 * 3000) / 100
    senales = {"EEG Fpz-Cz": (20 * np.sin(2 * np.pi * 10 * tiempo)).astype(np.float32)}

    resultado = cliente.clasificar(senales, ["EEG Fpz-Cz"], modelo, 60)

    assert len(llamadas) == 1
    assert llamadas[0] == (4, 15, 8, 3000)
    assert not np.any(resultado.estadios == "?")


def test_error_de_validacion_llega_en_lenguaje_llano(config, monkeypatch, registro):
    cliente = _cliente(config, monkeypatch)
    modelo = cliente.info_modelo()
    modelo.canales_modelo = ["EEG Fpz-Cz"]  # contrato roto a propósito
    with pytest.raises(ErrorAPI) as error:
        cliente.clasificar(registro.senales, ["EEG Fpz-Cz"], modelo, 30)
    assert "no aceptó" in error.value.mensaje
