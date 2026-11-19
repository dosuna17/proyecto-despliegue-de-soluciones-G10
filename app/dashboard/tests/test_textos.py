"""Pruebas de los textos con cantidades: singular/plural y escala del visor."""

from graficas import texto_escala
from textos import epocas


def test_epocas_singular_plural_y_miles():
    assert epocas(1) == "1 época"
    assert epocas(0) == "0 épocas"
    assert epocas(2) == "2 épocas"
    assert epocas(1239) == "1.239 épocas"


def test_escala_del_visor_nunca_muestra_cero():
    assert texto_escala(37.4) == "37"
    assert texto_escala(1.6) == "2"
    assert texto_escala(0.42) == "0.4"
    assert texto_escala(0.048) == "0.05"
