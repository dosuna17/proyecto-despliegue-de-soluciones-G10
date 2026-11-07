"""Pruebas del informe descargable (PDF y Excel)."""

import io

import numpy as np
import pytest
from openpyxl import load_workbook

import informe
from procesamiento import calcular_indicadores, comparar_con_experto, periodo_principal_sueno

CLASES = ["W", "N1", "N2", "N3", "REM"]


def _datos(con_experto=True):
    experto = np.array(["W"] * 120 + (["N1"] * 10 + ["N2"] * 60 + ["N3"] * 40 + ["N2"] * 30 + ["REM"] * 40) * 3
                       + ["W"] * 100, dtype=object)
    modelo = experto.copy()
    modelo[200:215] = "N1"  # algunas diferencias
    n = len(modelo)
    confianza = np.full(n, 0.92)
    confianza[200:230] = 0.5
    probabilidades = np.zeros((n, 5))
    for i, estadio in enumerate(modelo):
        probabilidades[i] = (1 - confianza[i]) / 4
        probabilidades[i, CLASES.index(estadio)] = confianza[i]
    ventana = periodo_principal_sueno(modelo, 60, 120, 60, 10)
    indicadores = calcular_indicadores(modelo, confianza, ventana, 0.83, 30, CLASES)
    from datetime import datetime
    return informe.DatosInforme(
        nombre_registro="PRUEBA", inicio=datetime(2026, 1, 1, 22, 0), canales=["EEG Fpz-Cz", "EOG horizontal"],
        modelo="TinySleepNet", estadios=modelo, confianzas=confianza, probabilidades=probabilidades, clases=CLASES,
        ventana=ventana, indicadores=indicadores, umbral=0.83, kappa_referencia=0.76, duracion_epoca_s=30,
        experto=experto if con_experto else None,
        concordancia=comparar_con_experto(modelo, experto, ventana, CLASES) if con_experto else None,
        recorte={"margen_min": 30, "max_interrupcion_min": 60, "min_bloque_sueno_min": 5},
    )


@pytest.mark.parametrize("con_experto", [True, False])
def test_hallazgos_describen_el_analisis(con_experto):
    texto = " ".join(informe.hallazgos(_datos(con_experto)))
    assert "de sueño" in texto and "revisarlas" in texto
    assert ("kappa" in texto) == con_experto
    assert "profesional del sueño" in texto  # el aviso siempre cierra los hallazgos


@pytest.mark.parametrize("con_experto", [True, False])
def test_pdf_tiene_tres_paginas(con_experto):
    contenido = informe.generar_pdf(_datos(con_experto))
    assert contenido.startswith(b"%PDF")
    assert contenido.count(b"/Type /Page\n") + contenido.count(b"/Type /Page ") + contenido.count(b"/Type/Page") >= 3 \
        or contenido.count(b"/Page") >= 3


@pytest.mark.parametrize("con_experto", [True, False])
def test_excel_organizado_en_hojas_con_century_gothic(con_experto):
    datos = _datos(con_experto)
    libro = load_workbook(io.BytesIO(informe.generar_excel(datos)))
    assert libro.sheetnames == ["Resumen", "Épocas", "Para revisar", "Cómo leer"]
    epocas = libro["Épocas"]
    assert epocas.max_row == len(datos.estadios) + 1  # una fila por época + encabezado
    assert epocas["C1"].value == "Estadio (modelo)"
    assert epocas["A2"].font.name == "Century Gothic"
    assert libro["Resumen"]["A1"].font.name == "Century Gothic"
    textos_resumen = " ".join(str(c.value) for fila in libro["Resumen"].iter_rows() for c in fila if c.value)
    assert "Hallazgos" in textos_resumen
    assert ("Comparación con el hipnograma del experto" in textos_resumen) == con_experto
    revisar = libro["Para revisar"]
    confianzas = [revisar.cell(f, 4).value for f in range(5, revisar.max_row + 1)]
    assert confianzas == sorted(confianzas) and len(confianzas) == 30


def test_sexta_clase_movimiento_aparece_en_informe():
    """Cuando el modelo agregue M, el informe la incluye sin cambios de código."""

    datos = _datos()
    clases = CLASES + ["M"]
    datos.estadios = datos.estadios.copy()
    datos.estadios[300:304] = "M"
    datos.clases = clases
    datos.probabilidades = np.hstack([datos.probabilidades, np.zeros((len(datos.estadios), 1))])
    datos.indicadores = calcular_indicadores(datos.estadios, datos.confianzas, datos.ventana, 0.83, 30, clases)
    datos.concordancia = comparar_con_experto(datos.estadios, datos.experto, datos.ventana, clases)
    assert datos.indicadores.minutos_por_estadio["M"] == 2.0
    assert "M" in datos.concordancia.clases
    assert informe.generar_pdf(datos).startswith(b"%PDF")
    libro = load_workbook(io.BytesIO(informe.generar_excel(datos)))
    assert [c.value for c in libro["Épocas"][1]].count("Prob. M") == 1
