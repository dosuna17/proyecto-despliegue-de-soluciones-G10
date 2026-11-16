"""Pruebas de la lógica del tablero que no depende de Streamlit ni de la API."""

import numpy as np
import pytest

from procesamiento import (
    bloques_secuencia,
    calcular_indicadores,
    comparar_con_experto,
    construir_bloque,
    kappa_cohen,
    numero_combinacion,
    preparar_entrada,
    periodo_principal_sueno,
    todas_las_combinaciones,
    validar_combinacion,
    Ventana,
)

CLASES = ["W", "N1", "N2", "N3", "REM"]
CANALES_TINYSLEEPNET = [
    "EEG Fpz-Cz",
    "EEG Pz-Oz",
    "EOG horizontal",
    "EMG submental",
]


# ---------------------------------------------------------------- combinaciones
def test_hay_doce_combinaciones_y_todas_tienen_eeg():
    combinaciones = todas_las_combinaciones()
    assert len(combinaciones) == 12
    assert len({frozenset(c) for c in combinaciones}) == 12
    assert all({"EEG Fpz-Cz", "EEG Pz-Oz"} & set(c) for c in combinaciones)


def test_numero_combinacion_sigue_la_tabla_de_la_propuesta():
    assert numero_combinacion(["EEG Fpz-Cz"]) == 1
    assert numero_combinacion(["EOG horizontal", "EEG Pz-Oz"]) == 6
    assert numero_combinacion(CANALES_TINYSLEEPNET[:3]) == 10
    assert numero_combinacion(CANALES_TINYSLEEPNET) == 12
    assert numero_combinacion(["EOG horizontal"]) is None


def test_validacion_exige_eeg_y_respeta_lo_que_admite_el_modelo():
    modelo = [c for c in todas_las_combinaciones() if "EMG submental" not in c]
    assert "al menos una señal EEG" in validar_combinacion(["EOG horizontal"], CANALES_TINYSLEEPNET, modelo)
    assert "no contiene" in validar_combinacion(["EEG Pz-Oz"], ["EEG Fpz-Cz"], modelo)
    assert "todavía no admite" in validar_combinacion(["EEG Fpz-Cz", "EMG submental"],
                                                      ["EEG Fpz-Cz", "EMG submental"], modelo)
    assert validar_combinacion(["EEG Pz-Oz", "EOG horizontal"], CANALES_TINYSLEEPNET, modelo) is None


# ---------------------------------------------------------------- entrada del modelo
def test_bloques_consecutivos_sin_traslape_como_en_la_evaluacion():
    # Igual que en el entrenamiento y el test: bloques de 15 sin traslape; las
    # épocas finales que no completan un bloque quedan sin clasificar.
    bloques = bloques_secuencia(40, 15)
    assert bloques == [(0, 15, 0), (15, 30, 15)]
    for inicio, fin, primera in bloques:
        assert fin - inicio == 15 and primera == inicio
    assert bloques_secuencia(45, 15)[-1] == (30, 45, 30)
    assert bloques_secuencia(10, 15) == []


def test_umbrales_por_estadio():
    from procesamiento import describir_umbrales, para_revisar

    umbrales = {"W": 0.0, "N1": 0.65, "N2": 0.0, "N3": 0.66, "REM": 0.75}
    estadios = ["W", "N1", "N1", "N2", "N3", "REM", "REM", "?"]
    confianzas = [0.30, 0.60, 0.70, 0.40, 0.65, 0.74, 0.90, np.nan]
    assert para_revisar(estadios, confianzas, umbrales).tolist() == [
        False, True, False, False, True, True, False, False,
    ]
    assert describir_umbrales(umbrales) == "N1 65 %, N3 66 %, REM 75 %; W y N2 no se marcan"


def test_bloque_respeta_contrato_de_ocho_canales():
    muestras = 3000
    senales = {"EEG Pz-Oz": np.arange(15 * muestras, dtype=np.float32)}
    bloque = construir_bloque(senales, ["EEG Pz-Oz"], CANALES_TINYSLEEPNET, 0, 15, muestras)
    assert bloque.shape == (15, 8, 3000)
    assert np.all(bloque[:, 0] == 0) and np.all(bloque[:, 2] == 0) and np.all(bloque[:, 3] == 0)
    assert np.all(bloque[:, 4] == 0) and np.all(bloque[:, 5] == 1)
    assert np.all(bloque[:, 6] == 0) and np.all(bloque[:, 7] == 0)
    assert bloque[1, 1, 0] == muestras  # la época 2 empieza donde termina la 1


# ---------------------------------------------------------------- recorte
def _noche(vigilia_inicio=200, sueno=600, vigilia_fin=300):
    return np.array(["W"] * vigilia_inicio + ["N2"] * sueno + ["W"] * vigilia_fin, dtype=object)


def test_recorte_deja_treinta_minutos_antes_y_despues():
    v = periodo_principal_sueno(_noche(), margen_epocas=60, max_interrupcion_epocas=120,
                                min_sueno_epocas=60, min_bloque_epocas=10)
    assert (v.inicio_sueno, v.fin_sueno) == (200, 799)
    assert (v.inicio, v.fin) == (140, 860)


def test_epocas_sueltas_en_la_vigilia_no_estiran_el_periodo():
    estadios = _noche()
    estadios[[20, 950]] = "N1"  # falsos positivos aislados
    v = periodo_principal_sueno(estadios, 60, 120, 60, min_bloque_epocas=10)
    assert (v.inicio_sueno, v.fin_sueno) == (200, 799)


def test_despertares_nocturnos_no_parten_la_noche_pero_una_siesta_lejana_si():
    estadios = _noche()
    estadios[450:480] = "W"  # despertar de 15 min
    v = periodo_principal_sueno(estadios, 60, 120, 60, 10)
    assert (v.inicio_sueno, v.fin_sueno) == (200, 799)
    con_siesta = np.concatenate([np.array(["N2"] * 40 + ["W"] * 400, dtype=object), estadios])
    v = periodo_principal_sueno(con_siesta, 60, 120, 60, 10)
    assert v.inicio_sueno == 440 + 200


def test_registro_sin_sueno_se_analiza_completo():
    v = periodo_principal_sueno(["W"] * 100, 60, 120, 60, 10)
    assert v.metodo == "sin_sueno" and (v.inicio, v.fin) == (0, 100)


# ---------------------------------------------------------------- indicadores
def test_indicadores_estandar():
    estadios = np.array(["W"] * 10 + ["N2"] * 20 + ["W"] * 4 + ["REM"] * 6 + ["W"] * 10, dtype=object)
    confianza = np.full(len(estadios), 0.9)
    confianza[[12, 13]] = 0.5
    v = periodo_principal_sueno(estadios, 10, 20, 5, 3)
    ind = calcular_indicadores(estadios, confianza, v, {c: 0.83 for c in CLASES}, 30, CLASES)
    assert ind.tiempo_total_sueno_min == 13.0
    assert ind.vigilia_tras_inicio_min == 2.0
    assert ind.latencia_rem_min == 12.0
    assert ind.epocas_baja_confianza == 2
    assert pytest.approx(sum(ind.porcentaje_sueno.values())) == 100


# ---------------------------------------------------------------- concordancia
def test_kappa_de_casos_conocidos():
    assert kappa_cohen(np.diag([10, 10])) == 1.0
    assert kappa_cohen(np.array([[5, 5], [5, 5]])) == 0.0


def test_comparacion_excluye_epocas_no_puntuadas_y_movimiento():
    pred = np.array(["W", "N2", "N2", "REM", "W"], dtype=object)
    exp = np.array(["W", "N2", "?", "M", "N1"], dtype=object)
    c = comparar_con_experto(pred, exp, Ventana(0, 5, 0, 4, "prediccion"), CLASES)
    assert c.epocas_comparadas == 3 and c.epocas_excluidas == 2
    assert c.exactitud == pytest.approx(2 / 3)
    assert c.matriz.sum() == 3


def test_preparar_entrada_reordena_canales_y_anade_mascaras_de_presencia():
    epocas = np.array(
        [
            [[10, 11, 12], [20, 21, 22]],
            [[30, 31, 32], [40, 41, 42]],
        ],
        dtype=np.float32,
    )

    entrada, presentes = preparar_entrada(
        epocas,
        ["EOG horizontal", "EEG Fpz-Cz"],
    )

    assert entrada.shape == (2, 8, 3)
    assert presentes == (True, False, True, False)
    np.testing.assert_array_equal(entrada[:, 0, :], epocas[:, 1, :])
    np.testing.assert_array_equal(entrada[:, 1, :], np.zeros((2, 3)))
    np.testing.assert_array_equal(entrada[:, 2, :], epocas[:, 0, :])
    np.testing.assert_array_equal(
        entrada[:, 4:, :],
        np.array(
            [
                [[1, 1, 1], [0, 0, 0], [1, 1, 1], [0, 0, 0]],
                [[1, 1, 1], [0, 0, 0], [1, 1, 1], [0, 0, 0]],
            ],
            dtype=np.float32,
        ),
    )


def test_preparar_entrada_rechaza_muestras_no_finitas():
    epocas = np.zeros((1, 1, 3), dtype=np.float32)
    epocas[0, 0, 1] = np.nan

    with pytest.raises(ValueError, match="valores no finitos"):
        preparar_entrada(epocas, ["EEG Fpz-Cz"])
