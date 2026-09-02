# Capa de conexión entre el dashboard y el modelo de IA.

# Hoy usa datos simulados (mock_data.py). Cuando el endpoint de FastAPI
# esté listo, esta es la ÚNICA función que hay que modificar — el resto
# de app.py no debería cambiar.

import mock_data

# Función para obtener la predicción del modelo para una época específica de un estudio:
# Devuelve un diccionario con la predicción, la confianza, las probabilidades
# de cada clase y la etiqueta experta correspondiente:
def obtener_prediccion(indice):
    etapas = mock_data.obtener_etapas()
    confianza = mock_data.obtener_confianza()
    etiquetas_expertas = mock_data.obtener_etiquetas_expertas()

    prediccion = etapas[indice]
    conf = confianza[indice]
    etiqueta_experta = etiquetas_expertas[indice]

    clases = ["W", "N1", "N2", "N3", "REM"]
    resto = (1 - conf) / 4
    probabilidades = {c: (conf if c == prediccion else resto) for c in clases}

    return {
        "prediccion": prediccion,
        "confianza": conf,
        "probabilidades": probabilidades,
        "etiqueta_experta": etiqueta_experta,
    }

# Cuando este listo:
# import requests

# def obtener_prediccion(indice):
#     respuesta = requests.post("http://IP_DE_LA_EC2:8000/predict", json={"epoca": indice})
#     return respuesta.json()