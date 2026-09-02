
# Datos simulados para el dashboard de sleep staging.

# Estas funciones devuelven listas de prueba (escritas a mano) mientras
# el equipo no tiene todavía datos reales procesados ni el modelo conectado

# Cuando haya datos reales disponibles, estas funciones se  reemplazan
# para que lean el estudio real, sin cambiar cómo se llaman desde app.py.

#Etapas de sueño simuladas para un estudio de prueba (20 épocas):
def obtener_etapas():
    return ["W", "W", "N1", "N2", "N2", "N2", "N3", "N3", "N2", "REM",
            "REM", "N2", "N1", "N2", "N3", "N3", "N2", "REM", "REM", "W"]

# Confianza simulada del modelo (0 a 1) para cada época del estudio de prueba:
def obtener_confianza():
    return [0.95, 0.93, 0.55, 0.85, 0.90, 0.88, 0.92, 0.91, 0.60, 0.75,
            0.80, 0.58, 0.52, 0.86, 0.90, 0.89, 0.83, 0.77, 0.79, 0.94]

# Etiquetas del experto (hipnograma original), con algunas diferencias 
# respecto a la predicción del modelo para probar las alertas de comparación:
def obtener_etiquetas_expertas():
    return ["W", "W", "N1", "N2", "N2", "N2", "N3", "N3", "N1", "REM",
            "REM", "N1", "N1", "N2", "N3", "N3", "N2", "N2", "REM", "W"]