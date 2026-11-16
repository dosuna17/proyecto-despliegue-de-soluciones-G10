"""Textos que ve el usuario, en un solo lugar.

Cada indicador tiene una ayuda que explica qué mide y qué le dice al usuario.
Centralizarlos evita contradicciones entre pantallas y facilita revisarlos.
Se trata al usuario de "usted" en toda la interfaz.
"""

TITULO = "Asistente de clasificación de estadios del sueño"
SUBTITULO = (
    "Clasifica automáticamente cada periodo de 30 segundos de una polisomnografía "
    "en vigilia (W), sueño ligero (N1, N2), sueño profundo (N3) o sueño REM."
)

AVISO_APOYO = (
    "**Herramienta de apoyo, no de diagnóstico.** La clasificación la genera un modelo "
    "de aprendizaje automático y puede contener errores. Debe ser revisada por un "
    "profesional del sueño antes de usarse en cualquier decisión clínica."
)

PASOS = [
    ("1. Cargue un registro", "Use un estudio de demostración o suba su archivo EDF."),
    ("2. Elija las señales", "Marque las señales que desea usar; se necesita al menos un EEG."),
    ("3. Revise el resultado", "Hipnograma, indicadores, señal por época y descarga en CSV."),
]

NOMBRES_ESTADIO = {
    "W": "Vigilia",
    "N1": "N1 · sueño ligero",
    "N2": "N2 · sueño ligero",
    "N3": "N3 · sueño profundo",
    "REM": "REM",
    "M": "Movimiento",
    "?": "Sin clasificar",
}

NOMBRES_CANAL = {
    "EEG Fpz-Cz": "EEG Fpz-Cz (frontal)",
    "EEG Pz-Oz": "EEG Pz-Oz (posterior)",
    "EOG horizontal": "EOG horizontal (movimiento ocular)",
    "EMG submental": "EMG submental (tono muscular del mentón)",
}

AYUDA_CANAL = {
    "EEG Fpz-Cz": "Actividad eléctrica cerebral frontal. Clave para distinguir sueño profundo (N3).",
    "EEG Pz-Oz": "Actividad eléctrica cerebral posterior. Muestra bien el ritmo alfa de la vigilia relajada.",
    "EOG horizontal": "Movimientos de los ojos. Ayuda a reconocer el sueño REM y la vigilia.",
    "EMG submental": "Tono muscular del mentón. Disminuye en REM; ayuda a separarlo de la vigilia.",
}

AYUDA = {
    "ventana": (
        "El modelo se entrenó con la noche recortada a 30 minutos antes y después del sueño. "
        "Por eso el análisis se hace en esa misma ventana: primero se clasifica todo el "
        "registro y luego se ubica el periodo principal de sueño a partir de esa clasificación."
    ),
    "tiempo_total_sueno": (
        "Suma de las épocas clasificadas como N1, N2, N3 o REM. Indica cuánto durmió la persona "
        "dentro de la ventana analizada."
    ),
    "inicio_fin": (
        "Hora de la primera y la última época de sueño. Delimitan el periodo de sueño usado "
        "para los demás indicadores."
    ),
    "vigilia_tras_inicio": (
        "Minutos despierto entre el inicio y el fin del sueño (WASO, por sus siglas en inglés). "
        "Un valor alto indica un sueño fragmentado."
    ),
    "latencia_rem": (
        "Minutos desde que la persona se duerme hasta su primera época de sueño REM. En adultos "
        "suele estar entre 70 y 120 minutos, porque el sueño avanza en ciclos de unos 90 minutos; "
        "un valor muy distinto merece la revisión del especialista."
    ),
    "baja_confianza": (
        "Épocas en las que la confianza del modelo quedó por debajo del umbral del estadio que eligió "
        "({umbrales}). Cada umbral se fijó para que las épocas no marcadas de ese estadio tengan una "
        "precisión cercana a la de los mejores modelos publicados; los estadios sin umbral ya la alcanzan "
        "sin filtrar. Son las que más conviene revisar manualmente; puede recorrerlas en la sección de señales."
    ),
    "confianza": (
        "La confianza es la probabilidad que el modelo asigna al estadio elegido (de 0 a 100 %). "
        "No es una garantía de acierto: una época con confianza alta también puede estar mal "
        "clasificada, pero las de confianza baja merecen revisión prioritaria."
    ),
    "segunda_opcion": (
        "El estadio al que el modelo le dio la segunda probabilidad más alta. Indica entre qué dos estadios "
        "dudó: si la confianza es baja y la segunda opción es REM, por ejemplo, conviene mirar en la señal los "
        "movimientos oculares (EOG) y el tono muscular para decidir entre ambos."
    ),
    "distribucion": (
        "Minutos en cada estadio y su porcentaje sobre el tiempo total de sueño. Es un dato estándar de los "
        "informes de polisomnografía: muestra, por ejemplo, si hubo poco sueño profundo (N3) o poco REM."
    ),
    "hipnograma": (
        "Estadio clasificado a lo largo de la noche, en el orden clínico habitual (vigilia arriba, N3 abajo). "
        "Cada bloque de color es el estadio de esas épocas y la línea negra muestra los cambios entre estadios. "
        "Las marcas grises bajo el modelo son épocas para revisar. Si cargó el hipnograma del experto, se muestra "
        "debajo con el mismo formato, y la franja inferior marca en rojo las épocas en que no coinciden."
    ),
    "senal": (
        "Señales registradas, una por color, con el estadio clasificado en la franja superior de cada época "
        "(⚠ = baja confianza). Sirve para verificar en la señal por qué el modelo eligió cada estadio, igual "
        "que se hace al calificar un estudio. En la vista de 5 minutos, la época seleccionada se resalta en gris."
    ),
    "concordancia": (
        "Compara la clasificación del modelo con el hipnograma del experto que usted subió. "
        "La exactitud es el porcentaje de épocas en que coinciden; el kappa de Cohen corrige "
        "las coincidencias que ocurrirían por azar (1 = acuerdo perfecto, 0 = azar). Como "
        "referencia, dos expertos humanos suelen coincidir con un kappa cercano a {kappa_ref:.2f}."
    ),
    "matriz": (
        "Cada fila es lo que dijo el experto y cada columna lo que dijo el modelo. La diagonal son "
        "los aciertos; fuera de ella se ve con qué estadio se confunde cada uno."
    ),
    "descarga": (
        "El informe PDF reúne lo que ve en esta página para leerlo, imprimirlo o adjuntarlo. El Excel trae los "
        "mismos datos para analizarlos: una hoja de resumen, una fila por época con filtros, la lista de épocas "
        "para revisar y una guía de cómo leer cada columna."
    ),
}

ERROR_GENERICO = (
    "Ocurrió un problema inesperado. Intente de nuevo; si continúa, avise al equipo de soporte."
)
