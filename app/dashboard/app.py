import math
import random
import mock_data
import streamlit as st
import plotly.graph_objects as go

# Inicializamos la variable de estado para controlar la pantalla actual:
if "screen" not in st.session_state:
    st.session_state.screen = "select"
# Inicializamos la variable de estado para almacenar el estudio seleccionado:
if "selected_study" not in st.session_state:
    st.session_state.selected_study = "SC4001E0"
# Inicializamos la variable de estado para almacenar el índice de la época seleccionada:
if "epoch_index" not in st.session_state:
      st.session_state.epoch_index = 0
# Inicializamos la variable de estado para almacenar las señales seleccionadas:
if "selected_channels" not in st.session_state:
    st.session_state.selected_channels = {}

# Pantalla 1:
# Creamos la interfaz de usuario con Streamlit:
if st.session_state.screen == "select":
        st.title("Sleep staging assistant")
        st.caption("Seleccione un estudio y configure las señales a utilizar")

        # Agregamos la lista de estudios disponibles para seleccionar
        st.subheader("1. Seleccionar estudio")  # Encabezado de la sección

        # Creamos los botones para seleccionar el estudio (solo se puede seleccionar uno a la vez):
        estudio_seleccionado = st.radio(
             "Estudios disponibles",
             ["SC4001E0", "SC4012E0", "ST7011J0"]
        )

        # Sección de señales a utilizar:
        st.subheader("2. Señales disponibles en este estudio")

        # Agregamos checkboxes para seleccionar las señales a utilizar:
        # La señal EEG Fpz-Cz es obligatoria y está preseleccionada y deshabilitada para que no se pueda desmarcar.
        eeg_fpz_cz = st.checkbox("EEG Fpz-Cz · obligatorio · 100 Hz", value=True, disabled=True)
        eeg_pz_oz = st.checkbox("EEG Pz-Oz · opcional · 100 Hz", value=True)
        eog = st.checkbox("EOG horizontal · opcional · 100 Hz", value=False)
        emg = st.checkbox("EMG submentoniano · opcional · 1 Hz", value=False)

        # Agregamos un mensaje informativo sobre la importancia de las señales:
        st.info(
        "Más señales pueden mejorar la clasificación, especialmente en épocas "
        "de transición, pero el modelo funciona con EEG únicamente.",icon="ℹ️" 
        )

# Guardamos el estudio seleccionado en la variable de estado y cambiamos a la pantalla 2 de análisis: 
        if st.button("Analizar estudio", type="primary"):
                st.session_state.selected_study = estudio_seleccionado
                st.session_state.selected_channels = {
                "EEG Fpz-Cz": eeg_fpz_cz,
                "EEG Pz-Oz": eeg_pz_oz,
                "EOG horizontal": eog,
                "EMG submentoniano": emg,
                }
                st.session_state.screen = "analysis"
                st.rerun()

# Pantalla 2:
# Si estamos en la pantalla de análisis, mostramos el estudio seleccionado y los canales activos:
else:
    st.title(f"Estudio {st.session_state.selected_study}")
    canales_activos = [nombre for nombre, activo in st.session_state.selected_channels.items() if activo]
    st.caption("Canales activos: " + ", ".join(canales_activos))
# Agregamos un botón para volver a la pantalla de selección de estudio:
    if st.button("Cambiar estudio", type="primary"):
        st.session_state.screen = "select"
        st.rerun()
# Mostramos los indicadores totales de épocas, la duración total del sueño y el porcentaje de sueño REM en la pantalla de análisis:
    etapas = mock_data.obtener_etapas()
# Agregamos una lista de confianza para cada época, que representa la confianza del modelo en la predicción de la etapa del sueño:
    confianza = mock_data.obtener_confianza()
# Agregamos una lista de etiquetas expertas para cada época, que representa la etapa del sueño determinada por un experto humano:
    etiquetas_expertas = mock_data.obtener_etiquetas_expertas()

    total_epocas = len(etapas)
    duracion_horas = (total_epocas * 30) / 3600
    porcentaje_rem = round(100 * etapas.count("REM") / total_epocas)
    epocas_baja_confianza = [i for i, c in enumerate(confianza) if c < 0.65]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Duración total", f"{duracion_horas:.2f} h")
    col2.metric("Épocas totales", total_epocas)
    col3.metric("Sueño REM", f"{porcentaje_rem}%")
    col4.metric("Baja confianza", f"{len(epocas_baja_confianza)} épocas")

# Agregamos un hipnograma interactivo utilizando Plotly para mostrar las etapas del sueño a lo largo del tiempo:
    st.subheader("Hipnograma")

# Definimos el orden vertical de las etapas del sueño y calculamos las posiciones en el eje y para cada etapa:
    orden_vertical = ["N3", "N2", "N1", "W", "REM"]
    posiciones_y = [orden_vertical.index(e) for e in etapas]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(total_epocas)),
        y=posiciones_y,
        mode="lines",
        line_shape="hv"
    ))

# Agregamos rectángulos verticales para resaltar las épocas con baja confianza en la predicción del modelo:
    for i in epocas_baja_confianza:
        fig.add_vrect(x0=i - 0.5, x1=i + 0.5, fillcolor="red", opacity=0.2, line_width=0)

    fig.add_vline(x=st.session_state.epoch_index, line_dash="dot", line_color="black")
    fig.update_layout(
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(len(orden_vertical))),
            ticktext=orden_vertical
        ),
        xaxis_title="Época",
        height=300
    )

    st.plotly_chart(fig, use_container_width=True)

# Agregamos un control deslizante para explorar las épocas individuales y mostrar la etapa del sueño correspondiente:
    st.subheader("Explorar época")

    st.session_state.epoch_index = st.slider(
        "Selecciona una época",
        min_value=0,
        max_value=total_epocas - 1,
        value=st.session_state.epoch_index
    )

# Mostramos la etapa del sueño predicha y la confianza del modelo para la época seleccionada:
    epoca_actual = etapas[st.session_state.epoch_index]
    st.write(f"Época {st.session_state.epoch_index}: predicha **{epoca_actual}**")

    st.write(f"Confianza: {int(confianza[st.session_state.epoch_index] * 100)}%")

    clases = ["W", "N1", "N2", "N3", "REM"]
    prediccion = etapas[st.session_state.epoch_index]
    conf = confianza[st.session_state.epoch_index]
    resto = (1 - conf) / 4

    prob_cols = st.columns(5)
    for col, clase in zip(prob_cols, clases):
        prob = conf if clase == prediccion else resto
        col.metric(clase, f"{int(prob * 100)}%")

# Agregamos la etiqueta experta correspondiente a la época seleccionada y mostramos una advertencia si la predicción del modelo no coincide con la etiqueta experta:       
    etiqueta_experta = etiquetas_expertas[st.session_state.epoch_index]
    st.write(f"Etiqueta experta del hipnograma original: **{etiqueta_experta}**")

    if etiqueta_experta != prediccion:
        st.warning("⚠️ El modelo no coincide con la etiqueta experta en esta época.")

# Agregamos botones para navegar entre las épocas con baja confianza, permitiendo al usuario explorar rápidamente estas épocas:
    col_prev, col_next = st.columns(2)

    with col_prev:
        if st.button("Época anterior", use_container_width=True, type="primary"):
            if st.session_state.epoch_index > 0:
                st.session_state.epoch_index -= 1
                st.rerun()

    with col_next:
        if st.button("Siguiente baja confianza", use_container_width=True, type="primary"):
            siguientes = [i for i in epocas_baja_confianza if i > st.session_state.epoch_index]
            if siguientes:
                st.session_state.epoch_index = siguientes[0]
                st.rerun()

# Agregamos una sección para mostrar la señal EEG simulada correspondiente a la época seleccionada:
    st.subheader("Señal EEG")

    tiempo = [i * 0.1 for i in range(300)]
    senal = [math.sin(t * 2) * 0.5 + random.uniform(-0.1, 0.1) for t in tiempo]

    fig_senal = go.Figure()
    fig_senal.add_trace(go.Scatter(x=tiempo, y=senal, mode="lines", line=dict(width=1)))
    fig_senal.update_layout(xaxis_title="Tiempo (s)", height=200, margin=dict(t=10, b=10))

    st.plotly_chart(fig_senal, use_container_width=True)

