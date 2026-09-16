import streamlit as st
import plotly.graph_objects as go

import model_interface as model

# Configuración de la página de Streamlit:
st.set_page_config(page_title="Sleep Staging Assistant", layout="wide")
#-------------------------------------------------------------------------------------------------------------------------
# Agregamos estilos CSS personalizados para mejorar la apariencia de la aplicación:
st.markdown("""
<style>
    /* Tarjetas de estudio (Pantalla 1) */
    /* Estudios seleccionables: el radio nativo de Streamlit se estiliza
       para verse como una tarjeta completa; ya no dibujamos tarjetas aparte */
    div[data-testid="stRadio"],
    div[data-testid="stRadio"] > div,
    div[data-testid="stRadio"] label,
    div[role="radiogroup"] {
        width: 100% !important;
        display: block !important;
    }
    div[role="radiogroup"] label {
        border: 1.5px solid #E2E5E9 !important;
        border-radius: 10px !important;
        padding: 18px 20px !important;
        margin-bottom: 10px !important;
        cursor: pointer;
        font-size: 1rem !important;
        color: #6B7280 !important;
        line-height: 1.6;
        box-sizing: border-box !important;
    }
    div[role="radiogroup"] label::first-line {
        color: #111827 !important;
    }
    div[data-testid="stElementContainer"]:has(div[data-testid="stRadio"]) {
        width: 100% !important;
    }    
    div[role="radiogroup"] label:has(input:checked) {
        border-color: #2563EB !important;
        background-color: #EFF6FF !important;
    }

    /* Pastillas de canal (verde = activo) */
    .channel-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 0.8rem;
        margin-right: 6px;
        background-color: #DCFCE7;
        color: #166534;
        border: 1px solid #86EFAC;
    }
    .channel-pill.inactive {
        background-color: #F3F4F6;
        color: #9CA3AF;
        border: 1px solid #E5E7EB;
    }

    /* Tarjetas KPI (Pantalla 2) */
    .kpi-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .kpi-card.warning {
        background-color: #FEF3C7;
        border-color: #FCD34D;
    }
    .kpi-label { font-size: 0.8rem; color: #64748B; margin-bottom: 4px; }
    .kpi-value { font-size: 1.6rem; font-weight: 700; color: #0F172A; }

    /* Tarjetas de probabilidad por clase */
    .prob-card {
        border-radius: 10px;
        padding: 10px;
        text-align: center;
        background-color: #F1F5F9;
        border: 1px solid #E2E8F0;
    }
    .prob-card.top {
        background-color: #DBEAFE;
        border-color: #2563EB;
    }
    .prob-card .clase { font-weight: 600; color: #334155; font-size: 0.85rem; }
    .prob-card .valor { font-size: 1.3rem; font-weight: 700; color: #0F172A; }

    /* Botones primarios (Streamlit ya los pinta de azul con type="primary"),
       aqui solo redondeamos un poco mas las esquinas para que combine */
    div.stButton > button[kind="primary"] {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

LOW_CONFIDENCE_THRESHOLD = 0.65

#-------------------------------------------------------------------------------------------------------------------
# Estados de la sesión:
#-------------------------------------------------------------------------------------------------------------------
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
#-------------------------------------------------------------------------------------------------------------------
# Pantalla 1: Selección del estudio y configuración de señales
#-------------------------------------------------------------------------------------------------------------------
def render_screen_select():
        st.title("🌙 Sleep staging assistant")
        st.caption("Seleccione un estudio y configure las señales a utilizar")

        # Agregamos la lista de estudios disponibles para seleccionar
        st.subheader("1. Seleccionar estudio")  # Encabezado de la sección
        estudios = ["SC4001E0", "SC4012E0", "ST7011J0"]
        metas = {
            "SC4001E0": "Sleep Cassette · 20h 04m",
            "SC4012E0": "Sleep Cassette · 19h 40m",
            "ST7011J0": "Sleep Telemetry · 9h 12m",
        }

        # Creamos los botones para seleccionar el estudio (solo se puede seleccionar uno a la vez):
        opciones_estudio = [f"**{estudio}**  \n{metas[estudio]}" for estudio in estudios]
        opcion_actual = f"**{st.session_state.selected_study}**  \n{metas[st.session_state.selected_study]}"

        estudio_seleccionado = st.radio(
             "Estudios disponibles", opciones_estudio, 
             index=opciones_estudio.index(opcion_actual),
             label_visibility="collapsed",
        )
        estudio_seleccionado = estudio_seleccionado.split("\n")[0].replace("**", "").strip()

        # Sección de señales a utilizar:
        st.subheader("2. Señales disponibles en este estudio")

        # Agregamos checkboxes para seleccionar las señales a utilizar:
        # La señal EEG Fpz-Cz es obligatoria y está preseleccionada y deshabilitada para que no se pueda desmarcar.
        eeg_fpz_cz = st.checkbox("EEG Fpz-Cz · obligatorio · 100 Hz", value=True, disabled=True)
        eeg_pz_oz = st.checkbox("EEG Pz-Oz · opcional · 100 Hz", value=True)
        eog = st.checkbox("EOG horizontal · opcional · 100 Hz", value=False)
        emg = st.checkbox("EMG submentoniano · opcional · 1 Hz", value=False)

        # Renderizamos las pastillas de canal con el estado activo/inactivo correspondiente:
        canales = {"EEG Fpz-Cz": eeg_fpz_cz, "EEG Pz-Oz": eeg_pz_oz,
                "EOG horizontal": eog, "EMG submentoniano": emg}
        pastillas = "".join(
            f'<span class="channel-pill{"" if activo else " inactive"}">{nombre}</span>'
            for nombre, activo in canales.items()
        )
        st.markdown(pastillas, unsafe_allow_html=True)
        st.write("")  # espaciado

        # Agregamos un mensaje informativo sobre la importancia de las señales:
        st.info(
        "Más señales pueden mejorar la clasificación, especialmente en épocas "
        "de transición, pero el modelo funciona con EEG únicamente.",icon="ℹ️" 
        )

        # Guardamos el estudio seleccionado en la variable de estado y cambiamos a la pantalla 2 de análisis: 
        if st.button("Analizar estudio", type="primary", use_container_width=True):
                canales_activos = [nombre for nombre, activo in canales.items() if activo]
                with st.spinner("Cargando estudio y calculando predicciones..."):
                    dataset = model.cargar_estudio(estudio_seleccionado, canales_activos)
                    try:
                        dataset, resultados = model.predecir_estudio(dataset, modelo="avanzado")
                    except RuntimeError as error:
                        st.error(str(error))
                        st.stop()
                    st.session_state.dataset = dataset
                    st.session_state.resultados = resultados
                st.session_state.selected_study = estudio_seleccionado
                st.session_state.selected_channels = canales
                st.session_state.epoch_index = 0
                st.session_state.screen = "analysis"
                st.rerun()

#-------------------------------------------------------------------------------------------------------------------
# Pantalla 2: Análisis del estudio seleccionado
#-------------------------------------------------------------------------------------------------------------------
def render_kpis(total_epocas, duracion_horas, eficiencia_sueno, n_baja_confianza):
    kpis = [
        ("Duración total", f"{int(duracion_horas)}h {int((duracion_horas % 1) * 60):02d}m", False),
        ("Épocas totales", f"{total_epocas}", False),
        ("Eficiencia de sueño", f"{eficiencia_sueno}%", False),
        ("Baja confianza", f"{n_baja_confianza} épocas", True),
    ]
    cols = st.columns(4)
    for col, (label, value, es_advertencia) in zip(cols, kpis):
        clase = "kpi-card warning" if es_advertencia else "kpi-card"
        with col:
            st.markdown(
                f"""<div class="{clase}">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value">{value}</div>
                </div>""",
                unsafe_allow_html=True,
            )

# Muestra la distribución de estadios en porcentajes, con un diseño de tarjetas para cada estadio:
def render_distribucion_estadios(etapas):
    orden_estadios = ["W", "N1", "N2", "N3", "REM"]
    total = len(etapas)
    porcentajes = {
        estadio: round(100 * etapas.count(estadio) / total)
        for estadio in orden_estadios
    }

    st.markdown(
        '<div class="kpi-label" style="margin-bottom: 8px;">Distribución de estadios</div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(5)
    for col, estadio in zip(cols, orden_estadios):
        with col:
            st.markdown(
                f"""<div class="kpi-card">
                    <div class="kpi-label">{estadio}</div>
                    <div class="kpi-value">{porcentajes[estadio]}%</div>
                </div>""",
                unsafe_allow_html=True,
            )

def render_screen_analysis():
    study_id = st.session_state.selected_study
    canales_activos = [nombre for nombre, activo in st.session_state.selected_channels.items() if activo]

    # Agregamos un encabezado con el título del estudio y un botón para cambiar de estudio:
    header_cols = st.columns([0.85, 0.15])
    with header_cols[0]:
        st.title(f"📊 Estudio {study_id}")
        st.caption("Canales activos: " + ", ".join(canales_activos))
    with header_cols[1]:
        if st.button("Cambiar estudio", use_container_width=True):
            st.session_state.screen = "select"
            st.rerun()

    # Mostramos los indicadores (KPIs) totales de épocas, la duración total del sueño y el porcentaje de sueño REM en la pantalla de análisis:
    etapas = [r["prediccion"] for r in st.session_state.resultados]
    # Confianza del modelo para cada época (provisional hasta conectar la API real):
    confianza = [r["confianza"] for r in st.session_state.resultados]
    # Etiqueta experta real, ya extraída del hipnograma original por segmentation.py:
    etiquetas_expertas = [r["etiqueta_experta"] for r in st.session_state.resultados]

    total_epocas = len(etapas)
    duracion_horas = (total_epocas * 30) / 3600
    # Eficiencia de sueño = proporción de épocas dormidas (todo lo que no es W) sobre el
    # total del registro. Asume que el registro ya está recortado al periodo de sueño
    # (ver recorte de vigilia del pipeline); con un registro sin recortar subestimaría
    # la eficiencia real.
    epocas_dormido = total_epocas - etapas.count("W")
    eficiencia_sueno = round(100 * epocas_dormido / total_epocas)
    epocas_baja_confianza = [i for i, c in enumerate(confianza) if c < LOW_CONFIDENCE_THRESHOLD]

    render_kpis(total_epocas, duracion_horas, eficiencia_sueno, len(epocas_baja_confianza))
    st.write("")
    render_distribucion_estadios(etapas)
    st.write("")

    #-------------------------------------------------------------------------------------------------------------------
    # Hipnograma: Interactivo con Plotly para mostrar las etapas del sueño a lo largo del tiempo
    #-------------------------------------------------------------------------------------------------------------------
    st.subheader("Hipnograma")

    # Definimos el orden vertical de las etapas del sueño y calculamos las posiciones en el eje y para cada etapa:
    orden_vertical = ["N3", "N2", "N1", "W", "REM"]
    posiciones_y = [orden_vertical.index(e) for e in etapas]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(total_epocas)),
        y=posiciones_y,
        mode="lines",line_shape="hv",
        line=dict(color="#2563EB", width=2), #azul = selección de una acción
    ))

    # Agregamos rectángulos verticales para resaltar las épocas con baja confianza en la predicción del modelo:
    for i in epocas_baja_confianza:
        fig.add_vrect(x0=i - 0.5, x1=i + 0.5, fillcolor="#EF4444", opacity=0.15, line_width=0)
    fig.add_vline(x=st.session_state.epoch_index, line_dash="dot", line_color="#0F172A")
    fig.update_layout(
        yaxis=dict(tickmode="array", tickvals=list(range(len(orden_vertical))), ticktext=orden_vertical),
        xaxis_title="Época", height=280,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="white",           
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("🟥 época de baja confianza · línea punteada = época seleccionada")

    st.divider()

    #-------------------------------------------------------------------------------------------------------------------
    # Explorar época: Control deslizante para seleccionar una época y mostrar la etapa del sueño correspondiente
    #-------------------------------------------------------------------------------------------------------------------
    st.subheader("Explorar época")

    st.session_state.epoch_index = st.slider(
        "Selecciona una época", min_value=0, max_value=total_epocas - 1,
        value=st.session_state.epoch_index,
    )

    # Mostramos la etapa del sueño predicha y la confianza del modelo para la época seleccionada:
    resultado = model.obtener_prediccion(st.session_state.resultados, st.session_state.epoch_index)
    prediccion = resultado["prediccion"]
    conf = resultado["confianza"]
    probabilidades = resultado["probabilidades"]
    etiqueta_experta = resultado["etiqueta_experta"]

    # Mostramos la predicción del modelo y la confianza en un formato legible para el usuario:
    st.markdown(
        f"**Época {st.session_state.epoch_index}** · predicha **{prediccion}**"
        f" · confianza **{int(conf * 100)}%**"
    ) 

    # Mostramos las probabilidades de cada clase en tarjetas separadas, resaltando la clase con la mayor probabilidad:
    st.caption("Probabilidad que el modelo asigna a cada estadio para esta época — suman 100%")
    prob_cols = st.columns(5)
    for col, clase in zip(prob_cols, probabilidades):
        valor = probabilidades[clase]
        es_top = clase == prediccion
        css = "prob-card top" if es_top else "prob-card"
        with col:
            st.markdown(
                f"""<div class="{css}">
                    <div class="clase">{clase}</div>
                    <div class="valor">{int(valor * 100)}%</div>
                </div>""",
                unsafe_allow_html=True,
            )

    st.write("")
    st.caption(f"Etiqueta experta del hipnograma original: **{etiqueta_experta}**")
    if etiqueta_experta != prediccion:
        st.warning("⚠️ El modelo no coincide con la etiqueta experta en esta época.")

    # Agregamos botones para navegar entre las épocas con baja confianza, permitiendo al usuario explorar rápidamente estas épocas:
    col_prev, col_next = st.columns(2)

    with col_prev:
        if st.button("Época anterior", use_container_width=True, disabled=st.session_state.epoch_index == 0):
            if st.session_state.epoch_index > 0:
                st.session_state.epoch_index -= 1
                st.rerun()
    with col_next:
        if st.button("Siguiente baja confianza", use_container_width=True, type="primary"):
            siguientes = [i for i in epocas_baja_confianza if i > st.session_state.epoch_index]
            if siguientes:
                st.session_state.epoch_index = siguientes[0]
                st.rerun()
            else:
                st.info("No hay más épocas de baja confianza después de la actual.")

    #-------------------------------------------------------------------------------------------------------------------
    # Señales fisiológicas: Mostramos señales simuladas para las épocas seleccionadas
    #-------------------------------------------------------------------------------------------------------------------
    st.subheader("Señales fisiológicas")

    # Señal real de la época seleccionada, ya cargada en session_state (ver Cambio 2):
    epoca_actual = st.session_state.dataset.X[st.session_state.epoch_index]  # (n_canales, 3000)
    sfreq = st.session_state.dataset.sfreq
    tiempo = [i / sfreq for i in range(epoca_actual.shape[1])]

    colores = {
        "EEG Fpz-Cz": "#16A34A",
        "EEG Pz-Oz": "#E50FEC",
        "EOG horizontal": "#7C3AED",
        "EMG submentoniano": "#F97316",
    }
    canales_activos_lista = st.session_state.dataset.channel_names

    # Las mostramos de a 2 por fila, para que no queden ni muy angostas ni
    # una debajo de otra ocupando toda la pantalla
    for i in range(0, len(canales_activos_lista), 2):
        fila = canales_activos_lista[i:i + 2]
        cols = st.columns(len(fila))
        for col, nombre_canal in zip(cols, fila):
            indice_canal = canales_activos_lista.index(nombre_canal)
            senal = epoca_actual[indice_canal]
            color = colores.get(nombre_canal, "#334155")
            fig_senal = go.Figure(go.Scatter(x=tiempo, y=senal, mode="lines", line=dict(width=1.2, color=color)))
            fig_senal.update_layout(
                title=nombre_canal, xaxis_title="Tiempo (s)", yaxis_title="Amplitud (µV)",
                height=200, margin=dict(t=30, b=10), plot_bgcolor="white",
            )
            with col:
                st.plotly_chart(fig_senal, use_container_width=True)

#-------------------------------------------------------------------------------------------------------------------
# Renderizamos la pantalla correspondiente según el estado actual de la sesión:
#-------------------------------------------------------------------------------------------------------------------
if st.session_state.screen == "select":
    render_screen_select()
else:
    render_screen_analysis()

