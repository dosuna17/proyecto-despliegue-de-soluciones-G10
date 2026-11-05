# Tablero de Sleep Staging Assistant.
#
# Estructura del archivo (en este orden):
#   1. Configuración de página y estilos
#   2. Constantes y estado de la sesión
#   3. Pantalla 1: selección de estudio y canales
#   4. Pantalla 2: helpers de KPIs y distribución de estadios
#   5. Pantalla 2: análisis completo del estudio (incluye selector de modelo)
#   6. Enrutamiento entre pantallas

import streamlit as st
import plotly.graph_objects as go

import model_interface as model

#-------------------------------------------------------------------------------------------------------------------
# 1. Configuración de página y estilos
#-------------------------------------------------------------------------------------------------------------------
st.set_page_config(page_title="Sleep Staging Assistant", layout="wide")

st.markdown("""
<style>
    /* Colores base para modo claro (por defecto) */
    :root {
        --card-border: #E2E5E9;
        --card-text: #6B7280;
        --card-text-strong: #111827;
        --selected-bg: rgba(37, 99, 235, 0.10);
        --selected-border: #2563EB;
        --surface-bg: #F8FAFC;
        --surface-border: #E2E8F0;
        --warning-bg: #FEF3C7;
        --warning-border: #FCD34D;
        --pill-active-bg: #DCFCE7;
        --pill-active-text: #166534;
        --pill-active-border: #86EFAC;
        --pill-inactive-bg: #F3F4F6;
        --pill-inactive-text: #9CA3AF;
        --pill-inactive-border: #E5E7EB;
    }

    /* Mismos tokens, redefinidos para modo oscuro -- se activa solo según
       la preferencia de tema del navegador/sistema del usuario, sin
       necesidad de detectarlo manualmente */
    @media (prefers-color-scheme: dark) {
        :root {
            --card-border: #3A3F4B;
            --card-text: #B4B9C4;
            --card-text-strong: #F3F4F6;
            --selected-bg: rgba(96, 165, 250, 0.18);
            --selected-border: #60A5FA;
            --surface-bg: #1E2130;
            --surface-border: #3A3F4B;
            --warning-bg: #3F2E0F;
            --warning-border: #B45309;
            --pill-active-bg: #14321F;
            --pill-active-text: #86EFAC;
            --pill-active-border: #22C55E;
            --pill-inactive-bg: #2A2E3A;
            --pill-inactive-text: #9CA3AF;
            --pill-inactive-border: #3A3F4B;
        }
    }

    /* Tarjetas de estudio (Pantalla 1) y de modelo (Pantalla 2) */
    /* El radio nativo de Streamlit se estiliza para verse como una tarjeta
       completa; ya no dibujamos tarjetas aparte */
    div[data-testid="stRadio"],
    div[data-testid="stRadio"] > div,
    div[data-testid="stRadio"] label,
    div[role="radiogroup"] {
        width: 100% !important;
        display: block !important;
    }
    div[role="radiogroup"] label {
        border: 1.5px solid var(--card-border) !important;
        border-radius: 10px !important;
        padding: 18px 20px !important;
        margin-bottom: 10px !important;
        cursor: pointer;
        font-size: 1rem !important;
        color: var(--card-text) !important;
        line-height: 1.6;
        box-sizing: border-box !important;
    }
    /* Fuerza el color en cualquier elemento anidado que Streamlit genere
       dentro del label (p, strong, span del markdown) -- sin esto, algunos
       navegadores dejan el texto con el color por defecto del tema, que en
       modo oscuro puede quedar invisible sobre el fondo de la tarjeta */
    div[role="radiogroup"] label * {
        color: inherit !important;
    }
    div[role="radiogroup"] label::first-line {
        color: var(--card-text-strong) !important;
    }
    div[data-testid="stElementContainer"]:has(div[data-testid="stRadio"]) {
        width: 100% !important;
    }    
    div[role="radiogroup"] label:has(input:checked) {
        border-color: var(--selected-border) !important;
        background-color: var(--selected-bg) !important;
    }

    /* Pastillas de canal */
    .channel-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 0.8rem;
        margin-right: 6px;
        background-color: var(--pill-active-bg);
        color: var(--pill-active-text);
        border: 1px solid var(--pill-active-border);
    }
    .channel-pill.inactive {
        background-color: var(--pill-inactive-bg);
        color: var(--pill-inactive-text);
        border: 1px solid var(--pill-inactive-border);
    }

    /* Tarjetas KPI (Pantalla 2) */
    .kpi-card {
        background-color: var(--surface-bg);
        border: 1px solid var(--surface-border);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .kpi-card.warning {
        background-color: var(--warning-bg);
        border-color: var(--warning-border);
    }
    .kpi-label { font-size: 0.8rem; color: var(--card-text); margin-bottom: 4px; }
    .kpi-value { font-size: 1.6rem; font-weight: 700; color: var(--card-text-strong); }

    /* Tarjetas de probabilidad por clase */
    .prob-card {
        border-radius: 10px;
        padding: 10px;
        text-align: center;
        background-color: var(--surface-bg);
        border: 1px solid var(--surface-border);
    }
    .prob-card.top {
        background-color: var(--selected-bg);
        border-color: var(--selected-border);
    }
    .prob-card .clase { font-weight: 600; color: var(--card-text); font-size: 0.85rem; }
    .prob-card .valor { font-size: 1.3rem; font-weight: 700; color: var(--card-text-strong); }

    /* Botones primarios (Streamlit ya los pinta de azul con type="primary"),
       aqui solo redondeamos un poco mas las esquinas para que combine */
    div.stButton > button[kind="primary"] {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

#-------------------------------------------------------------------------------------------------------------------
# 2. Constantes y estado de la sesión
#-------------------------------------------------------------------------------------------------------------------
LOW_CONFIDENCE_THRESHOLD = 0.65

# Estudios de ejemplo disponibles en la Pantalla 1 (metadata solo informativa,
# no afecta la carga real, eso lo hace model.cargar_estudio):
ESTUDIOS_DISPONIBLES = ["SC4001E0", "SC4012E0", "ST7011J0"]
METADATA_ESTUDIOS = {
    "SC4001E0": "Sleep Cassette · 20h 04m",
    "SC4012E0": "Sleep Cassette · 19h 40m",
    "ST7011J0": "Sleep Telemetry · 9h 12m",
}

if "screen" not in st.session_state:
    st.session_state.screen = "select"
if "selected_study" not in st.session_state:
    st.session_state.selected_study = "SC4001E0"
if "epoch_index" not in st.session_state:
    st.session_state.epoch_index = 0
if "selected_channels" not in st.session_state:
    st.session_state.selected_channels = {}

#-------------------------------------------------------------------------------------------------------------------
# 3. Pantalla 1: selección de estudio y canales
#-------------------------------------------------------------------------------------------------------------------
# Dibuja el selector de estudio (tarjetas de radio) y devuelve el estudio elegido.
def _elegir_estudio():
    st.subheader("1. Seleccionar estudio")

    opciones_estudio = [f"**{estudio}**  \n{METADATA_ESTUDIOS[estudio]}" for estudio in ESTUDIOS_DISPONIBLES]
    opcion_actual = f"**{st.session_state.selected_study}**  \n{METADATA_ESTUDIOS[st.session_state.selected_study]}"

    estudio_elegido = st.radio(
        "Estudios disponibles", opciones_estudio,
        index=opciones_estudio.index(opcion_actual),
        label_visibility="collapsed",
    )
    return estudio_elegido.split("\n")[0].replace("**", "").strip()


# Dibuja los checkboxes de canales y devuelve la lista de canales activos.
# EEG Fpz-Cz siempre está activo y no se puede desmarcar (entrada mínima del
# proyecto). Un canal opcional solo se puede activar si al menos un modelo
# disponible lo necesita (ver model.canales_soportados()). Si ningún
# modelo usa esa señal todavía, el checkbox aparece deshabilitado, no
# oculto, para que quede claro que existe pero aún no está en uso.
#
# Esto SOLO decide qué señales se cargan del estudio, la elección de qué
# modelo usar para predecir vive en la Pantalla 2.
def _elegir_canales():
    st.subheader("2. Señales disponibles en este estudio")

    soportados = model.canales_soportados()

    eeg_fpz_cz = st.checkbox("EEG Fpz-Cz · obligatorio · 100 Hz", value=True, disabled=True)

    pz_oz_habilitado = "EEG Pz-Oz" in soportados
    eeg_pz_oz = st.checkbox(
        "EEG Pz-Oz · opcional · 100 Hz" + ("" if pz_oz_habilitado else " · sin modelo aún"),
        value=pz_oz_habilitado, disabled=not pz_oz_habilitado,
    )

    eog_habilitado = "EOG horizontal" in soportados
    eog = st.checkbox(
        "EOG horizontal · opcional · 100 Hz" + ("" if eog_habilitado else " · sin modelo aún"),
        value=eog_habilitado, disabled=not eog_habilitado,
    )

    canales = {"EEG Fpz-Cz": eeg_fpz_cz, "EEG Pz-Oz": eeg_pz_oz,
               "EOG horizontal": eog}
    canales_activos = [nombre for nombre, activo in canales.items() if activo]

    pastillas = "".join(
        f'<span class="channel-pill{"" if activo else " inactive"}">{nombre}</span>'
        for nombre, activo in canales.items()
    )
    st.markdown(pastillas, unsafe_allow_html=True)
    st.write("")

    st.info(
        "Solo se pueden activar las señales para las que ya existe un modelo "
        "entrenado. Nuevas señales se habilitarán aquí automáticamente en "
        "cuanto el equipo las entrene y registre.", icon="ℹ️"
    )
    return canales_activos


# Carga el estudio real con los canales elegidos y pasa a la Pantalla 2.
# La predicción NO se calcula aquí, se calcula en la Pantalla 2, según
# el modelo que el usuario elija ahí (ver _elegir_modelo_y_predecir).
#
# Cualquier error al cargar el estudio se atrapa aquí: el detalle técnico
# se imprime en consola para el equipo, y a pantalla solo llega un
# mensaje en lenguaje llano, nunca una ruta de archivo ni un traceback.
def _cargar_y_avanzar(estudio_id, canales_activos):
    with st.spinner("Cargando estudio..."):
        try:
            dataset = model.cargar_estudio(estudio_id, canales_activos)
        except Exception as error:
            print(f"[app.py] Error cargando el estudio {estudio_id}: {error}")
            st.error(
                "No fue posible cargar el estudio seleccionado. "
                "Contacte al equipo técnico si el problema persiste."
            )
            st.stop()

        st.session_state.dataset = dataset
        st.session_state.resultados_por_modelo = {}  # cache, se llena en Pantalla 2

    st.session_state.selected_study = estudio_id
    st.session_state.selected_channels = {nombre: True for nombre in canales_activos}
    st.session_state.epoch_index = 0
    st.session_state.screen = "analysis"
    st.rerun()

# Pantalla 1 completa: título, selector de estudio, selector de canales,
# y botón para cargar.
def render_screen_select():
    st.title("🌙 Sleep staging assistant")
    st.caption("Seleccione un estudio y configure las señales a utilizar")

    estudio_elegido = _elegir_estudio()
    canales_activos = _elegir_canales()

    if st.button("Analizar estudio", type="primary", use_container_width=True):
        _cargar_y_avanzar(estudio_elegido, canales_activos)

#-------------------------------------------------------------------------------------------------------------------
# 4. Pantalla 2: helpers de KPIs y distribución de estadios
#-------------------------------------------------------------------------------------------------------------------
# Dibuja los 4 KPIs principales del estudio en tarjetas.
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


# Dibuja el % de tiempo del estudio en cada uno de los 5 estadios.
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

#-------------------------------------------------------------------------------------------------------------------
# 5. Pantalla 2: análisis completo del estudio
#-------------------------------------------------------------------------------------------------------------------
# Encabezado con el título del estudio, los canales cargados, y el botón
# para volver a la Pantalla 1.
def _render_encabezado(study_id, canales_cargados):
    header_cols = st.columns([0.85, 0.15])
    with header_cols[0]:
        st.title(f"📊 Estudio {study_id}")
        st.caption("Canales cargados: " + ", ".join(canales_cargados))
    with header_cols[1]:
        if st.button("Cambiar estudio", use_container_width=True):
            st.session_state.screen = "select"
            st.rerun()


# Selector de modelo: muestra solo los modelos que se pueden calcular con
# los canales cargados, calcula sus predicciones la primera vez (y las
# cachea en session_state para no repetir la llamada a la API cada vez que
# el usuario cambia de época), y devuelve (dataset_usado, resultados) del
# modelo elegido. XGBoost es el modelo por defecto.
def _elegir_modelo_y_predecir(canales_cargados):
    st.subheader("Modelo")

    modelos = model.modelos_disponibles(canales_cargados)
    if not modelos:
        st.error(
            "Ningún modelo entrenado puede usarse con los canales cargados "
            f"({', '.join(canales_cargados)}). Vuelve a la Pantalla 1 y activa "
            "más señales."
        )
        st.stop()

    opciones = [m["etiqueta"] for m in modelos]
    etiqueta_elegida = st.radio("Ver resultados de:", opciones, horizontal=True)
    modelo_info = next(m for m in modelos if m["etiqueta"] == etiqueta_elegida)
    clave_modelo = modelo_info["modelo"]

    if clave_modelo not in st.session_state.resultados_por_modelo:
        with st.spinner(f"Calculando predicciones con {modelo_info['etiqueta']}..."):
            if clave_modelo == "avanzado":
                subconjunto = model.preparar_entrada_tinysleepnet(st.session_state.dataset)
            else:
                subconjunto = model.subset_canales(
                    st.session_state.dataset,
                    modelo_info["canales_requeridos"],
                )
            try:
                dataset_usado, resultados = model.predecir_estudio(subconjunto, modelo=clave_modelo)
            except RuntimeError as error:
                st.error(str(error))
                st.stop()
            st.session_state.resultados_por_modelo[clave_modelo] = (dataset_usado, resultados)

    return st.session_state.resultados_por_modelo[clave_modelo]


# Hipnograma interactivo: una línea escalonada por época, con las épocas de
# baja confianza resaltadas y una línea punteada marcando la época seleccionada.
def _render_hipnograma(etapas, total_epocas, epocas_baja_confianza):
    st.subheader("Hipnograma")

    orden_vertical = ["N3", "N2", "N1", "W", "REM"]
    posiciones_y = [orden_vertical.index(e) for e in etapas]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(total_epocas)),
        y=posiciones_y,
        mode="lines", line_shape="hv",
        line=dict(color="#2563EB", width=2),  # azul = selección de una acción
    ))

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


# Panel de "Explorar época": deslizador + predicción, probabilidades por
# clase, etiqueta experta, y navegación entre épocas de baja confianza.
def _render_explorar_epoca(resultados, total_epocas, epocas_baja_confianza):
    st.subheader("Explorar época")

    st.session_state.epoch_index = st.slider(
        "Selecciona una época", min_value=0, max_value=total_epocas - 1,
        value=st.session_state.epoch_index,
    )

    resultado = model.obtener_prediccion(resultados, st.session_state.epoch_index)
    prediccion = resultado["prediccion"]
    conf = resultado["confianza"]
    probabilidades = resultado["probabilidades"]
    etiqueta_experta = resultado["etiqueta_experta"]

    st.markdown(
        f"**Época {st.session_state.epoch_index}** · predicha **{prediccion}**"
        f" · confianza **{int(conf * 100)}%**"
    )

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


# Grafica la señal real del dataset_usado (puede tener menos canales que lo
# cargado en Pantalla 1, si el modelo elegido usa solo un subconjunto) para
# la época seleccionada, de a 2 canales por fila.
def _render_senales_fisiologicas(dataset_usado):
    st.subheader("Señales fisiológicas")

    epoca_actual = dataset_usado.X[st.session_state.epoch_index]  # (n_canales, 3000)
    sfreq = dataset_usado.sfreq
    tiempo = [i / sfreq for i in range(epoca_actual.shape[1])]

    colores = {
        "EEG Fpz-Cz": "#16A34A",
        "EEG Pz-Oz": "#E50FEC",
        "EOG horizontal": "#7C3AED",
        "EMG submentoniano": "#F97316",
    }
    canales_lista = dataset_usado.channel_names

    for i in range(0, len(canales_lista), 2):
        fila = canales_lista[i:i + 2]
        cols = st.columns(len(fila))
        for col, nombre_canal in zip(cols, fila):
            indice_canal = canales_lista.index(nombre_canal)
            senal = epoca_actual[indice_canal]
            color = colores.get(nombre_canal, "#334155")
            fig_senal = go.Figure(go.Scatter(x=tiempo, y=senal, mode="lines", line=dict(width=1.2, color=color)))
            fig_senal.update_layout(
                title=nombre_canal, xaxis_title="Tiempo (s)", yaxis_title="Amplitud (µV)",
                height=200, margin=dict(t=30, b=10), plot_bgcolor="white",
            )
            with col:
                st.plotly_chart(fig_senal, use_container_width=True)


# Pantalla 2 completa: encabezado, selector de modelo, KPIs, distribución,
# hipnograma, explorar época y señales fisiológicas en ese orden.
def render_screen_analysis():
    study_id = st.session_state.selected_study
    canales_cargados = [nombre for nombre, activo in st.session_state.selected_channels.items() if activo]
    _render_encabezado(study_id, canales_cargados)

    dataset_usado, resultados = _elegir_modelo_y_predecir(canales_cargados)

    etapas = [r["prediccion"] for r in resultados]
    confianza = [r["confianza"] for r in resultados]

    total_epocas = len(etapas)
    duracion_horas = (total_epocas * 30) / 3600
    # Eficiencia de sueño = proporción de épocas dormidas (todo lo que no es
    # W) sobre el total del registro. Asume que el registro ya está
    # recortado al periodo de sueño (ver recorte de vigilia del pipeline);
    # con un registro sin recortar subestimaría la eficiencia real.
    epocas_dormido = total_epocas - etapas.count("W")
    eficiencia_sueno = round(100 * epocas_dormido / total_epocas)
    epocas_baja_confianza = [i for i, c in enumerate(confianza) if c < LOW_CONFIDENCE_THRESHOLD]

    render_kpis(total_epocas, duracion_horas, eficiencia_sueno, len(epocas_baja_confianza))
    st.write("")
    render_distribucion_estadios(etapas)
    st.write("")

    _render_hipnograma(etapas, total_epocas, epocas_baja_confianza)
    st.divider()
    _render_explorar_epoca(resultados, total_epocas, epocas_baja_confianza)
    _render_senales_fisiologicas(dataset_usado)

#-------------------------------------------------------------------------------------------------------------------
# 6. Enrutamiento entre pantallas
#-------------------------------------------------------------------------------------------------------------------
if st.session_state.screen == "select":
    render_screen_select()
else:
    render_screen_analysis()
