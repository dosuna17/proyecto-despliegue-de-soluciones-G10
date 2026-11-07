# Imagen del tablero. Se construye desde la raíz del repositorio:
#   docker build -f app/dashboard/Dockerfile -t sleep-staging-dashboard .
#
# Es independiente de la imagen de la API: no incluye PyTorch, MLflow ni el
# dataset, por eso pesa mucho menos y arranca en segundos. Todo lo que el
# tablero necesita para hablar con la API llega por variables de entorno.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY app/dashboard/requirements.txt app/dashboard/requirements.txt
RUN pip install -r app/dashboard/requirements.txt

COPY configs/app.yaml configs/app.yaml
COPY app/dashboard app/dashboard

# Streamlit lee .streamlit/config.toml de la carpeta desde donde se ejecuta.
WORKDIR /app/app/dashboard

# Usuario sin privilegios: el contenedor no necesita ser root.
RUN useradd --create-home --uid 1000 tablero
USER tablero

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4)"

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
