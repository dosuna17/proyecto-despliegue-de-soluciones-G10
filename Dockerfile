# Imagen compartida para ejecutar la API y el tablero del proyecto.
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY app/api/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir torch==2.14.0+cpu \
        --index-url https://download.pytorch.org/whl/cpu

COPY app ./app
COPY src ./src
COPY configs ./configs
EXPOSE 8000

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
