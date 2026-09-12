import os

from fastapi import FastAPI

app = FastAPI(
    title="Sleep Stage Classification API",
    version="0.1.0",
)


MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
    }
