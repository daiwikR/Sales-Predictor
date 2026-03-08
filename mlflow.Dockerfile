# ============================================================================
# MLflow Tracking Server Dockerfile
# Runs a standalone MLflow server backed by SQLite + local artifact storage.
# ============================================================================
FROM python:3.11-slim

RUN pip install --no-cache-dir mlflow==2.13.0 boto3

RUN mkdir -p /mlflow/artifacts

WORKDIR /mlflow

EXPOSE 5000

CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "sqlite:////mlflow/mlflow.db", \
     "--default-artifact-root", "/mlflow/artifacts", \
     "--serve-artifacts"]
