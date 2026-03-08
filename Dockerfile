# ============================================================================
# Superstore MLOps Forecast — Application Dockerfile
# Serves both the Streamlit dashboard and FastAPI prediction endpoint.
# Build argument TARGET selects the entrypoint: dashboard | api
# ============================================================================
FROM python:3.11-slim AS base

LABEL maintainer="superstore-platform@corp.com"
LABEL description="Superstore Sales Analytics & Forecasting Platform"
LABEL version="1.0"

# System-level build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- Python dependencies (cached layer) ----
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ---- Application source ----
COPY src/          ./src/
COPY dashboard/    ./dashboard/
COPY models/       ./models/
COPY data/         ./data/

# Runtime directories must exist
RUN mkdir -p models data

# ---- Build argument for target service ----
ARG TARGET=dashboard
ENV APP_TARGET=${TARGET}

EXPOSE 8501 8000

# Entrypoint selects service based on APP_TARGET
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

ENTRYPOINT ["/docker-entrypoint.sh"]
