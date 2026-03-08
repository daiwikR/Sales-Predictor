#!/bin/sh
# Dynamic entrypoint: selects dashboard or API service.
set -e

case "${APP_TARGET}" in
  api)
    echo "[INFO] Starting FastAPI prediction server on :8000"
    exec uvicorn src.api:app --host 0.0.0.0 --port 8000 --log-level info
    ;;
  dashboard|*)
    echo "[INFO] Starting Streamlit dashboard on :8501"
    exec streamlit run dashboard/app.py \
        --server.port=8501 \
        --server.address=0.0.0.0 \
        --server.headless=true \
        --browser.gatherUsageStats=false
    ;;
esac
