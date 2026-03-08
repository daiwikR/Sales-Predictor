# ============================================================================
# Superstore MLOps Forecast — Makefile
# ============================================================================
PYTHON      := python
PIP         := pip
PORT_DASH   := 8501
PORT_API    := 8000
PORT_MLFLOW := 5000

.PHONY: help install prepare train quick-train serve dashboard mlflow-ui \
        docker-up docker-down docker-logs clean lint

# ─── Default target ─────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  Superstore Sales Analytics & Forecasting Platform"
	@echo "  ─────────────────────────────────────────────────"
	@echo "  install       Install Python dependencies from requirements.txt"
	@echo "  prepare       Run ETL pipeline (generates clean.parquet + processed.parquet)"
	@echo "  train         Full training: Optuna (50 trials) + ensemble + 30-day forecast"
	@echo "  quick-train   Quick training: Optuna (10 trials) for rapid iteration"
	@echo "  serve         Start FastAPI prediction server on :$(PORT_API)"
	@echo "  dashboard     Launch Streamlit dashboard on :$(PORT_DASH)"
	@echo "  mlflow-ui     Open MLflow tracking UI on :$(PORT_MLFLOW)"
	@echo "  docker-up     Build and start all services via Docker Compose"
	@echo "  docker-down   Stop all Docker services"
	@echo "  docker-logs   Tail logs for all Docker services"
	@echo "  clean         Remove generated artefacts (models, parquets)"
	@echo "  lint          Run basic Python syntax check"
	@echo ""

# ─── Installation ───────────────────────────────────────────────────────────
install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "[OK] Dependencies installed."

# ─── Data preparation ───────────────────────────────────────────────────────
prepare:
	@echo "[INFO] Running ETL pipeline..."
	$(PYTHON) src/data_prep.py
	@echo "[OK] Data prepared: data/clean.parquet  data/processed.parquet"

# ─── Model training ─────────────────────────────────────────────────────────
train: prepare
	@echo "[INFO] Training ensemble model (50 Optuna trials)..."
	$(PYTHON) src/train.py --n-trials 50 --cv-splits 5
	@echo "[OK] Training complete. Artefacts: models/*.pkl  data/forecast.parquet"

quick-train: prepare
	@echo "[INFO] Quick training (10 Optuna trials)..."
	$(PYTHON) src/train.py --n-trials 10 --cv-splits 3
	@echo "[OK] Quick training complete."

# ─── Services ───────────────────────────────────────────────────────────────
serve:
	@echo "[INFO] Starting FastAPI server on :$(PORT_API)..."
	uvicorn src.api:app --host 0.0.0.0 --port $(PORT_API) --reload

dashboard:
	@echo "[INFO] Launching Streamlit dashboard on :$(PORT_DASH)..."
	streamlit run dashboard/app.py \
		--server.port $(PORT_DASH) \
		--server.address 0.0.0.0 \
		--server.headless true \
		--browser.gatherUsageStats false

mlflow-ui:
	@echo "[INFO] Starting MLflow UI on :$(PORT_MLFLOW)..."
	mlflow ui \
		--backend-store-uri sqlite:///mlruns.db \
		--port $(PORT_MLFLOW) \
		--host 0.0.0.0

# ─── Docker ─────────────────────────────────────────────────────────────────
docker-up:
	@echo "[INFO] Building and starting Docker services..."
	docker-compose up --build -d
	@echo "[OK] Services started."
	@echo "  Dashboard : http://localhost:$(PORT_DASH)"
	@echo "  API       : http://localhost:$(PORT_API)"
	@echo "  MLflow    : http://localhost:$(PORT_MLFLOW)"

docker-down:
	docker-compose down --volumes
	@echo "[OK] All services stopped."

docker-logs:
	docker-compose logs -f --tail=100

# ─── Cleanup ────────────────────────────────────────────────────────────────
clean:
	@echo "[INFO] Removing generated artefacts..."
	rm -f models/*.pkl models/*.json
	rm -f data/clean.parquet data/processed.parquet data/forecast.parquet
	rm -f mlruns.db
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "[OK] Clean complete."

lint:
	$(PYTHON) -m py_compile src/data_prep.py src/train.py src/predict.py src/api.py dashboard/app.py
	@echo "[OK] No syntax errors found."
