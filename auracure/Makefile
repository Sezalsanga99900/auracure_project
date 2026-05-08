# ══════════════════════════════════════════════════════════════════════
#  AuraEcho+ — Makefile
#  Cardiac AI Decision Support System
#
#  Usage:
#    make run      → Start the Streamlit app
#    make setup    → Full first-time setup
#    make test     → Run all unit tests
#    make clean    → Remove cache and temp files
#    make train    → Retrain the ML model
#    make help     → Show this help message
# ══════════════════════════════════════════════════════════════════════

# ── Configuration ─────────────────────────────────────────────────────
PYTHON      := python3
VENV        := .venv
VENV_PYTHON := $(VENV)/bin/python
PIP         := $(VENV)/bin/pip
STREAMLIT   := $(VENV)/bin/streamlit
PYTEST      := $(VENV)/bin/pytest
APP         := app.py
PORT        := 8501

# ── Detect OS for cross-platform compatibility ────────────────────────
UNAME := $(shell uname -s)
ifeq ($(UNAME), Windows_NT)
	VENV_PYTHON := $(VENV)/Scripts/python
	PIP         := $(VENV)/Scripts/pip
	STREAMLIT   := $(VENV)/Scripts/streamlit
	PYTEST      := $(VENV)/Scripts/pytest
endif

# ── Colour output ────────────────────────────────────────────────────
BOLD  := \033[1m
GREEN := \033[0;32m
CYAN  := \033[0;36m
YELLOW:= \033[1;33m
RED   := \033[0;31m
NC    := \033[0m

# ══════════════════════════════════════════════════════════════════════
# DEFAULT TARGET
# ══════════════════════════════════════════════════════════════════════

.DEFAULT_GOAL := help

.PHONY: help run setup install train test lint clean clean-db \
        check-env status ollama-pull freeze db-init db-stats \
        db-export db-backup demo-data clean-model clean-all \
        venv run-headless run-dark dev evaluate test-fast \
        test-single format ollama-start ollama-status

# ══════════════════════════════════════════════════════════════════════
# HELP
# ══════════════════════════════════════════════════════════════════════

help:  ## Show this help message
	@echo ""
	@echo "$(BOLD)AuraEcho+ — Available Commands$(NC)"
	@echo "$(CYAN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-18s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Quick start:$(NC)  make setup && make run"
	@echo ""

# ══════════════════════════════════════════════════════════════════════
# SETUP & INSTALLATION
# ══════════════════════════════════════════════════════════════════════

setup:  ## Full first-time project setup (venv + deps + model training)
	@echo "$(BOLD)$(GREEN)▶  Running full setup...$(NC)"
	@bash setup.sh
	@echo "$(GREEN)✅ Setup complete. Run 'make run' to start.$(NC)"

install:  ## Install Python dependencies only (assumes venv exists)
	@echo "$(CYAN)▶  Installing dependencies...$(NC)"
	@$(PIP) install --upgrade pip --quiet
	@$(PIP) install -r requirements.txt --quiet
	@echo "$(GREEN)✅ Dependencies installed$(NC)"

venv:  ## Create virtual environment
	@echo "$(CYAN)▶  Creating virtual environment...$(NC)"
	@$(PYTHON) -m venv $(VENV)
	@echo "$(GREEN)✅ Virtual environment created at $(VENV)$(NC)"
	@echo "   Activate with: source $(VENV)/bin/activate"

# ══════════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════════

run:  ## Launch AuraEcho+ Streamlit app (default port 8501)
	@echo "$(BOLD)$(GREEN)▶  Starting AuraEcho+...$(NC)"
	@echo "$(CYAN)   URL: http://localhost:$(PORT)$(NC)"
	@echo "$(YELLOW)   Press Ctrl+C to stop$(NC)"
	@echo ""
	@$(STREAMLIT) run $(APP) \
		--server.port=$(PORT) \
		--server.headless=false \
		--server.runOnSave=true \
		--theme.base=light \
		--theme.primaryColor="#1a73e8" \
		--theme.secondaryBackgroundColor="#f0f4f8"

run-headless:  ## Launch in headless mode (no browser auto-open)
	@$(STREAMLIT) run $(APP) \
		--server.port=$(PORT) \
		--server.headless=true \
		--server.address=0.0.0.0

run-dark:  ## Launch with dark theme
	@$(STREAMLIT) run $(APP) \
		--server.port=$(PORT) \
		--theme.base=dark \
		--theme.primaryColor="#2980b9"

dev:  ## Launch in dev mode (auto-reload on file save)
	@echo "$(CYAN)▶  Dev mode — watching for changes...$(NC)"
	@STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
		$(STREAMLIT) run $(APP) \
		--server.port=$(PORT) \
		--server.runOnSave=true

# ══════════════════════════════════════════════════════════════════════
# MACHINE LEARNING
# ══════════════════════════════════════════════════════════════════════

train:  ## Retrain the cardiac risk model from heart_data.csv
	@echo "$(CYAN)▶  Training Random Forest model...$(NC)"
	@$(VENV_PYTHON) -c "\
from core.risk_model import train_model; \
model, metrics = train_model(); \
print('  Accuracy:  {:.4f}'.format(metrics['accuracy'])); \
print('  F1 Score:  {:.4f}'.format(metrics['f1'])); \
print('  ROC-AUC:   {:.4f}'.format(metrics['roc_auc'])); \
print('  CV AUC:    {:.4f} ± {:.4f}'.format(metrics.get('cv_roc_auc_mean', 0), metrics.get('cv_roc_auc_std', 0))); \
print('  Train size:', metrics['train_size']); \
print('  Test size: ', metrics['test_size']); \
"
	@echo "$(GREEN)✅ Model trained and saved to models/$(NC)"

evaluate:  ## Print current model performance metrics
	@$(VENV_PYTHON) -c "\
from core.risk_model import load_model; \
import pandas as pd; \
from core.preprocess import preprocess_dataframe; \
from sklearn.metrics import classification_report; \
df = pd.read_csv('data/heart_data.csv'); \
X, y = preprocess_dataframe(df); \
model = load_model(); \
preds = model.predict(X); \
print(classification_report(y, preds, target_names=['No Disease','Disease'])); \
"

# ══════════════════════════════════════════════════════════════════════
# TESTING
# ══════════════════════════════════════════════════════════════════════

test:  ## Run all unit tests with coverage report
	@echo "$(CYAN)▶  Running tests...$(NC)"
	@$(PYTEST) tests/ \
		--verbose \
		--tb=short \
		--cov=. \
		--cov-report=term-missing \
		--cov-omit="$(VENV)/*,tests/*" \
		-q
	@echo "$(GREEN)✅ Tests complete$(NC)"

test-fast:  ## Run tests without coverage (faster)
	@$(PYTEST) tests/ -q --tb=short

test-single:  ## Run a single test file  (usage: make test-single FILE=tests/test_risk_model.py)
	@$(PYTEST) $(FILE) -v --tb=long

# Note: test-watch removed — requires pytest-watch plugin not in requirements

# ══════════════════════════════════════════════════════════════════════
# CODE QUALITY
# ══════════════════════════════════════════════════════════════════════

# Note: flake8 and black are optional dev dependencies
# Install with: pip install flake8 black

lint:  ## Check code style with flake8
	@echo "$(CYAN)▶  Running linter...$(NC)"
	@if [ -f "$(VENV)/bin/flake8" ]; then \
		$(VENV)/bin/flake8 . \
			--exclude=.venv,__pycache__,data,assets,models,database \
			--max-line-length=100 \
			--extend-ignore=E203,W503 \
		&& echo "$(GREEN)✅ No lint errors$(NC)" \
		|| echo "$(YELLOW)⚠️  Lint issues found$(NC)"; \
	else \
		echo "$(YELLOW)⚠️  flake8 not installed. Run: pip install flake8$(NC)"; \
	fi

format:  ## Auto-format code with black
	@echo "$(CYAN)▶  Formatting code with black...$(NC)"
	@if [ -f "$(VENV)/bin/black" ]; then \
		$(VENV)/bin/black . \
			--exclude=".venv|__pycache__|data|assets|models|database" \
			--line-length=100; \
		echo "$(GREEN)✅ Code formatted$(NC)"; \
	else \
		echo "$(YELLOW)⚠️  black not installed. Run: pip install black$(NC)"; \
	fi

# ══════════════════════════════════════════════════════════════════════
# OLLAMA
# ══════════════════════════════════════════════════════════════════════

ollama-start:  ## Start the Ollama service
	@echo "$(CYAN)▶  Starting Ollama service...$(NC)"
	@ollama serve &
	@sleep 2
	@echo "$(GREEN)✅ Ollama started$(NC)"

ollama-pull:  ## Pull the llama3 model (~4GB)
	@echo "$(CYAN)▶  Pulling llama3 model...$(NC)"
	@ollama pull llama3
	@echo "$(GREEN)✅ llama3 ready$(NC)"

ollama-status:  ## Check Ollama status and loaded models
	@ollama list 2>/dev/null || echo "$(RED)Ollama not running$(NC)"

# ══════════════════════════════════════════════════════════════════════
# DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════

status:  ## Show full system status (DB, model, API keys)
	@echo "$(BOLD)$(CYAN)AuraEcho+ System Status$(NC)"
	@echo "$(CYAN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@$(VENV_PYTHON) -c "\
import os; \
import sqlite3; \
from core.mode_detector import is_online, get_connection_info; \
from ai.offline_ai import is_ollama_available; \
from database.local_db import LOCAL_DB_PATH, DB_TABLE_PATIENTS, DB_TABLE_PREDICTIONS, DB_TABLE_SYNC_QUEUE; \
info = get_connection_info(force_refresh=True); \
print('🌐 Internet:  ', '✅ Online' if info.get('online') else '❌ Offline', \
      f\"({info.get('latency_ms', 0):.0f}ms)\" if info.get('online') else ''); \
print('🤖 Ollama:    ', '✅ Ready' if is_ollama_available() else '❌ Not running'); \
print('🔑 Groq Key:  ', '✅ Set' if os.getenv('GROQ_API_KEY') else '❌ Missing'); \
print('🔑 OpenAI:    ', '✅ Set' if os.getenv('OPENAI_API_KEY') else '❌ Missing'); \
print('🔑 Firebase:  ', '✅ Found' if os.path.exists('database/firebase_credentials.json') else '❌ Missing'); \
try: \
    conn = sqlite3.connect(LOCAL_DB_PATH); \
    cur = conn.cursor(); \
    cur.execute(f'SELECT COUNT(*) FROM {DB_TABLE_PATIENTS}'); \
    patients = cur.fetchone()[0]; \
    cur.execute(f'SELECT COUNT(*) FROM {DB_TABLE_PREDICTIONS}'); \
    predictions = cur.fetchone()[0]; \
    cur.execute(f'SELECT COUNT(*) FROM {DB_TABLE_SYNC_QUEUE} WHERE status=\"PENDING\"'); \
    pending = cur.fetchone()[0]; \
    conn.close(); \
    print('🗄️  Patients:  ', patients, 'records'); \
    print('📊 Assessments:', predictions, 'records'); \
    print('⏳ Pending Sync:', pending, 'items'); \
except Exception as e: \
    print('🗄️  Database:   ❌ Error:', e); \
"

check-env:  ## Verify all required environment variables
	@echo "$(CYAN)▶  Checking environment variables...$(NC)"
	@$(VENV_PYTHON) -c "\
import os; \
required = ['GROQ_API_KEY']; \
optional = ['OPENAI_API_KEY', 'GOOGLE_APPLICATION_CREDENTIALS', 'ADMIN_PASSWORD']; \
all_ok = True; \
for var in required: \
    val = os.getenv(var); \
    if val: print(f'  ✅ {var} = {val[:8]}...'); \
    else: print(f'  ❌ {var} = NOT SET (required)'); all_ok = False; \
for var in optional: \
    val = os.getenv(var); \
    status = '✅' if val else '⚠️ '; \
    print(f'  {status} {var} = {val[:20] if val else \"not set (optional)\"}'); \
print(); \
print('Status:', '✅ All required vars set' if all_ok else '❌ Missing required vars'); \
"

# ══════════════════════════════════════════════════════════════════════
# DATABASE
# ══════════════════════════════════════════════════════════════════════

db-init:  ## Initialise the local SQLite database
	@echo "$(CYAN)▶  Initialising database...$(NC)"
	@$(VENV_PYTHON) -c "from database.local_db import init_db; init_db(); print('Database ready')"
	@$(VENV_PYTHON) -c "from services.auth_service import init_auth_db; init_auth_db(); print('Auth DB ready')"
	@echo "$(GREEN)✅ Databases initialised$(NC)"

db-stats:  ## Show database statistics
	@$(VENV_PYTHON) -c "\
import sqlite3; \
from database.local_db import LOCAL_DB_PATH, DB_TABLE_PATIENTS, DB_TABLE_PREDICTIONS, DB_TABLE_SYNC_QUEUE; \
conn = sqlite3.connect(LOCAL_DB_PATH); \
cur = conn.cursor(); \
cur.execute(f'SELECT COUNT(*) FROM {DB_TABLE_PATIENTS}'); \
print('  patients:', cur.fetchone()[0]); \
cur.execute(f'SELECT COUNT(*) FROM {DB_TABLE_PREDICTIONS}'); \
print('  predictions:', cur.fetchone()[0]); \
cur.execute(f'SELECT status, COUNT(*) FROM {DB_TABLE_SYNC_QUEUE} GROUP BY status'); \
for status, count in cur.fetchall(): \
    print(f'  sync_queue.{status}: {count}'); \
conn.close(); \
"

db-export:  ## Export all patients to CSV
	@echo "$(CYAN)▶  Exporting to CSV...$(NC)"
	@$(VENV_PYTHON) -c "\
from database.local_db import export_to_csv; \
path = export_to_csv(); \
print(f'Exported to: {path}'); \
"

db-backup:  ## Create a backup of the database
	@echo "$(CYAN)▶  Creating database backup...$(NC)"
	@$(VENV_PYTHON) -c "\
import shutil; \
from database.local_db import LOCAL_DB_PATH; \
from utils.helpers import now_str; \
backup_path = LOCAL_DB_PATH.replace('.db', f'_backup_{now_str().replace(\" \", \"_\").replace(\":\", \"-\")}.db'); \
shutil.copy2(LOCAL_DB_PATH, backup_path); \
print(f'Backup created: {backup_path}'); \
"

# ══════════════════════════════════════════════════════════════════════
# DEPENDENCIES
# ══════════════════════════════════════════════════════════════════════

freeze:  ## Update requirements.txt with currently installed versions
	@echo "$(CYAN)▶  Freezing requirements...$(NC)"
	@$(PIP) freeze > requirements-frozen.txt
	@echo "$(GREEN)✅ Saved to requirements-frozen.txt$(NC)"
	@echo "$(YELLOW)   Review before overwriting requirements.txt$(NC)"

# ══════════════════════════════════════════════════════════════════════
# CLEANUP
# ══════════════════════════════════════════════════════════════════════

clean:  ## Remove Python cache, temp files, and build artifacts
	@echo "$(CYAN)▶  Cleaning project...$(NC)"
	@find . -type d -name __pycache__ \
		-not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" \
		-not -path "./.venv/*" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" \
		-not -path "./.venv/*" -delete 2>/dev/null || true
	@find . -type d -name "*.egg-info" \
		-not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name ".coverage" -delete 2>/dev/null || true
	@rm -f .coverage coverage.xml 2>/dev/null || true
	@echo "$(GREEN)✅ Cache cleaned$(NC)"

clean-db:  ## ⚠️  Delete local databases (IRREVERSIBLE)
	@echo "$(RED)$(BOLD)⚠️  WARNING: This will delete ALL local patient data!$(NC)"
	@read -r -p "  Type 'DELETE' to confirm: " confirm; \
	if [ "$$confirm" = "DELETE" ]; then \
		rm -f database/auraecho.db database/auth.db; \
		echo "$(GREEN)Databases deleted$(NC)"; \
	else \
		echo "$(CYAN)Cancelled$(NC)"; \
	fi

clean-model:  ## Delete trained model files (forces retrain on next run)
	@echo "$(CYAN)▶  Deleting model files...$(NC)"
	@rm -f models/risk_model.pkl models/scaler.pkl
	@echo "$(GREEN)✅ Model files deleted — will retrain on next run$(NC)"

clean-all: clean clean-db clean-model  ## ⚠️  Full reset (cache + DB + models)
	@echo "$(GREEN)✅ Full clean complete$(NC)"

# ══════════════════════════════════════════════════════════════════════
# DEMO
# ══════════════════════════════════════════════════════════════════════

demo-data:  ## Load sample patient data for demo
	@echo "$(CYAN)▶  Loading demo patient data...$(NC)"
	@$(VENV_PYTHON) -c "\
import json; \
from database.local_db import save_patient, save_prediction; \
from core.risk_model import predict_risk; \
with open('data/sample_input.json') as f: \
    patient = json.load(f); \
patient_id = save_patient(patient); \
result = predict_risk(patient); \
save_prediction(patient_id, result.to_dict()); \
print(f'Demo patient loaded: ID={patient_id}'); \
print(f'Risk level: {result.risk_level} ({result.confidence_pct:.1f}%)'); \
"
	@echo "$(GREEN)✅ Demo data loaded$(NC)"