#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════
#  AuraEcho+ — One-Command Setup Script
#  Usage: bash setup.sh
#  Tested on: Ubuntu 22.04, macOS 13+, WSL2 (Windows)
# ══════════════════════════════════════════════════════════════════════

set -e  # Exit immediately on any error

# ── Colour codes ──────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'  # No Colour

# ── Banner ────────────────────────────────────────────────────────────
echo -e "${BLUE}"
echo "  ╔══════════════════════════════════════════╗"
echo "  ║         AuraEcho+  Setup Script          ║"
echo "  ║   Cardiac AI Decision Support System     ║"
echo "  ╚══════════════════════════════════════════╝"
echo -e "${NC}"

# ── Helper functions ──────────────────────────────────────────────────
info()    { echo -e "${CYAN}[INFO]${NC}  $1"; }
success() { echo -e "${GREEN}[OK]${NC}    $1"; }
warning() { echo -e "${YELLOW}[WARN]${NC}  $1"; }
error()   { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }
step()    { echo -e "\n${BOLD}${BLUE}▶  Step $1: $2${NC}"; }

# ══════════════════════════════════════════════════════════════════════
# STEP 1 — Check Python version
# ══════════════════════════════════════════════════════════════════════
step "1" "Checking Python version"

if ! command -v python3 &>/dev/null; then
    error "Python 3 not found. Install Python 3.10+ from https://python.org"
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_MAJOR=$(python3 -c "import sys; print(sys.version_info.major)")
PYTHON_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")

info "Found Python ${PYTHON_VERSION}"

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    error "Python 3.10+ required. Current version: ${PYTHON_VERSION}"
fi

success "Python ${PYTHON_VERSION} — compatible"

# ══════════════════════════════════════════════════════════════════════
# STEP 2 — Create virtual environment
# ══════════════════════════════════════════════════════════════════════
step "2" "Creating Python virtual environment"

VENV_DIR=".venv"

if [ -d "$VENV_DIR" ]; then
    warning "Virtual environment already exists at .venv — skipping creation"
    info "To recreate: rm -rf .venv && bash setup.sh"
else
    python3 -m venv "$VENV_DIR"
    success "Virtual environment created at .venv"
fi

# Activate venv
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
success "Virtual environment activated"

# Upgrade pip silently
pip install --upgrade pip --quiet
info "pip upgraded to latest version"

# ══════════════════════════════════════════════════════════════════════
# STEP 3 — Install Python dependencies
# ══════════════════════════════════════════════════════════════════════
step "3" "Installing Python dependencies"

if [ ! -f "requirements.txt" ]; then
    error "requirements.txt not found. Are you in the project root directory?"
fi

info "Installing from requirements.txt (this may take 2-3 minutes)..."
pip install -r requirements.txt --quiet

success "All Python packages installed"

# Verify critical packages
CRITICAL_PACKAGES=("streamlit" "sklearn" "numpy" "pandas" "plotly" "groq" "requests")
for pkg in "${CRITICAL_PACKAGES[@]}"; do
    if python3 -c "import ${pkg//-/_}" 2>/dev/null; then
        success "  ✓ ${pkg}"
    else
        warning "  ✗ ${pkg} — may have installed under a different name"
    fi
done

# ══════════════════════════════════════════════════════════════════════
# STEP 4 — Create project directories
# ══════════════════════════════════════════════════════════════════════
step "4" "Creating required directories"

# FIXED: models/ not data/models/ (matches constants.py MODEL_SAVE_PATH)
DIRS=(
    "data"
    "database"
    "models"
    "assets"
    "logs"
)

for dir in "${DIRS[@]}"; do
    mkdir -p "$dir"
    success "  ✓ $dir"
done

# ══════════════════════════════════════════════════════════════════════
# STEP 5 — Set up .env file
# ══════════════════════════════════════════════════════════════════════
step "5" "Setting up environment configuration"

if [ -f ".env" ]; then
    warning ".env already exists — skipping creation (your existing keys are safe)"
else
    cat > .env << 'EOF'
# ══════════════════════════════════════════════════════════
# AuraEcho+ Environment Configuration
# ⚠️  NEVER commit this file to Git — it contains secrets!
# ══════════════════════════════════════════════════════════

# ── Online AI — Groq (Primary, ultra-fast) ──────────────
# Get free key at: https://console.groq.com
GROQ_API_KEY=your_groq_api_key_here

# ── Online AI — OpenAI (Fallback) ───────────────────────
# Get key at: https://platform.openai.com/api-keys
OPENAI_API_KEY=your_openai_api_key_here

# ── Firebase (Cloud Sync) ───────────────────────────────
# Option A: Path to service account JSON file
GOOGLE_APPLICATION_CREDENTIALS=database/firebase_credentials.json

# Option B: Or paste JSON directly (for containerized deployments)
# FIREBASE_SERVICE_ACCOUNT_JSON='{"type": "service_account", ...}'

# ── Admin Account (created on first run if no users exist)
# If ADMIN_PASSWORD is not set, a secure random password will be
# generated and displayed in the console on first run.
ADMIN_USERNAME=admin
ADMIN_PASSWORD=your_secure_admin_password

# ── App Configuration ────────────────────────────────────
APP_ENV=development
DEBUG=false
LOG_LEVEL=INFO
EOF
    success ".env file created — add your API keys before running"
fi

# ══════════════════════════════════════════════════════════════════════
# STEP 6 — Train ML model and fit scaler
# ══════════════════════════════════════════════════════════════════════
step "6" "Training cardiac risk model"

if [ ! -f "data/heart_data.csv" ]; then
    warning "data/heart_data.csv not found — model training skipped"
    warning "Add data/heart_data.csv and re-run setup, or model will train on first app run"
else
    info "Training Random Forest model on data/heart_data.csv..."
    
    # Run training in activated venv
    python3 -c "
import sys
sys.path.insert(0, '.')
from core.risk_model import train_model

try:
    model, metrics = train_model()
    print('  ✓ Model trained successfully')
    print('  Metrics:')
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f'    • {k}: {v:.4f}')
        else:
            print(f'    • {k}: {v}')
except Exception as e:
    print(f'  ✗ Training failed: {e}', file=sys.stderr)
    sys.exit(1)
" && success "Model saved to models/risk_model.pkl" \
  || warning "Model training failed — will train automatically on first app run"
fi

# Create sample input if missing
if [ ! -f "data/sample_input.json" ]; then
    info "Creating sample input file..."
    cat > data/sample_input.json << 'EOF'
{
    "name": "John Doe",
    "age": 58,
    "sex": 1,
    "cp": 2,
    "trestbps": 130,
    "chol": 240,
    "fbs": 0,
    "restecg": 1,
    "thalach": 140,
    "exang": 0,
    "oldpeak": 1.5,
    "slope": 1,
    "ca": 1,
    "thal": 2,
    "symptoms": "Occasional chest discomfort during exercise"
}
EOF
    success "Sample input created at data/sample_input.json"
fi

# ══════════════════════════════════════════════════════════════════════
# STEP 7 — Check Ollama (offline AI)
# ══════════════════════════════════════════════════════════════════════
step "7" "Checking Ollama (offline AI)"

if command -v ollama &>/dev/null; then
    success "Ollama is installed"
    
    # FIXED: Check if ollama service is running
    if curl -s http://localhost:11434/api/tags &>/dev/null; then
        success "Ollama service is running"
    else
        warning "Ollama service is not running"
        info "  Start Ollama with: ollama serve"
    fi
    
    # Check if llama3 model is pulled
    if ollama list 2>/dev/null | grep -q "llama3"; then
        success "llama3 model is available"
    else
        warning "llama3 model not pulled yet"
        echo -e "${YELLOW}  To enable offline AI, run:${NC}"
        echo "    ollama pull llama3"
        echo ""
        read -r -p "  Pull llama3 now? (takes ~4GB download) [y/N]: " pull_response
        if [[ "$pull_response" =~ ^[Yy]$ ]]; then
            info "Pulling llama3... (this will take a few minutes)"
            ollama pull llama3 && success "llama3 pulled successfully" \
                                || warning "Pull failed — run 'ollama pull llama3' manually"
        fi
    fi
else
    warning "Ollama not installed — offline AI will not be available"
    echo -e "${YELLOW}  Install Ollama from: https://ollama.ai${NC}"
    echo "  Then run: ollama pull llama3"
fi

# ══════════════════════════════════════════════════════════════════════
# STEP 8 — Check Firebase credentials
# ══════════════════════════════════════════════════════════════════════
step "8" "Checking Firebase configuration"

# FIXED: Check correct path and env var
if [ -f "database/firebase_credentials.json" ]; then
    success "Firebase credentials found at database/firebase_credentials.json"
elif [ -f "data/firebase_credentials.json" ]; then
    warning "Found credentials at data/firebase_credentials.json"
    info "  Consider moving to database/firebase_credentials.json"
else
    warning "Firebase credentials not found"
    info "  Cloud sync will be disabled until credentials are added"
    info "  Download from: Firebase Console → Project Settings → Service Accounts"
    info "  Save as: database/firebase_credentials.json"
    info "  Or set GOOGLE_APPLICATION_CREDENTIALS in .env"
fi

# ══════════════════════════════════════════════════════════════════════
# DONE
# ══════════════════════════════════════════════════════════════════════
echo ""
echo -e "${GREEN}${BOLD}"
echo "  ╔══════════════════════════════════════════╗"
echo "  ║         Setup Complete! ✅               ║"
echo "  ╚══════════════════════════════════════════╝"
echo -e "${NC}"

echo -e "${BOLD}Next steps:${NC}"
echo ""
echo "  1. Add your API keys to .env"
echo "     ${CYAN}nano .env${NC}"
echo ""
echo "  2. Activate the virtual environment"
echo "     ${CYAN}source .venv/bin/activate${NC}"
echo ""
echo "  3. Launch AuraEcho+"
echo "     ${CYAN}streamlit run app.py${NC}"
echo "     ${CYAN}  — or —${NC}"
echo "     ${CYAN}make run${NC}"
echo ""
echo "  4. Open in browser"
echo "     ${CYAN}http://localhost:8501${NC}"
echo ""
echo -e "${YELLOW}First-time login:${NC}"
echo "  On first run, a default admin account will be created."
echo "  Check the console for the generated password if ADMIN_PASSWORD is not set in .env"
echo ""
echo -e "${BOLD}${GREEN}Happy diagnosing! ❤️${NC}"