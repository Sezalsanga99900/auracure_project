# =============================================================================
# utils/constants.py
# AuraEcho+ — Central Constants Registry
# All thresholds, labels, model configs, and app-wide settings live here.
# =============================================================================

# -----------------------------------------------------------------------------
# APP METADATA
# -----------------------------------------------------------------------------
APP_NAME = "AuraEcho+"
APP_VERSION = "1.0.0"
APP_TAGLINE = "AI-Powered Cardiac Risk Assessment"
AUTHOR = "AuraEcho Team"

# -----------------------------------------------------------------------------
# RISK LEVEL LABELS & THRESHOLDS
# -----------------------------------------------------------------------------
RISK_LEVELS = {
    "LOW":    (0.0,  0.35),
    "MEDIUM": (0.35, 0.65),
    "HIGH":   (0.65, 1.0),    # FIXED: was 1.01
}

RISK_LABELS = {
    "LOW":    "Low Risk",
    "MEDIUM": "Medium Risk",
    "HIGH":   "High Risk",
}

RISK_COLORS = {
    "LOW":    "#2ecc71",
    "MEDIUM": "#f39c12",
    "HIGH":   "#e74c3c",
}

RISK_ICONS = {
    "LOW":    "✅",
    "MEDIUM": "⚠️",
    "HIGH":   "🚨",
}

RISK_DESCRIPTIONS = {
    "LOW": (
        "The patient's profile suggests a low probability of significant cardiac "
        "disease. Routine monitoring and lifestyle maintenance are recommended."
    ),
    "MEDIUM": (
        "The patient shows moderate cardiac risk indicators. Further diagnostic "
        "evaluation and lifestyle modifications are advised."
    ),
    "HIGH": (
        "The patient presents high-risk cardiac indicators. Immediate clinical "
        "attention and comprehensive cardiac workup are strongly recommended."
    ),
}

# -----------------------------------------------------------------------------
# FEATURE COLUMNS (must match heart_data.csv header exactly)
# -----------------------------------------------------------------------------
FEATURE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

TARGET_COLUMN = "target"

CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
NUMERICAL_FEATURES   = ["age", "trestbps", "chol", "thalach", "oldpeak"]

FEATURE_LABELS = {
    "age":      "Age (years)",
    "sex":      "Sex (1 = Male, 0 = Female)",
    "cp":       "Chest Pain Type",
    "trestbps": "Resting Blood Pressure (mm Hg)",
    "chol":     "Serum Cholesterol (mg/dl)",
    "fbs":      "Fasting Blood Sugar > 120 mg/dl",
    "restecg":  "Resting ECG Results",
    "thalach":  "Max Heart Rate Achieved",
    "exang":    "Exercise-Induced Angina",
    "oldpeak":  "ST Depression (Oldpeak)",
    "slope":    "Slope of Peak Exercise ST",
    "ca":       "Major Vessels (Fluoroscopy)",
    "thal":     "Thalassemia Type",
}

# Numerical feature value ranges
FEATURE_RANGES = {
    "age":      (1,   120),
    "trestbps": (60,  250),
    "chol":     (50,  700),
    "thalach":  (50,  250),
    "oldpeak":  (0.0, 10.0),
}

# Categorical feature valid values
FEATURE_VALID_VALUES = {
    "sex":     [0, 1],
    "cp":      [0, 1, 2, 3],
    "fbs":     [0, 1],
    "restecg": [0, 1, 2],
    "exang":   [0, 1],
    "slope":   [0, 1, 2],
    "ca":      [0, 1, 2, 3],
    "thal":    [1, 2, 3],
}

# -----------------------------------------------------------------------------
# CATEGORICAL ENCODINGS  →  core/preprocess.py
# ADDED: was missing — caused ImportError in preprocess.py
# -----------------------------------------------------------------------------
CATEGORICAL_ENCODINGS = {
    "sex": {
        "female": 0, "male": 1,
        "Female": 0, "Male": 1,
        "0": 0, "1": 1,
    },
    "cp": {
        "typical angina":    0,
        "atypical angina":   1,
        "non-anginal pain":  2,
        "asymptomatic":      3,
        "Typical Angina":    0,
        "Atypical Angina":   1,
        "Non-Anginal Pain":  2,
        "Asymptomatic":      3,
    },
    "fbs": {
        "no": 0,  "yes": 1,
        "No": 0,  "Yes": 1,
        "false": 0, "true": 1,
        "False": 0, "True": 1,
    },
    "restecg": {
        "normal":                        0,
        "st-t wave abnormality":         1,
        "left ventricular hypertrophy":  2,
        "Normal":                        0,
        "ST-T Wave Abnormality":         1,
        "Left Ventricular Hypertrophy":  2,
    },
    "exang": {
        "no": 0,  "yes": 1,
        "No": 0,  "Yes": 1,
    },
    "slope": {
        "upsloping":   0,
        "flat":        1,
        "downsloping": 2,
        "Upsloping":   0,
        "Flat":        1,
        "Downsloping": 2,
    },
    "ca": {
        "0": 0, "1": 1, "2": 2, "3": 3,
    },
    "thal": {
        "normal":            1,
        "fixed defect":      2,
        "reversible defect": 3,
        "Normal":            1,
        "Fixed Defect":      2,
        "Reversible Defect": 3,
    },
}

# -----------------------------------------------------------------------------
# CHEST PAIN / CATEGORICAL DECODE LABELS  →  ui/ + helpers.py
# -----------------------------------------------------------------------------
CHEST_PAIN_LABELS = {
    0: "Typical Angina",
    1: "Atypical Angina",
    2: "Non-Anginal Pain",
    3: "Asymptomatic",
}

THAL_LABELS = {
    1: "Normal",
    2: "Fixed Defect",
    3: "Reversible Defect",
}

SLOPE_LABELS = {
    0: "Upsloping",
    1: "Flat",
    2: "Downsloping",
}

RESTECG_LABELS = {
    0: "Normal",
    1: "ST-T Wave Abnormality",
    2: "Left Ventricular Hypertrophy",
}

# -----------------------------------------------------------------------------
# SIMILARITY ENGINE (KNN)  →  core/similarity.py
# -----------------------------------------------------------------------------
KNN_N_NEIGHBORS      = 5
KNN_TOP_DISPLAY      = 3
KNN_METRIC           = "cosine"
KNN_ALGORITHM        = "brute"
SIMILARITY_SCORE_MIN = 0.0
SIMILARITY_SCORE_MAX = 100.0
SIMILARITY_POOL_SIZE = 50      # ADDED: pool size for filtered similarity search

# -----------------------------------------------------------------------------
# RISK MODEL WEIGHTS  →  core/risk_model.py
# -----------------------------------------------------------------------------
RISK_WEIGHTS = {
    "age":      0.10,
    "sex":      0.05,
    "cp":       0.15,
    "trestbps": 0.08,
    "chol":     0.10,
    "fbs":      0.05,
    "restecg":  0.07,
    "thalach":  0.10,
    "exang":    0.08,
    "oldpeak":  0.10,
    "slope":    0.05,
    "ca":       0.05,
    "thal":     0.02,
}

# FIXED: Derived from FEATURE_RANGES — no duplication
RISK_BASELINES = {
    feat: {"min": rng[0], "max": rng[1]}
    for feat, rng in FEATURE_RANGES.items()
}

# -----------------------------------------------------------------------------
# RANDOM FOREST HYPERPARAMETERS  →  core/risk_model.py
# -----------------------------------------------------------------------------
RF_N_ESTIMATORS  = 200
RF_MAX_DEPTH     = 12
RF_RANDOM_STATE  = 42

# -----------------------------------------------------------------------------
# EXPLANATION THRESHOLDS  →  core/risk_model.py
# -----------------------------------------------------------------------------
CONFIDENCE_LOW_THRESHOLD  = 60.0
CONFIDENCE_HIGH_THRESHOLD = 85.0

# -----------------------------------------------------------------------------
# AI / LLM MODEL NAMES
# -----------------------------------------------------------------------------

# ── Offline (Ollama) ──────────────────────────────────────────────────────────
OFFLINE_MODEL_NAME    = "llama3"
OFFLINE_MODEL_TIMEOUT = 60
OFFLINE_BASE_URL      = "http://localhost:11434"

# ADDED: Ollama aliases — needed by ai/offline_ai.py
OLLAMA_BASE_URL    = OFFLINE_BASE_URL
OLLAMA_MODEL       = OFFLINE_MODEL_NAME
OLLAMA_TIMEOUT     = OFFLINE_MODEL_TIMEOUT

# ── Online (Groq primary) ─────────────────────────────────────────────────────
ONLINE_PROVIDER    = "groq"
ONLINE_MODEL_NAME  = "llama3-70b-8192"
ONLINE_MODEL_TIMEOUT = 30
GROQ_API_BASE_URL  = "https://api.groq.com/openai/v1"

# ADDED: Groq aliases — needed by ai/online_ai.py
GROQ_MODEL         = ONLINE_MODEL_NAME
GROQ_TIMEOUT       = ONLINE_MODEL_TIMEOUT

# ── Online fallback (OpenAI) ──────────────────────────────────────────────────
FALLBACK_PROVIDER   = "openai"
FALLBACK_MODEL_NAME = "gpt-3.5-turbo"

# ADDED: OpenAI aliases — needed by ai/online_ai.py
OPENAI_MODEL        = FALLBACK_MODEL_NAME
OPENAI_TIMEOUT      = ONLINE_MODEL_TIMEOUT

# ── Shared LLM generation settings ───────────────────────────────────────────
LLM_MAX_TOKENS   = 1024
LLM_TEMPERATURE  = 0.3
LLM_TOP_P        = 0.9

# ADDED: Per-provider token/temp aliases — needed by ai/online_ai.py + offline_ai.py
OLLAMA_MAX_TOKENS   = LLM_MAX_TOKENS
OLLAMA_TEMPERATURE  = LLM_TEMPERATURE
GROQ_MAX_TOKENS     = LLM_MAX_TOKENS
GROQ_TEMPERATURE    = LLM_TEMPERATURE
OPENAI_MAX_TOKENS   = LLM_MAX_TOKENS
OPENAI_TEMPERATURE  = LLM_TEMPERATURE

# ADDED: Shared API timeout — needed by ai/online_ai.py
API_TIMEOUT = ONLINE_MODEL_TIMEOUT

# -----------------------------------------------------------------------------
# PROMPT SETTINGS  →  ai/prompt_builder.py
# -----------------------------------------------------------------------------
PROMPT_SYSTEM_ROLE = (
    "You are a board-certified cardiologist AI assistant. "
    "Provide concise, evidence-based cardiac risk assessments. "
    "Always remind the physician that AI output is supportive, not diagnostic."
)

PROMPT_MAX_SIMILAR_CASES   = 5
PROMPT_INCLUDE_DISCLAIMER  = True

# -----------------------------------------------------------------------------
# DATABASE  →  database/local_db.py, database/cloud_db.py
# -----------------------------------------------------------------------------
LOCAL_DB_PATH         = "database/auraecho.db"
LOCAL_CSV_BACKUP_PATH = "database/records_backup.csv"
DB_TABLE_PATIENTS     = "patients"
DB_TABLE_PREDICTIONS  = "predictions"
DB_TABLE_SYNC_QUEUE   = "sync_queue"

FIREBASE_COLLECTION   = "auraecho_records"
MONGO_DB_NAME         = "auraecho"
MONGO_COLLECTION      = "patient_records"

SYNC_BATCH_SIZE       = 50

# -----------------------------------------------------------------------------
# NETWORK / CONNECTIVITY  →  core/mode_detector.py
# -----------------------------------------------------------------------------

# ADDED: TCP host tuples — mode_detector.py uses socket probing not HTTP
CONNECTIVITY_CHECK_HOSTS = [
    ("8.8.8.8",        53),   # Google DNS
    ("1.1.1.1",        53),   # Cloudflare DNS
    ("208.67.222.222", 53),   # OpenDNS
    ("api.groq.com",   443),  # Groq API
]

# Kept for reference (HTTP-based, not used by mode_detector)
CONNECTIVITY_CHECK_URLS = [
    "https://www.google.com",
    "https://api.groq.com",
    "https://cloudflare.com",
]

CONNECTIVITY_TIMEOUT   = 3     # seconds per probe
CONNECTIVITY_RETRIES   = 2     # retries per host
CONNECTIVITY_CACHE_TTL = 30    # ADDED: seconds before re-probing

MODE_ONLINE  = "online"
MODE_OFFLINE = "offline"

# ADDED: UI mode labels — needed by core/mode_detector.py
MODE_ONLINE_LABEL  = "🟢 Online"
MODE_OFFLINE_LABEL = "🔴 Offline"

# -----------------------------------------------------------------------------
# AUTHENTICATION  →  services/auth_service.py
# -----------------------------------------------------------------------------
SESSION_TTL_HOURS      = 8
SESSION_EXPIRY_MINUTES = SESSION_TTL_HOURS * 60   # FIXED: derived, no duplicate

JWT_ALGORITHM        = "HS256"
PASSWORD_MIN_LENGTH  = 8
MAX_LOGIN_ATTEMPTS   = 5
LOCKOUT_DURATION_MIN = 15

AUTH_DB_PATH = "database/auth.db"

ROLES = {
    "DOCTOR": "doctor",
    "NURSE":  "nurse",
    "ADMIN":  "admin",
    "VIEWER": "viewer",
}

# Role value aliases
ROLE_DOCTOR = ROLES["DOCTOR"]
ROLE_NURSE  = ROLES["NURSE"]
ROLE_ADMIN  = ROLES["ADMIN"]

# -----------------------------------------------------------------------------
# ROLE PERMISSIONS MAP  →  services/auth_service.py + ui/role_dashboard.py
# -----------------------------------------------------------------------------
ROLE_PERMISSIONS = {
    "doctor": [
        "view_dashboard",
        "view_diagnosis",
        "edit_patient",
        "view_analytics",
        "manage_patients",
        "view_ai_insights",
        "export_reports",
    ],
    "nurse": [
        "view_dashboard",
        "view_diagnosis",
        "view_patient",
        "enter_vitals",
    ],
    "admin": [
        "view_dashboard",
        "manage_users",
        "view_analytics",
        "system_settings",
        "export_reports",
    ],
    "viewer": [
        "view_dashboard",
        "view_patient",
    ],
}

# -----------------------------------------------------------------------------
# DATA FILE PATHS
# -----------------------------------------------------------------------------
HEART_DATA_PATH    = "data/heart_data.csv"
SAMPLE_INPUT_PATH  = "data/sample_input.json"
ASSETS_CSS_PATH    = "assets/styles.css"
ASSETS_LOGO_PATH   = "assets/logo.png"

# -----------------------------------------------------------------------------
# MODEL SAVE PATHS  →  core/risk_model.py
# -----------------------------------------------------------------------------
MODEL_SAVE_PATH  = "models/risk_model.pkl"
SCALER_SAVE_PATH = "models/scaler.pkl"

# -----------------------------------------------------------------------------
# UI THEME COLORS  →  ui/ modules
# -----------------------------------------------------------------------------
UI_PRIMARY_COLOR    = "#1a73e8"
UI_SECONDARY_COLOR  = "#ffffff"
UI_BACKGROUND_COLOR = "#f0f4f8"
UI_CARD_COLOR       = "#ffffff"
UI_TEXT_PRIMARY     = "#1a1a2e"
UI_TEXT_SECONDARY   = "#5f6368"
UI_BORDER_COLOR     = "#e8f0fe"
UI_SUCCESS_COLOR    = "#2ecc71"
UI_WARNING_COLOR    = "#f39c12"
UI_DANGER_COLOR     = "#e74c3c"
UI_SHADOW           = "0 2px 8px rgba(0,0,0,0.08)"

# -----------------------------------------------------------------------------
# UI / STREAMLIT  →  ui/ modules
# -----------------------------------------------------------------------------
PAGE_TITLE     = "AuraEcho+ | Cardiac AI"
PAGE_ICON      = "🫀"
PAGE_LAYOUT    = "wide"
SIDEBAR_STATE  = "expanded"

CHART_THEME       = "plotly_dark"
CHART_FONT_FAMILY = "Inter, sans-serif"
CHART_BG_COLOR    = "#0f1117"
CHART_GRID_COLOR  = "#2a2a3e"

RESULTS_TOP_N_CASES = 5

# -----------------------------------------------------------------------------
# SYNC SERVICE  →  services/sync_service.py
# -----------------------------------------------------------------------------
SYNC_INTERVAL_SECONDS = 300
MAX_SYNC_RETRIES      = 3

# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------
LOG_LEVEL  = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
LOG_FILE   = "auraecho.log"

# -----------------------------------------------------------------------------
# MISCELLANEOUS
# -----------------------------------------------------------------------------
DECIMAL_PRECISION = 4
DATE_FORMAT       = "%Y-%m-%d %H:%M:%S"
UNKNOWN_LABEL     = "Unknown"
NA_PLACEHOLDER    = "N/A"