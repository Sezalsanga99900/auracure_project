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
# Risk score is a float in [0.0, 1.0] produced by risk_model.py
RISK_LEVELS = {
    "LOW":    (0.0,  0.35),   # score < 0.35
    "MEDIUM": (0.35, 0.65),   # 0.35 <= score < 0.65
    "HIGH":   (0.65, 1.01),   # score >= 0.65
}

RISK_LABELS = {
    "LOW":    "Low Risk",
    "MEDIUM": "Medium Risk",
    "HIGH":   "High Risk",
}

RISK_COLORS = {
    "LOW":    "#2ecc71",   # green
    "MEDIUM": "#f39c12",   # amber
    "HIGH":   "#e74c3c",   # red
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
    "cp",           # chest pain type (0–3)
    "trestbps",     # resting blood pressure (mm Hg)
    "chol",         # serum cholesterol (mg/dl)
    "fbs",          # fasting blood sugar > 120 mg/dl (1 = true)
    "restecg",      # resting ECG results (0–2)
    "thalach",      # maximum heart rate achieved
    "exang",        # exercise-induced angina (1 = yes)
    "oldpeak",      # ST depression induced by exercise
    "slope",        # slope of peak exercise ST segment (0–2)
    "ca",           # number of major vessels coloured by fluoroscopy (0–3)
    "thal",         # thalassemia (1 = normal; 2 = fixed defect; 3 = reversible)
]

TARGET_COLUMN = "target"   # 1 = disease present, 0 = absent

CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
NUMERICAL_FEATURES   = ["age", "trestbps", "chol", "thalach", "oldpeak"]

# Human-readable feature labels for UI display
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

# Valid value ranges for each numerical feature (used by validators.py)
FEATURE_RANGES = {
    "age":      (1,   120),
    "trestbps": (60,  250),
    "chol":     (50,  700),
    "thalach":  (50,  250),
    "oldpeak":  (0.0, 10.0),
}

# Valid discrete values for categorical features
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
# CHEST PAIN TYPE LABELS
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
# SIMILARITY ENGINE (KNN) SETTINGS  →  core/similarity.py
# -----------------------------------------------------------------------------
KNN_N_NEIGHBORS        = 5          # neighbours to retrieve
KNN_TOP_DISPLAY        = 3          # top matches shown in UI
KNN_METRIC             = "cosine"   # distance metric: "cosine" | "euclidean"
KNN_ALGORITHM          = "brute"    # sklearn KNN algorithm
SIMILARITY_SCORE_MIN   = 0.0        # similarity normalised to [0, 1]
SIMILARITY_SCORE_MAX   = 1.0

# -----------------------------------------------------------------------------
# RISK MODEL WEIGHTS  →  core/risk_model.py
# Feature importance weights used in the hand-crafted risk scorer.
# Weights sum to 1.0 for interpretability.
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

# Normalisation baselines used when computing risk contribution per feature
RISK_BASELINES = {
    "age":      {"min": 1,   "max": 120},
    "trestbps": {"min": 60,  "max": 250},
    "chol":     {"min": 50,  "max": 700},
    "thalach":  {"min": 50,  "max": 250},
    "oldpeak":  {"min": 0.0, "max": 10.0},
}

# -----------------------------------------------------------------------------
# AI / LLM MODEL NAMES  →  ai/offline_ai.py, ai/online_ai.py
# -----------------------------------------------------------------------------

# Offline (Ollama local inference)
OFFLINE_MODEL_NAME    = "llama3"
OFFLINE_MODEL_TIMEOUT = 60          # seconds before timeout
OFFLINE_BASE_URL      = "http://localhost:11434"

# Online (Groq — fast cloud inference)
ONLINE_PROVIDER       = "groq"
ONLINE_MODEL_NAME     = "llama3-70b-8192"
ONLINE_MODEL_TIMEOUT  = 30
GROQ_API_BASE_URL     = "https://api.groq.com/openai/v1"

# Fallback online provider if Groq is unavailable
FALLBACK_PROVIDER     = "openai"
FALLBACK_MODEL_NAME   = "gpt-3.5-turbo"

# Shared LLM generation settings
LLM_MAX_TOKENS        = 1024
LLM_TEMPERATURE       = 0.3        # lower = more deterministic / clinical
LLM_TOP_P             = 0.9

# -----------------------------------------------------------------------------
# PROMPT SETTINGS  →  ai/prompt_builder.py
# -----------------------------------------------------------------------------
PROMPT_SYSTEM_ROLE = (
    "You are a board-certified cardiologist AI assistant. "
    "Provide concise, evidence-based cardiac risk assessments. "
    "Always remind the physician that AI output is supportive, not diagnostic."
)

PROMPT_MAX_SIMILAR_CASES = 5  # how many similar cases to embed in the prompt
PROMPT_INCLUDE_DISCLAIMER = True

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

SYNC_BATCH_SIZE       = 50     # number of offline records pushed per sync cycle

# -----------------------------------------------------------------------------
# NETWORK / CONNECTIVITY  →  core/mode_detector.py
# -----------------------------------------------------------------------------
CONNECTIVITY_CHECK_URLS = [
    "https://www.google.com",
    "https://api.groq.com",
    "https://cloudflare.com",
]
CONNECTIVITY_TIMEOUT    = 3     # seconds
CONNECTIVITY_RETRIES    = 2

MODE_ONLINE  = "online"
MODE_OFFLINE = "offline"

# -----------------------------------------------------------------------------
# AUTHENTICATION  →  services/auth_service.py
# -----------------------------------------------------------------------------
SESSION_EXPIRY_MINUTES = 480    # 8-hour clinical shift
JWT_ALGORITHM          = "HS256"
PASSWORD_MIN_LENGTH    = 8
MAX_LOGIN_ATTEMPTS     = 5
LOCKOUT_DURATION_MIN   = 15     # minutes after max failed attempts

ROLES = {
    "DOCTOR":     "doctor",
    "NURSE":      "nurse",
    "ADMIN":      "admin",
    "VIEWER":     "viewer",
}

# -----------------------------------------------------------------------------
# DATA FILE PATHS
# -----------------------------------------------------------------------------
HEART_DATA_PATH    = "data/heart_data.csv"
SAMPLE_INPUT_PATH  = "data/sample_input.json"
ASSETS_CSS_PATH    = "assets/styles.css"
ASSETS_LOGO_PATH   = "assets/logo.png"

# -----------------------------------------------------------------------------
# UI / STREAMLIT  →  ui/ modules
# -----------------------------------------------------------------------------
PAGE_TITLE         = "AuraEcho+ | Cardiac AI"
PAGE_ICON          = "🫀"
PAGE_LAYOUT        = "wide"
SIDEBAR_STATE      = "expanded"

# Plotly chart theme
CHART_THEME        = "plotly_dark"
CHART_FONT_FAMILY  = "Inter, sans-serif"
CHART_BG_COLOR     = "#0f1117"
CHART_GRID_COLOR   = "#2a2a3e"

# Number of similar cases displayed in the results panel
RESULTS_TOP_N_CASES = 3

# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------
LOG_LEVEL  = "INFO"    # DEBUG | INFO | WARNING | ERROR | CRITICAL
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
LOG_FILE   = "auraecho.log"

# -----------------------------------------------------------------------------
# MISCELLANEOUS
# -----------------------------------------------------------------------------
DECIMAL_PRECISION = 4      # float rounding for scores / probabilities
DATE_FORMAT       = "%Y-%m-%d %H:%M:%S"
UNKNOWN_LABEL     = "Unknown"
NA_PLACEHOLDER    = "N/A"
