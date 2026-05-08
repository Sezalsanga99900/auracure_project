# =============================================================================
# utils/helpers.py
# AuraEcho+ — Shared Utility Functions
# Generic, reusable helpers used across the entire project.
# No project-specific logic here — pure utility belt.
# =============================================================================

import os
import json
import logging
import hashlib
import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np

from utils.constants import (
    LOG_LEVEL,
    LOG_FORMAT,
    LOG_FILE,
    DATE_FORMAT,
    DECIMAL_PRECISION,
    RISK_LEVELS,
    RISK_LABELS,
    RISK_COLORS,
    RISK_ICONS,
    CHEST_PAIN_LABELS,
    THAL_LABELS,
    SLOPE_LABELS,
    RESTECG_LABELS,
    FEATURE_LABELS,
    UNKNOWN_LABEL,
    NA_PLACEHOLDER,
    ROLE_PERMISSIONS,
    SAMPLE_INPUT_PATH,
    MODEL_SAVE_PATH,
    SCALER_SAVE_PATH,
    UI_SUCCESS_COLOR,
    UI_WARNING_COLOR,
    MODE_ONLINE,
    MODE_OFFLINE,
    MODE_ONLINE_LABEL,
    MODE_OFFLINE_LABEL,
)

# =============================================================================
# LOGGING SETUP
# =============================================================================

def get_logger(name: str) -> logging.Logger:
    """
    Returns a named logger configured with file + stream handlers.
    Call once at the top of each module:  logger = get_logger(__name__)
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    formatter = logging.Formatter(LOG_FORMAT)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    try:
        fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    except OSError:
        pass

    return logger


logger = get_logger(__name__)


# =============================================================================
# TYPE ALIASES
# =============================================================================

PatientDict  = Dict[str, Any]
FeatureArray = np.ndarray


# =============================================================================
# NUMBER / MATH HELPERS
# =============================================================================

def safe_round(value: Any, decimals: int = DECIMAL_PRECISION) -> float:
    """Round a value safely, returning 0.0 on errors."""
    try:
        return round(float(value), decimals)
    except (TypeError, ValueError):
        return 0.0


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp *value* to [lo, hi]."""
    return max(lo, min(hi, value))


def normalise(value: float, min_val: float, max_val: float) -> float:
    """
    Min-max normalise *value* to [0, 1].
    Returns 0.0 if min_val == max_val to avoid division by zero.
    """
    if max_val == min_val:
        return 0.0
    return clamp((value - min_val) / (max_val - min_val))


def pct(value: float, total: float, decimals: int = 1) -> float:
    """Return value/total * 100, or 0.0 if total is zero."""
    if total == 0:
        return 0.0
    return safe_round((value / total) * 100, decimals)


# =============================================================================
# RISK LEVEL HELPERS
# =============================================================================

def score_to_risk_level(score: float) -> str:
    """
    Map a risk score in [0, 1] to a risk level key: 'LOW' | 'MEDIUM' | 'HIGH'.

    Example:
        >>> score_to_risk_level(0.72)
        'HIGH'
    """
    score = clamp(score)
    for level, (lo, hi) in RISK_LEVELS.items():
        if lo <= score < hi:
            return level
    return "HIGH"


def risk_badge(score: float) -> Dict[str, str]:
    """
    Returns a dict with label, color, and icon for a risk score.

    Returns:
        {
            "level":  "HIGH",
            "label":  "High Risk",
            "color":  "#e74c3c",
            "icon":   "🚨",
            "score":  "0.7832",
        }
    """
    level = score_to_risk_level(score)
    return {
        "level": level,
        "label": RISK_LABELS[level],
        "color": RISK_COLORS[level],
        "icon":  RISK_ICONS[level],
        "score": f"{score:.4f}",
    }


def normalize_risk_level(raw: Optional[str]) -> Optional[str]:
    """
    ADDED: Normalize risk level — converts label or key → key.

    Handles both:
        "High Risk"  → "HIGH"   (label → key)
        "HIGH"       → "HIGH"   (key   → key)

    Used across ai/ modules to fix key/label confusion.
    """
    if not raw:
        return None
    if raw in RISK_LEVELS:
        return raw                                  # already a key
    label_to_key = {v: k for k, v in RISK_LABELS.items()}
    return label_to_key.get(raw, raw.upper())


# =============================================================================
# ROLE & PERMISSION HELPERS
# =============================================================================

def has_permission(role: str, permission: str) -> bool:
    """
    Check if a role has a specific permission.

    Args:
        role       : role key e.g. 'doctor', 'nurse'
        permission : permission string e.g. 'view_analytics'

    Returns:
        True if role exists and has permission, False otherwise.
    """
    if not role or not permission:
        return False
    perms = ROLE_PERMISSIONS.get(role.lower(), [])
    return permission in perms


def get_role_permissions(role: str) -> List[str]:
    """
    Get all permissions for a specific role.

    Args:
        role : role key

    Returns:
        List of permission strings, empty list if role not found.
    """
    return ROLE_PERMISSIONS.get(role.lower(), [])


# =============================================================================
# DICT / PATIENT RECORD HELPERS
# =============================================================================

def flatten_dict(d: Dict[str, Any], prefix: str = "", sep: str = ".") -> Dict[str, Any]:
    """
    Recursively flatten a nested dict.

    Example:
        {"a": {"b": 1}} → {"a.b": 1}
    """
    items: Dict[str, Any] = {}
    for k, v in d.items():
        new_key = f"{prefix}{sep}{k}" if prefix else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep))
        else:
            items[new_key] = v
    return items


def safe_get(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    """dict.get() with a fallback that also handles None values."""
    val = d.get(key, default)
    return default if val is None else val


def strip_none(d: Dict[str, Any]) -> Dict[str, Any]:
    """Remove keys whose value is None from a dict (shallow)."""
    return {k: v for k, v in d.items() if v is not None}


def patient_to_display(patient: PatientDict) -> Dict[str, str]:
    """
    Convert a raw patient feature dict into human-readable label → value pairs
    for display in the UI results panel.

    Example:
        {"cp": 2, "age": 54} → {"Chest Pain Type": "Non-Anginal Pain", ...}
    """
    display: Dict[str, str] = {}
    for key, value in patient.items():
        label = FEATURE_LABELS.get(key, key.replace("_", " ").title())

        if key == "cp":
            value = CHEST_PAIN_LABELS.get(int(value), UNKNOWN_LABEL)
        elif key == "thal":
            value = THAL_LABELS.get(int(value), UNKNOWN_LABEL)
        elif key == "slope":
            value = SLOPE_LABELS.get(int(value), UNKNOWN_LABEL)
        elif key == "restecg":
            value = RESTECG_LABELS.get(int(value), UNKNOWN_LABEL)
        elif key == "sex":
            value = "Male" if int(value) == 1 else "Female"
        elif key in ("fbs", "exang"):
            value = "Yes" if int(value) == 1 else "No"
        else:
            value = safe_round(value) if isinstance(value, float) else value

        display[label] = str(value)
    return display


# =============================================================================
# DATE / TIME HELPERS
# =============================================================================

def now_str() -> str:
    """Current UTC timestamp as a formatted string."""
    return datetime.datetime.now(datetime.timezone.utc).strftime(DATE_FORMAT)


def ts_to_str(ts: datetime.datetime) -> str:
    """Convert a datetime object to the app's standard string format."""
    return ts.strftime(DATE_FORMAT)


def str_to_ts(s: str) -> Optional[datetime.datetime]:
    """Parse a date string back to a datetime, or None on failure."""
    try:
        return datetime.datetime.strptime(s, DATE_FORMAT)
    except (ValueError, TypeError):
        return None


def elapsed_seconds(start: datetime.datetime) -> float:
    """Seconds elapsed since *start* (UTC)."""
    return (
        datetime.datetime.now(datetime.timezone.utc) - start
    ).total_seconds()


# =============================================================================
# FILE / IO HELPERS
# =============================================================================

def load_json(path: str) -> Optional[Dict[str, Any]]:
    """
    Load a JSON file and return its contents as a dict.
    Returns None if the file does not exist or cannot be parsed.
    """
    if not os.path.exists(path):
        logger.warning("load_json: file not found → %s", path)
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        logger.error("load_json: JSON parse error in %s — %s", path, exc)
        return None


def save_json(data: Any, path: str, indent: int = 2) -> bool:
    """
    Save *data* to a JSON file. Creates parent directories if needed.
    Returns True on success, False on failure.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, default=str)
        return True
    except OSError as exc:
        logger.error("save_json: could not write %s — %s", path, exc)
        return False


def load_csv(path: str) -> Optional[pd.DataFrame]:
    """
    Load a CSV file into a DataFrame.
    Returns None if the file does not exist or cannot be read.
    """
    if not os.path.exists(path):
        logger.warning("load_csv: file not found → %s", path)
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        logger.error("load_csv: error reading %s — %s", path, exc)
        return None


def ensure_dir(path: str) -> None:
    """Create *path* (and all parents) if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def file_exists(path: str) -> bool:
    """Return True if *path* points to an existing file."""
    return os.path.isfile(path)


def load_sample_input() -> Optional[Dict[str, Any]]:
    """
    Load the sample patient input from JSON (data/sample_input.json).
    Convenience wrapper for quick testing.
    """
    return load_json(SAMPLE_INPUT_PATH)


def model_files_exist() -> Tuple[bool, bool]:
    """
    Check if pre-trained model and scaler files exist.

    Returns:
        Tuple of (model_exists, scaler_exists)
    """
    return file_exists(MODEL_SAVE_PATH), file_exists(SCALER_SAVE_PATH)


# =============================================================================
# STRING HELPERS
# =============================================================================

def truncate(text: str, max_len: int = 120, suffix: str = "...") -> str:
    """Truncate *text* to *max_len* characters, appending *suffix* if cut."""
    if len(text) <= max_len:
        return text
    return text[: max_len - len(suffix)] + suffix


def safe_str(value: Any, fallback: str = NA_PLACEHOLDER) -> str:
    """Convert *value* to string; return *fallback* for None / empty."""
    if value is None:
        return fallback
    s = str(value).strip()
    return s if s else fallback


def title_case(text: str) -> str:
    """'hello_world_key' → 'Hello World Key'"""
    return text.replace("_", " ").title()


def hash_string(text: str) -> str:
    """Return the SHA-256 hex digest of *text*."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# =============================================================================
# DATAFRAME / NUMPY HELPERS
# =============================================================================

def df_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert a DataFrame to a list of dicts (orient='records')."""
    return df.to_dict(orient="records")


def records_to_df(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert a list of dicts back to a DataFrame."""
    return pd.DataFrame(records)


def series_to_dict(s: pd.Series) -> Dict[str, Any]:
    """Convert a pandas Series to a plain dict."""
    return s.to_dict()


def is_numeric(value: Any) -> bool:
    """Return True if *value* can be interpreted as a number."""
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def coerce_numeric(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Coerce specified *columns* in *df* to numeric, setting errors as NaN.
    Returns a new DataFrame (does not mutate the original).
    """
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def fill_missing(
    df: pd.DataFrame,
    strategy: str = "median",
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Fill NaN values in *df*.

    Args:
        strategy : 'median' | 'mean' | 'zero' | 'mode'
        columns  : list of column names to fill; defaults to all numeric cols.

    Returns a new DataFrame.
    """
    df = df.copy()
    cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()

    for col in cols:
        if col not in df.columns:
            continue
        if strategy == "median":
            df[col] = df[col].fillna(df[col].median())
        elif strategy == "mean":
            df[col] = df[col].fillna(df[col].mean())
        elif strategy == "zero":
            df[col] = df[col].fillna(0)
        elif strategy == "mode":
            mode_val = df[col].mode()
            df[col] = df[col].fillna(mode_val[0] if not mode_val.empty else 0)
        else:
            raise ValueError(f"Unknown fill strategy: {strategy!r}")

    return df


# =============================================================================
# FORMATTING HELPERS  (for UI display)
# =============================================================================

def format_score(score: float) -> str:
    """Format a risk score as a percentage string, e.g. '72.34%'."""
    return f"{score * 100:.2f}%"


def format_similarity(sim: float) -> str:
    """Format a similarity score, e.g. '87.4% match'."""
    return f"{sim * 100:.1f}% match"


def format_patient_id(patient_id: Union[int, str]) -> str:
    """Zero-pad a patient ID for display, e.g. 7 → 'P-000007'."""
    return f"P-{int(patient_id):06d}"


def bullet_list(items: List[str], bullet: str = "•") -> str:
    """Join a list of strings into a bullet-separated display string."""
    return "\n".join(f"{bullet} {item}" for item in items)


def format_mode(mode: str) -> Dict[str, Any]:
    """
    Format online/offline mode for UI display.
    FIXED: Now uses MODE_ONLINE_LABEL / MODE_OFFLINE_LABEL from constants.

    Returns:
        Dict with label, color, icon, is_online, badge_class
    """
    mode = mode.lower()
    if mode == MODE_ONLINE:
        return {
            "label":       MODE_ONLINE_LABEL,
            "color":       UI_SUCCESS_COLOR,
            "icon":        "🌐",
            "is_online":   True,
            "badge_class": "online-badge",
        }
    return {
        "label":       MODE_OFFLINE_LABEL,
        "color":       UI_WARNING_COLOR,
        "icon":        "📴",
        "is_online":   False,
        "badge_class": "offline-badge",
    }


# =============================================================================
# ENVIRONMENT HELPERS
# =============================================================================

def get_env(key: str, default: str = "") -> str:
    """Fetch an environment variable with a safe default."""
    return os.environ.get(key, default)


def require_env(key: str) -> str:
    """
    Fetch a required environment variable.
    Raises EnvironmentError if not set.
    """
    value = os.environ.get(key)
    if not value:
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set. "
            f"Add it to your .env file."
        )
    return value


def mask_key(key: str, visible: int = 4) -> str:
    """
    Mask an API key for safe logging.
    e.g. 'sk-abcd1234efgh5678' → 'sk-a****5678'
    """
    if len(key) <= visible * 2:
        return "*" * len(key)
    return key[:visible] + "*" * (len(key) - visible * 2) + key[-visible:]


# =============================================================================
# RUNTIME CHECKS
# =============================================================================

def check_dependencies() -> Tuple[bool, List[str]]:
    """
    Verify that all required Python packages are importable.
    Returns (all_ok: bool, missing: List[str]).

    FIXED: Corrected package import names.
    """
    required = {
        "streamlit":   "streamlit",
        "pandas":      "pandas",
        "numpy":       "numpy",
        "sklearn":     "sklearn",
        "plotly":      "plotly",
        "requests":    "requests",
        "dotenv":      "dotenv",
        "groq":        "groq",
        "openai":      "openai",
    }
    missing = []
    for display_name, import_name in required.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(display_name)
    return len(missing) == 0, missing

import streamlit as st
import os

def local_css(file_name):
    """
    Reads a local CSS file and injects it into the Streamlit app.
    """
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else:
        st.error(f"CSS file not found at {file_name}")