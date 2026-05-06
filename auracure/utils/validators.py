# =============================================================================
# utils/validators.py
# AuraEcho+ — Input Validation Layer
# Validates patient form data before it reaches core/AI modules.
# All functions return (is_valid: bool, error_message: str).
# =============================================================================

import re
from typing import Any, Dict, List, Optional, Tuple

from utils.constants import (
    FEATURE_COLUMNS,
    FEATURE_RANGES,
    FEATURE_VALID_VALUES,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    PASSWORD_MIN_LENGTH,
    FEATURE_LABELS,
)
from utils.helpers import get_logger, is_numeric

logger = get_logger(__name__)

# Type alias
ValidationResult = Tuple[bool, str]   # (is_valid, error_message)


# =============================================================================
# LOW-LEVEL PRIMITIVE VALIDATORS
# =============================================================================

def is_non_empty(value: Any) -> bool:
    """Return True if *value* is not None and not an empty string."""
    if value is None:
        return False
    return str(value).strip() != ""


def is_int_like(value: Any) -> bool:
    """Return True if *value* can be cast to an integer."""
    try:
        int(float(str(value)))
        return True
    except (TypeError, ValueError):
        return False


def is_float_like(value: Any) -> bool:
    """Return True if *value* can be cast to a float."""
    try:
        float(str(value))
        return True
    except (TypeError, ValueError):
        return False


def in_range(value: float, lo: float, hi: float) -> bool:
    """Return True if *lo* <= *value* <= *hi*."""
    return lo <= value <= hi


def in_allowed(value: Any, allowed: List) -> bool:
    """Return True if *value* (int-cast) is in *allowed*."""
    try:
        return int(float(str(value))) in allowed
    except (TypeError, ValueError):
        return False


# =============================================================================
# PER-FEATURE VALIDATORS
# =============================================================================

def validate_age(value: Any) -> ValidationResult:
    if not is_non_empty(value):
        return False, "Age is required."
    if not is_int_like(value):
        return False, "Age must be a whole number."
    age = int(float(str(value)))
    lo, hi = FEATURE_RANGES["age"]
    if not in_range(age, lo, hi):
        return False, f"Age must be between {lo} and {hi} years."
    return True, ""


def validate_sex(value: Any) -> ValidationResult:
    if not is_non_empty(value):
        return False, "Sex is required."
    if not in_allowed(value, FEATURE_VALID_VALUES["sex"]):
        return False, "Sex must be 0 (Female) or 1 (Male)."
    return True, ""


def validate_chest_pain(value: Any) -> ValidationResult:
    if not is_non_empty(value):
        return False, "Chest pain type is required."
    if not in_allowed(value, FEATURE_VALID_VALUES["cp"]):
        return False, "Chest pain type must be 0, 1, 2, or 3."
    return True, ""


def validate_resting_bp(value: Any) -> ValidationResult:
    if not is_non_empty(value):
        return False, "Resting blood pressure is required."
    if not is_float_like(value):
        return False, "Resting blood pressure must be a number."
    bp = float(str(value))
    lo, hi = FEATURE_RANGES["trestbps"]
    if not in_range(bp, lo, hi):
        return False, f"Resting BP must be between {lo} and {hi} mm Hg."
    return True, ""


def validate_cholesterol(value: Any) -> ValidationResult:
    if not is_non_empty(value):
        return False, "Cholesterol is required."
    if not is_float_like(value):
        return False, "Cholesterol must be a number."
    chol = float(str(value))
    lo, hi = FEATURE_RANGES["chol"]
    if not in_range(chol, lo, hi):
        return False, f"Cholesterol must be between {lo} and {hi} mg/dl."
    return True, ""


def validate_fbs(value: Any) -> ValidationResult:
    if not is_non_empty(value):
        return False, "Fasting blood sugar field is required."
    if not in_allowed(value, FEATURE_VALID_VALUES["fbs"]):
        return False, "Fasting blood sugar must be 0 (≤120 mg/dl) or 1 (>120 mg/dl)."
    return True, ""


def validate_restecg(value: Any) -> ValidationResult:
    if not is_non_empty(value):
        return False, "Resting ECG result is required."
    if not in_allowed(value, FEATURE_VALID_VALUES["restecg"]):
        return False, "Resting ECG must be 0, 1, or 2."
    return True, ""


def validate_max_hr(value: Any) -> ValidationResult:
    if not is_non_empty(value):
        return False, "Maximum heart rate is required."
    if not is_float_like(value):
        return False, "Maximum heart rate must be a number."
    hr = float(str(value))
    lo, hi = FEATURE_RANGES["thalach"]
    if not in_range(hr, lo, hi):
        return False, f"Max heart rate must be between {lo} and {hi} bpm."
    return True, ""


def validate_exang(value: Any) -> ValidationResult:
    if not is_non_empty(value):
        return False, "Exercise-induced angina field is required."
    if not in_allowed(value, FEATURE_VALID_VALUES["exang"]):
        return False, "Exercise-induced angina must be 0 (No) or 1 (Yes)."
    return True, ""


def validate_oldpeak(value: Any) -> ValidationResult:
    if not is_non_empty(value):
        return False, "ST depression (oldpeak) is required."
    if not is_float_like(value):
        return False, "Oldpeak must be a number."
    op = float(str(value))
    lo, hi = FEATURE_RANGES["oldpeak"]
    if not in_range(op, lo, hi):
        return False, f"Oldpeak must be between {lo} and {hi}."
    return True, ""


def validate_slope(value: Any) -> ValidationResult:
    if not is_non_empty(value):
        return False, "ST slope is required."
    if not in_allowed(value, FEATURE_VALID_VALUES["slope"]):
        return False, "Slope must be 0, 1, or 2."
    return True, ""


def validate_ca(value: Any) -> ValidationResult:
    if not is_non_empty(value):
        return False, "Number of major vessels (ca) is required."
    if not in_allowed(value, FEATURE_VALID_VALUES["ca"]):
        return False, "Number of major vessels must be 0, 1, 2, or 3."
    return True, ""


def validate_thal(value: Any) -> ValidationResult:
    if not is_non_empty(value):
        return False, "Thalassemia type is required."
    if not in_allowed(value, FEATURE_VALID_VALUES["thal"]):
        return False, "Thalassemia must be 1 (Normal), 2 (Fixed Defect), or 3 (Reversible)."
    return True, ""


# =============================================================================
# FIELD → VALIDATOR DISPATCH TABLE
# =============================================================================

_FIELD_VALIDATORS = {
    "age":      validate_age,
    "sex":      validate_sex,
    "cp":       validate_chest_pain,
    "trestbps": validate_resting_bp,
    "chol":     validate_cholesterol,
    "fbs":      validate_fbs,
    "restecg":  validate_restecg,
    "thalach":  validate_max_hr,
    "exang":    validate_exang,
    "oldpeak":  validate_oldpeak,
    "slope":    validate_slope,
    "ca":       validate_ca,
    "thal":     validate_thal,
}


# =============================================================================
# PATIENT DICT VALIDATOR (main entry point)
# =============================================================================

def validate_patient(patient: Dict[str, Any]) -> Tuple[bool, Dict[str, str]]:
    """
    Validate a full patient feature dictionary.

    Args:
        patient: dict mapping feature names to raw input values.

    Returns:
        (all_valid: bool, errors: Dict[field_key → error_message])

    Example:
        patient = {"age": 54, "sex": 1, "cp": 2, ...}
        ok, errors = validate_patient(patient)
        if not ok:
            for field, msg in errors.items():
                print(f"{field}: {msg}")
    """
    errors: Dict[str, str] = {}

    for field in FEATURE_COLUMNS:
        validator = _FIELD_VALIDATORS.get(field)
        if validator is None:
            logger.warning("validate_patient: no validator for field '%s'", field)
            continue

        value = patient.get(field)
        ok, msg = validator(value)
        if not ok:
            errors[field] = msg

    all_valid = len(errors) == 0

    if not all_valid:
        logger.debug("validate_patient: %d field error(s) — %s", len(errors), list(errors.keys()))

    return all_valid, errors


def validate_single_field(field: str, value: Any) -> ValidationResult:
    """
    Validate a single patient field by name.
    Useful for real-time form validation in the Streamlit sidebar.

    Returns:
        (is_valid: bool, error_message: str)
    """
    validator = _FIELD_VALIDATORS.get(field)
    if validator is None:
        return True, ""   # unknown field — pass through
    return validator(value)


# =============================================================================
# AUTHENTICATION VALIDATORS
# =============================================================================

def validate_email(email: str) -> ValidationResult:
    """Basic e-mail format check."""
    if not is_non_empty(email):
        return False, "Email address is required."
    pattern = r"^[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}$"
    if not re.match(pattern, email.strip()):
        return False, "Please enter a valid email address."
    return True, ""


def validate_password(password: str) -> ValidationResult:
    """
    Password strength check:
    - Minimum length (from constants)
    - At least one uppercase letter
    - At least one digit
    """
    if not is_non_empty(password):
        return False, "Password is required."
    if len(password) < PASSWORD_MIN_LENGTH:
        return False, f"Password must be at least {PASSWORD_MIN_LENGTH} characters."
    if not any(c.isupper() for c in password):
        return False, "Password must contain at least one uppercase letter."
    if not any(c.isdigit() for c in password):
        return False, "Password must contain at least one digit."
    return True, ""


def validate_username(username: str) -> ValidationResult:
    """Username: 3–30 alphanumeric / underscore characters."""
    if not is_non_empty(username):
        return False, "Username is required."
    username = username.strip()
    if not (3 <= len(username) <= 30):
        return False, "Username must be 3–30 characters long."
    if not re.match(r"^[a-zA-Z0-9_]+$", username):
        return False, "Username may only contain letters, numbers, and underscores."
    return True, ""


# =============================================================================
# DATA FILE VALIDATORS
# =============================================================================

def validate_csv_columns(df_columns: List[str]) -> ValidationResult:
    """
    Check that a loaded DataFrame contains all required feature columns.
    Used when loading heart_data.csv to catch malformed files early.
    """
    required = set(FEATURE_COLUMNS + ["target"])
    present  = set(df_columns)
    missing  = required - present

    if missing:
        missing_labels = [FEATURE_LABELS.get(c, c) for c in sorted(missing)]
        return False, f"CSV is missing required columns: {', '.join(missing_labels)}"

    return True, ""


def validate_sample_input(data: Dict[str, Any]) -> Tuple[bool, Dict[str, str]]:
    """
    Validate a sample_input.json payload.
    Delegates to validate_patient after confirming it is a dict.
    """
    if not isinstance(data, dict):
        return False, {"_": "Sample input must be a JSON object (dict)."}

    return validate_patient(data)


# =============================================================================
# BATCH VALIDATOR (for syncing offline records)
# =============================================================================

def validate_batch(
    records: List[Dict[str, Any]],
) -> Tuple[List[Dict], List[Tuple[int, Dict[str, str]]]]:
    """
    Validate a list of patient records.

    Returns:
        valid_records:   list of records that passed validation
        invalid_records: list of (index, errors) tuples for failed records
    """
    valid: List[Dict] = []
    invalid: List[Tuple[int, Dict[str, str]]] = []

    for i, record in enumerate(records):
        ok, errors = validate_patient(record)
        if ok:
            valid.append(record)
        else:
            invalid.append((i, errors))

    logger.info(
        "validate_batch: %d/%d records valid, %d invalid",
        len(valid), len(records), len(invalid),
    )
    return valid, invalid


# =============================================================================
# FORMATTING HELPERS
# =============================================================================

def errors_to_str(errors: Dict[str, str]) -> str:
    """
    Flatten a field-error dict into a single readable string.

    Example:
        {"age": "Age must be positive", "chol": "Cholesterol out of range"}
        → "• age: Age must be positive\n• chol: Cholesterol out of range"
    """
    lines = [f"• {FEATURE_LABELS.get(k, k)}: {v}" for k, v in errors.items()]
    return "\n".join(lines)
