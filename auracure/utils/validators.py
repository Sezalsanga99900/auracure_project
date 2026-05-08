# =============================================================================
# utils/validators.py
# AuraEcho+ — Input Validation Layer
# Validates patient form data before it reaches core/AI modules.
# All functions return (is_valid: bool, error_message: str).
# =============================================================================

import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from utils.constants import (
    FEATURE_COLUMNS,
    FEATURE_RANGES,
    FEATURE_VALID_VALUES,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    PASSWORD_MIN_LENGTH,
    FEATURE_LABELS,
    ROLES,
    MAX_LOGIN_ATTEMPTS,
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
# PATIENT DEMOGRAPHIC VALIDATORS
# =============================================================================

def validate_patient_name(name: Any) -> ValidationResult:
    """
    Validate patient name (required for UI display, not for ML model).
    """
    if not is_non_empty(name):
        return False, "Patient name is required."

    name_str = str(name).strip()
    if len(name_str) < 2:
        return False, "Name must be at least 2 characters."
    if len(name_str) > 100:
        return False, "Name must be less than 100 characters."
    if not re.match(r"^[a-zA-Z\s\-\.']+$", name_str):
        return False, "Name contains invalid characters."
    return True, ""


def validate_symptoms(symptoms: Any) -> ValidationResult:
    """
    Validate symptoms description (free text field in UI).
    Optional field — empty is allowed.
    """
    if not is_non_empty(symptoms):
        return True, ""   # Optional field

    symptoms_str = str(symptoms).strip()
    if len(symptoms_str) > 2000:
        return False, "Symptoms description must be less than 2000 characters."
    return True, ""


def validate_medical_report(file_obj: Any) -> ValidationResult:
    """
    Validate uploaded medical report file.
    Supports PDF, JPG, PNG, DICOM formats.
    Optional field — None is allowed.
    """
    if file_obj is None:
        return True, ""   # Optional field

    # Handle Streamlit UploadedFile
    if hasattr(file_obj, "name"):
        filename = file_obj.name
        size = getattr(file_obj, "size", None)
        if size is None:
            try:
                size = os.path.getsize(str(file_obj))
            except OSError:
                return False, "Cannot determine file size."
    else:
        filename = str(file_obj)
        try:
            size = os.path.getsize(filename)
        except OSError:
            return False, "Cannot access uploaded file."

    # Check extension
    allowed_exts = {".pdf", ".jpg", ".jpeg", ".png", ".dcm", ".dicom"}
    ext = os.path.splitext(filename.lower())[1]
    if ext not in allowed_exts:
        return False, f"Invalid file type. Allowed: {', '.join(sorted(allowed_exts))}"

    # Check size (max 10 MB)
    max_size = 10 * 1024 * 1024
    if size > max_size:
        return False, "File size must be less than 10 MB."

    return True, ""


# =============================================================================
# PER-FEATURE VALIDATORS (Clinical Features)
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
    """
    FIXED: Resting BP is always a whole number clinically.
    Changed float check → int check.
    """
    if not is_non_empty(value):
        return False, "Resting blood pressure is required."
    if not is_int_like(value):
        return False, "Resting blood pressure must be a whole number."
    bp = int(float(str(value)))
    lo, hi = FEATURE_RANGES["trestbps"]
    if not in_range(bp, lo, hi):
        return False, f"Resting BP must be between {lo} and {hi} mm Hg."
    return True, ""


def validate_cholesterol(value: Any) -> ValidationResult:
    """
    FIXED: Cholesterol is always a whole number (mg/dl) clinically.
    Changed float check → int check.
    """
    if not is_non_empty(value):
        return False, "Cholesterol is required."
    if not is_int_like(value):
        return False, "Cholesterol must be a whole number."
    chol = int(float(str(value)))
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
    """
    FIXED: Max heart rate is always a whole number clinically.
    Changed float check → int check.
    """
    if not is_non_empty(value):
        return False, "Maximum heart rate is required."
    if not is_int_like(value):
        return False, "Maximum heart rate must be a whole number."
    hr = int(float(str(value)))
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

_FIELD_VALIDATORS: Dict[str, Any] = {
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

def validate_patient(
    patient: Dict[str, Any],
    check_structure: bool = False,       # ADDED: structure check for similarity.py
) -> Tuple[bool, Dict[str, str]]:
    """
    Validate a full patient feature dictionary.

    Args:
        patient         : dict mapping feature names to raw input values.
        check_structure : if True, also checks that all required keys exist.

    Returns:
        (all_valid: bool, errors: Dict[field_key → error_message])
    """
    errors: Dict[str, str] = {}

    # ADDED: Structure check — all required keys present
    if check_structure:
        missing_keys = set(FEATURE_COLUMNS) - set(patient.keys())
        if missing_keys:
            for k in sorted(missing_keys):
                errors[k] = f"Required field '{k}' is missing."
            return False, errors

    for field in FEATURE_COLUMNS:
        validator = _FIELD_VALIDATORS.get(field)
        if validator is None:
            logger.warning(
                "validate_patient: no validator for field '%s'", field
            )
            continue

        value = patient.get(field)
        ok, msg = validator(value)
        if not ok:
            errors[field] = msg

    all_valid = len(errors) == 0

    if not all_valid:
        logger.debug(
            "validate_patient: %d field error(s) — %s",
            len(errors), list(errors.keys()),
        )

    return all_valid, errors


def validate_single_field(field: str, value: Any) -> ValidationResult:
    """
    Validate a single patient field by name.
    Useful for real-time form validation in the Streamlit sidebar.
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
    - ADDED: At least one special character (clinical security standard)
    """
    if not is_non_empty(password):
        return False, "Password is required."
    if len(password) < PASSWORD_MIN_LENGTH:
        return False, f"Password must be at least {PASSWORD_MIN_LENGTH} characters."
    if not any(c.isupper() for c in password):
        return False, "Password must contain at least one uppercase letter."
    if not any(c.isdigit() for c in password):
        return False, "Password must contain at least one digit."
    # ADDED: special character requirement
    special_chars = set("!@#$%^&*()_+-=[]{}|;':\",./<>?")
    if not any(c in special_chars for c in password):
        return False, "Password must contain at least one special character."
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


def validate_role(role: Any) -> ValidationResult:
    """
    Validate role selection against allowed roles from constants.
    """
    if not is_non_empty(role):
        return False, "Role is required."

    role_str = str(role).strip().lower()
    valid_roles = [r.lower() for r in ROLES.values()]

    if role_str not in valid_roles:
        return False, f"Invalid role. Must be one of: {', '.join(valid_roles)}."
    return True, ""


def validate_login_attempt(attempts: int) -> ValidationResult:
    """
    ADDED: Check if login attempts exceed the maximum allowed.
    Used by services/auth_service.py.
    """
    if attempts >= MAX_LOGIN_ATTEMPTS:
        return False, (
            f"Account locked after {MAX_LOGIN_ATTEMPTS} failed attempts. "
            f"Please try again later."
        )
    return True, ""


def validate_api_key(key: str, provider: str = "API") -> ValidationResult:
    """
    ADDED: Basic API key format validation.
    Used by services and UI system status checks.
    """
    if not is_non_empty(key):
        return False, f"{provider} API key is required."
    key = key.strip()
    if len(key) < 20:
        return False, f"{provider} API key appears too short."
    if " " in key:
        return False, f"{provider} API key must not contain spaces."
    return True, ""


# =============================================================================
# DATA FILE VALIDATORS
# =============================================================================

def validate_csv_columns(df_columns: List[str]) -> ValidationResult:
    """
    Check that a loaded DataFrame contains all required feature columns.
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
# BATCH VALIDATOR
# =============================================================================

def validate_batch(
    records: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Tuple[int, Dict[str, str]]]]:
    """
    Validate a list of patient records.

    Returns:
        valid_records   : list of records that passed validation
        invalid_records : list of (index, errors) for failed records
    """
    valid:   List[Dict[str, Any]]               = []
    invalid: List[Tuple[int, Dict[str, str]]]   = []

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

    FIXED: Handles '_' generic error key cleanly (no ugly underscore shown).

    Example:
        {"age": "Age must be positive", "_": "Generic error"}
        → "• Age (years): Age must be positive\\n• Generic error"
    """
    lines = []
    for k, v in errors.items():
        if k == "_":
            lines.append(f"• {v}")
        else:
            label = FEATURE_LABELS.get(k, k)
            lines.append(f"• {label}: {v}")
    return "\n".join(lines)


def format_validation_errors(errors: Dict[str, str]) -> str:
    """
    ADDED: Alias for errors_to_str().
    Used by core/similarity.py which imports this name directly.
    """
    return errors_to_str(errors)