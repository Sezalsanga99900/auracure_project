# =============================================================================
# tests/test_validators.py
# AuraEcho+ — Validation Layer Tests
#
# Coverage:
#     • Primitive validators (is_non_empty, is_int_like, etc.)
#     • Demographic validators (name, symptoms, medical report)
#     • Clinical feature validators (all 13 features)
#     • Patient dict validation (with structure check)
#     • Auth validators (email, password, username, role, login, API key)
#     • Data file validators (CSV columns, sample input)
#     • Batch validation
#     • Formatting helpers
#
# Run:
#     pytest tests/test_validators.py -v
# =============================================================================

import pytest
import os
import tempfile
from unittest.mock import Mock

from utils.validators import (
    # Primitives
    is_non_empty,
    is_int_like,
    is_float_like,
    in_range,
    in_allowed,
    # Demographics
    validate_patient_name,
    validate_symptoms,
    validate_medical_report,
    # Clinical features
    validate_age,
    validate_sex,
    validate_chest_pain,
    validate_resting_bp,
    validate_cholesterol,
    validate_fbs,
    validate_restecg,
    validate_max_hr,
    validate_exang,
    validate_oldpeak,
    validate_slope,
    validate_ca,
    validate_thal,
    # Patient dict
    validate_patient,
    validate_single_field,
    # Auth
    validate_email,
    validate_password,
    validate_username,
    validate_role,
    validate_login_attempt,
    validate_api_key,
    # Data files
    validate_csv_columns,
    validate_sample_input,
    # Batch
    validate_batch,
    # Formatting
    errors_to_str,
    format_validation_errors,
)
from utils.constants import (
    FEATURE_COLUMNS,
    FEATURE_RANGES,
    FEATURE_VALID_VALUES,
    PASSWORD_MIN_LENGTH,
    MAX_LOGIN_ATTEMPTS,
    ROLES,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def valid_patient():
    """Returns a fully valid patient dictionary."""
    return {
        "age": 55,
        "sex": 1,
        "cp": 2,
        "trestbps": 130,
        "chol": 240,
        "fbs": 0,
        "restecg": 1,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 1.5,
        "slope": 1,
        "ca": 1,
        "thal": 2,
    }


@pytest.fixture
def mock_uploaded_file():
    """Returns a mock Streamlit UploadedFile object."""
    mock = Mock()
    mock.name = "report.pdf"
    mock.size = 1024 * 1024  # 1MB
    return mock


# ─────────────────────────────────────────────────────────────────────────────
# Primitive Validators
# ─────────────────────────────────────────────────────────────────────────────

class TestPrimitives:
    def test_is_non_empty(self):
        assert is_non_empty("hello") is True
        assert is_non_empty("  hello  ") is True
        assert is_non_empty("") is False
        assert is_non_empty("   ") is False
        assert is_non_empty(None) is False
        assert is_non_empty(0) is True  # 0 is non-empty when stringified

    def test_is_int_like(self):
        assert is_int_like(55) is True
        assert is_int_like("55") is True
        assert is_int_like("55.0") is True
        assert is_int_like(55.0) is True
        assert is_int_like("55.5") is False
        assert is_int_like("abc") is False
        assert is_int_like(None) is False

    def test_is_float_like(self):
        assert is_float_like(55.5) is True
        assert is_float_like("55.5") is True
        assert is_float_like("55") is True
        assert is_float_like("abc") is False
        assert is_float_like(None) is False

    def test_in_range(self):
        assert in_range(50, 0, 100) is True
        assert in_range(0, 0, 100) is True
        assert in_range(100, 0, 100) is True
        assert in_range(-1, 0, 100) is False
        assert in_range(101, 0, 100) is False

    def test_in_allowed(self):
        assert in_allowed(1, [0, 1, 2]) is True
        assert in_allowed("1", [0, 1, 2]) is True
        assert in_allowed(3, [0, 1, 2]) is False
        assert in_allowed("abc", [0, 1, 2]) is False


# ─────────────────────────────────────────────────────────────────────────────
# Demographic Validators
# ─────────────────────────────────────────────────────────────────────────────

class TestDemographics:
    def test_validate_patient_name_valid(self):
        ok, msg = validate_patient_name("John Doe")
        assert ok is True
        assert msg == ""

    def test_validate_patient_name_empty(self):
        ok, msg = validate_patient_name("")
        assert ok is False
        assert "required" in msg.lower()

    def test_validate_patient_name_too_short(self):
        ok, msg = validate_patient_name("A")
        assert ok is False
        assert "at least 2" in msg.lower()

    def test_validate_patient_name_invalid_chars(self):
        ok, msg = validate_patient_name("John123")
        assert ok is False
        assert "invalid characters" in msg.lower()

    def test_validate_symptoms_valid(self):
        ok, msg = validate_symptoms("Chest pain on exertion")
        assert ok is True

    def test_validate_symptoms_optional(self):
        ok, msg = validate_symptoms("")
        assert ok is True  # Optional field

    def test_validate_symptoms_too_long(self):
        ok, msg = validate_symptoms("x" * 2001)
        assert ok is False
        assert "less than 2000" in msg.lower()

    def test_validate_medical_report_none(self):
        ok, msg = validate_medical_report(None)
        assert ok is True  # Optional

    def test_validate_medical_report_valid_pdf(self, mock_uploaded_file):
        ok, msg = validate_medical_report(mock_uploaded_file)
        assert ok is True

    def test_validate_medical_report_invalid_ext(self, mock_uploaded_file):
        mock_uploaded_file.name = "report.exe"
        ok, msg = validate_medical_report(mock_uploaded_file)
        assert ok is False
        assert "invalid file type" in msg.lower()

    def test_validate_medical_report_too_large(self, mock_uploaded_file):
        mock_uploaded_file.size = 15 * 1024 * 1024  # 15MB
        ok, msg = validate_medical_report(mock_uploaded_file)
        assert ok is False
        assert "less than 10MB" in msg.lower()

    def test_validate_medical_report_file_path(self):
        # Create temp file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"dummy")
            path = f.name
        try:
            ok, msg = validate_medical_report(path)
            assert ok is True
        finally:
            os.unlink(path)


# ─────────────────────────────────────────────────────────────────────────────
# Clinical Feature Validators
# ─────────────────────────────────────────────────────────────────────────────

class TestClinicalFeatures:
    def test_validate_age_valid(self):
        ok, msg = validate_age(55)
        assert ok is True

    def test_validate_age_out_of_range(self):
        ok, msg = validate_age(150)
        assert ok is False
        assert "between" in msg.lower()

    def test_validate_age_not_int(self):
        ok, msg = validate_age("abc")
        assert ok is False
        assert "whole number" in msg.lower()

    def test_validate_sex_valid(self):
        for val in FEATURE_VALID_VALUES["sex"]:
            ok, msg = validate_sex(val)
            assert ok is True

    def test_validate_sex_invalid(self):
        ok, msg = validate_sex(2)
        assert ok is False

    def test_validate_chest_pain_valid(self):
        for val in FEATURE_VALID_VALUES["cp"]:
            ok, msg = validate_chest_pain(val)
            assert ok is True

    def test_validate_resting_bp_valid_int(self):
        ok, msg = validate_resting_bp(120)
        assert ok is True

    def test_validate_resting_bp_float_rejected(self):
        # FIXED: BP must be int, float should fail
        ok, msg = validate_resting_bp(120.5)
        assert ok is False
        assert "whole number" in msg.lower()

    def test_validate_resting_bp_out_of_range(self):
        ok, msg = validate_resting_bp(300)
        assert ok is False
        assert "between" in msg.lower()

    def test_validate_cholesterol_valid_int(self):
        ok, msg = validate_cholesterol(200)
        assert ok is True

    def test_validate_cholesterol_float_rejected(self):
        # FIXED: Chol must be int
        ok, msg = validate_cholesterol(200.5)
        assert ok is False
        assert "whole number" in msg.lower()

    def test_validate_fbs_valid(self):
        for val in FEATURE_VALID_VALUES["fbs"]:
            ok, msg = validate_fbs(val)
            assert ok is True

    def test_validate_max_hr_valid_int(self):
        ok, msg = validate_max_hr(150)
        assert ok is True

    def test_validate_max_hr_float_rejected(self):
        # FIXED: HR must be int
        ok, msg = validate_max_hr(150.5)
        assert ok is False
        assert "whole number" in msg.lower()

    def test_validate_exang_valid(self):
        for val in FEATURE_VALID_VALUES["exang"]:
            ok, msg = validate_exang(val)
            assert ok is True

    def test_validate_oldpeak_valid_float(self):
        ok, msg = validate_oldpeak(1.5)
        assert ok is True

    def test_validate_oldpeak_out_of_range(self):
        ok, msg = validate_oldpeak(15.0)
        assert ok is False
        assert "between" in msg.lower()

    def test_validate_slope_valid(self):
        for val in FEATURE_VALID_VALUES["slope"]:
            ok, msg = validate_slope(val)
            assert ok is True

    def test_validate_ca_valid(self):
        for val in FEATURE_VALID_VALUES["ca"]:
            ok, msg = validate_ca(val)
            assert ok is True

    def test_validate_thal_valid(self):
        for val in FEATURE_VALID_VALUES["thal"]:
            ok, msg = validate_thal(val)
            assert ok is True


# ─────────────────────────────────────────────────────────────────────────────
# Patient Dict Validation
# ─────────────────────────────────────────────────────────────────────────────

class TestPatientDict:
    def test_validate_patient_valid(self, valid_patient):
        ok, errors = validate_patient(valid_patient)
        assert ok is True
        assert errors == {}

    def test_validate_patient_missing_field(self, valid_patient):
        del valid_patient["age"]
        ok, errors = validate_patient(valid_patient)
        assert ok is False
        assert "age" in errors

    def test_validate_patient_invalid_value(self, valid_patient):
        valid_patient["age"] = 200
        ok, errors = validate_patient(valid_patient)
        assert ok is False
        assert "age" in errors

    def test_validate_patient_check_structure_missing(self, valid_patient):
        # FIXED: check_structure parameter
        del valid_patient["sex"]
        ok, errors = validate_patient(valid_patient, check_structure=True)
        assert ok is False
        assert "sex" in errors
        assert "missing" in errors["sex"].lower()

    def test_validate_patient_check_structure_all_present(self, valid_patient):
        ok, errors = validate_patient(valid_patient, check_structure=True)
        assert ok is True

    def test_validate_single_field_valid(self):
        ok, msg = validate_single_field("age", 55)
        assert ok is True

    def test_validate_single_field_invalid(self):
        ok, msg = validate_single_field("age", 200)
        assert ok is False

    def test_validate_single_field_unknown(self):
        ok, msg = validate_single_field("unknown_field", "value")
        assert ok is True  # Unknown fields pass through


# ─────────────────────────────────────────────────────────────────────────────
# Auth Validators
# ─────────────────────────────────────────────────────────────────────────────

class TestAuthValidators:
    def test_validate_email_valid(self):
        ok, msg = validate_email("user@example.com")
        assert ok is True

    def test_validate_email_invalid(self):
        ok, msg = validate_email("invalid-email")
        assert ok is False
        assert "valid email" in msg.lower()

    def test_validate_password_valid(self):
        ok, msg = validate_password("SecurePass123!")
        assert ok is True

    def test_validate_password_too_short(self):
        ok, msg = validate_password("Short1!")
        assert ok is False
        assert f"at least {PASSWORD_MIN_LENGTH}" in msg.lower()

    def test_validate_password_no_uppercase(self):
        ok, msg = validate_password("nouppercase1!")
        assert ok is False
        assert "uppercase" in msg.lower()

    def test_validate_password_no_digit(self):
        ok, msg = validate_password("NoDigitPass!")
        assert ok is False
        assert "digit" in msg.lower()

    def test_validate_password_no_special(self):
        # FIXED: Special char required
        ok, msg = validate_password("NoSpecial1")
        assert ok is False
        assert "special character" in msg.lower()

    def test_validate_username_valid(self):
        ok, msg = validate_username("john_doe123")
        assert ok is True

    def test_validate_username_too_short(self):
        ok, msg = validate_username("ab")
        assert ok is False
        assert "3–30" in msg

    def test_validate_username_invalid_chars(self):
        ok, msg = validate_username("user@name")
        assert ok is False
        assert "letters, numbers, and underscores" in msg.lower()

    def test_validate_role_valid(self):
        for role in ROLES.values():
            ok, msg = validate_role(role)
            assert ok is True

    def test_validate_role_invalid(self):
        ok, msg = validate_role("superuser")
        assert ok is False
        assert "invalid role" in msg.lower()

    def test_validate_login_attempt_ok(self):
        ok, msg = validate_login_attempt(3)
        assert ok is True

    def test_validate_login_attempt_locked(self):
        ok, msg = validate_login_attempt(MAX_LOGIN_ATTEMPTS)
        assert ok is False
        assert "locked" in msg.lower()

    def test_validate_api_key_valid(self):
        ok, msg = validate_api_key("sk-1234567890abcdefghij")
        assert ok is True

    def test_validate_api_key_too_short(self):
        ok, msg = validate_api_key("short")
        assert ok is False
        assert "too short" in msg.lower()

    def test_validate_api_key_spaces(self):
        ok, msg = validate_api_key("sk-12345 67890abcdefghij")
        assert ok is False
        assert "spaces" in msg.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Data File Validators
# ─────────────────────────────────────────────────────────────────────────────

class TestDataFileValidators:
    def test_validate_csv_columns_valid(self):
        cols = FEATURE_COLUMNS + ["target"]
        ok, msg = validate_csv_columns(cols)
        assert ok is True

    def test_validate_csv_columns_missing(self):
        cols = ["age", "sex"]  # Missing many
        ok, msg = validate_csv_columns(cols)
        assert ok is False
        assert "missing" in msg.lower()

    def test_validate_sample_input_valid(self, valid_patient):
        ok, errors = validate_sample_input(valid_patient)
        assert ok is True

    def test_validate_sample_input_not_dict(self):
        ok, errors = validate_sample_input("not a dict")
        assert ok is False
        assert "_" in errors
        assert "JSON object" in errors["_"]


# ─────────────────────────────────────────────────────────────────────────────
# Batch Validation
# ─────────────────────────────────────────────────────────────────────────────

class TestBatchValidation:
    def test_validate_batch_all_valid(self, valid_patient):
        records = [valid_patient, valid_patient.copy()]
        valid, invalid = validate_batch(records)
        assert len(valid) == 2
        assert len(invalid) == 0

    def test_validate_batch_mixed(self, valid_patient):
        invalid_patient = valid_patient.copy()
        invalid_patient["age"] = 200
        records = [valid_patient, invalid_patient]
        valid, invalid = validate_batch(records)
        assert len(valid) == 1
        assert len(invalid) == 1
        assert invalid[0][0] == 1  # Index 1
        assert "age" in invalid[0][1]

    def test_validate_batch_all_invalid(self, valid_patient):
        invalid_patient = valid_patient.copy()
        invalid_patient["age"] = 200
        records = [invalid_patient, invalid_patient.copy()]
        valid, invalid = validate_batch(records)
        assert len(valid) == 0
        assert len(invalid) == 2


# ─────────────────────────────────────────────────────────────────────────────
# Formatting Helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestFormattingHelpers:
    def test_errors_to_str_simple(self):
        errors = {"age": "Age must be positive", "chol": "Out of range"}
        result = errors_to_str(errors)
        assert "Age (years): Age must be positive" in result
        assert "Serum Cholesterol" in result

    def test_errors_to_str_generic_key(self):
        # FIXED: _ key handling
        errors = {"_": "Generic error message"}
        result = errors_to_str(errors)
        assert "• Generic error message" in result
        assert "_" not in result

    def test_format_validation_errors_alias(self):
        errors = {"age": "Invalid"}
        result1 = errors_to_str(errors)
        result2 = format_validation_errors(errors)
        assert result1 == result2