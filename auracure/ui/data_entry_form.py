# =============================================================================
# ui/data_entry_form.py
# AuraEcho+ — Structured Patient Data Entry Form
#
# Responsibility:
#     Render a comprehensive patient data entry form with real-time validation,
#     clear error feedback, and structured sections. Used for dedicated data
#     entry pages and as a reusable component.
#
# Features:
#     • Real-time field validation with error messages
#     • Sectioned layout (Demographics, Vitals, Clinical, Test Results)
#     • Load sample / Clear form controls
#     • Role-aware submit permissions
#     • Session state persistence
#     • Validation summary before submit
#
# Public API:
#     render_data_entry_form(form_key, initial_data, show_submit) → dict
# =============================================================================

import streamlit as st
from typing import Any, Dict, List, Optional, Tuple

from utils.constants import (
    FEATURE_COLUMNS,
    FEATURE_LABELS,
    FEATURE_RANGES,
    FEATURE_VALID_VALUES,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    CHEST_PAIN_LABELS,
    THAL_LABELS,
    SLOPE_LABELS,
    RESTECG_LABELS,
    UI_PRIMARY_COLOR,
    UI_SUCCESS_COLOR,
    UI_WARNING_COLOR,
    UI_DANGER_COLOR,
    ROLE_PERMISSIONS,
)
from utils.validators import (
    validate_patient,
    validate_single_field,
    validate_patient_name,
    errors_to_str,
)
from utils.helpers import get_logger, load_sample_input, patient_to_display
from services.auth_service import user_has_permission

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# CSS Styles
# ─────────────────────────────────────────────

FORM_CSS = """
<style>
    /* Form section */
    .form-section {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1a73e8;
    }
    .form-section-title {
        font-size: 1rem;
        font-weight: 600;
        color: #1a1a2e;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Field error */
    .field-error {
        color: #e74c3c;
        font-size: 0.75rem;
        margin-top: 0.25rem;
        display: flex;
        align-items: center;
        gap: 0.25rem;
    }
    
    /* Field valid */
    .field-valid {
        color: #2ecc71;
        font-size: 0.75rem;
        margin-top: 0.25rem;
    }
    
    /* Validation summary */
    .validation-summary {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 0.75rem;
        border-radius: 6px;
        margin-bottom: 1rem;
    }
    .validation-error {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
    }
    .validation-success {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
    
    /* Help text */
    .help-text {
        color: #5f6368;
        font-size: 0.75rem;
        margin-top: 0.25rem;
    }
</style>
"""


# ─────────────────────────────────────────────
# Field Render Helpers
# ─────────────────────────────────────────────

def _render_text_field(
    label: str,
    value: str,
    key: str,
    validator: callable,
    placeholder: str = "",
) -> Tuple[str, Optional[str]]:
    """Render a text field with validation."""
    new_value = st.text_input(
        label,
        value=value,
        key=key,
        placeholder=placeholder,
    )
    
    # Validate
    if new_value:
        is_valid, error = validator(new_value)
        if not is_valid:
            st.markdown(
                f'<div class="field-error">⚠️ {error}</div>',
                unsafe_allow_html=True,
            )
            return new_value, error
        else:
            st.markdown(
                '<div class="field-valid">✓ Valid</div>',
                unsafe_allow_html=True,
            )
    
    return new_value, None


def _render_number_field(
    label: str,
    value: float,
    key: str,
    feature: str,
    step: float = 1.0,
) -> Tuple[float, Optional[str]]:
    """Render a number field with range validation."""
    min_val, max_val = FEATURE_RANGES.get(feature, (0, 1000))
    
    new_value = st.number_input(
        label,
        min_value=min_val,
        max_value=max_val,
        value=value,
        key=key,
        step=step,
    )
    
    # Validate
    is_valid, error = validate_single_field(feature, new_value)
    if not is_valid:
        st.markdown(
            f'<div class="field-error">⚠️ {error}</div>',
            unsafe_allow_html=True,
        )
        return new_value, error
    
    return new_value, None


def _render_select_field(
    label: str,
    value: int,
    key: str,
    feature: str,
    options: Dict[int, str],
) -> Tuple[int, Optional[str]]:
    """Render a select field with validation."""
    # Reverse mapping for display
    value_to_label = {v: k for k, v in options.items()}
    
    selected_label = st.selectbox(
        label,
        options=list(options.values()),
        index=list(options.values()).index(options.get(value, list(options.values())[0])),
        key=key,
    )
    
    # Get numeric value
    new_value = value_to_label[selected_label]
    
    # Validate
    is_valid, error = validate_single_field(feature, new_value)
    if not is_valid:
        st.markdown(
            f'<div class="field-error">⚠️ {error}</div>',
            unsafe_allow_html=True,
        )
        return new_value, error
    
    return new_value, None


def _render_radio_field(
    label: str,
    value: int,
    key: str,
    feature: str,
    options: Dict[int, str],
    horizontal: bool = True,
) -> Tuple[int, Optional[str]]:
    """Render a radio field with validation."""
    value_to_label = {v: k for k, v in options.items()}
    
    selected_label = st.radio(
        label,
        options=list(options.values()),
        index=list(options.values()).index(options.get(value, list(options.values())[0])),
        key=key,
        horizontal=horizontal,
    )
    
    new_value = value_to_label[selected_label]
    
    is_valid, error = validate_single_field(feature, new_value)
    if not is_valid:
        st.markdown(
            f'<div class="field-error">⚠️ {error}</div>',
            unsafe_allow_html=True,
        )
        return new_value, error
    
    return new_value, None


# ─────────────────────────────────────────────
# Section Renderers
# ─────────────────────────────────────────────

def _render_demographics_section(
    form_data: Dict[str, Any],
    errors: Dict[str, str],
    form_key: str,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Render patient demographics section."""
    st.markdown(
        """
        <div class="form-section">
            <div class="form-section-title">👤 Patient Demographics</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Name
    name, name_err = _render_text_field(
        "Patient Name",
        form_data.get("name", ""),
        f"{form_key}_name",
        validate_patient_name,
        placeholder="Enter full name",
    )
    form_data["name"] = name
    if name_err:
        errors["name"] = name_err
    else:
        errors.pop("name", None)
    
    # Age and Sex in columns
    col1, col2 = st.columns(2)
    
    with col1:
        age, age_err = _render_number_field(
            "Age (years)",
            form_data.get("age", 50),
            f"{form_key}_age",
            "age",
            step=1,
        )
        form_data["age"] = int(age)
        if age_err:
            errors["age"] = age_err
        else:
            errors.pop("age", None)
    
    with col2:
        sex, sex_err = _render_radio_field(
            "Sex",
            form_data.get("sex", 1),
            f"{form_key}_sex",
            "sex",
            {0: "Female", 1: "Male"},
        )
        form_data["sex"] = sex
        if sex_err:
            errors["sex"] = sex_err
        else:
            errors.pop("sex", None)
    
    return form_data, errors


def _render_vitals_section(
    form_data: Dict[str, Any],
    errors: Dict[str, str],
    form_key: str,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Render vital signs section."""
    st.markdown(
        """
        <div class="form-section">
            <div class="form-section-title">💓 Vital Signs</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Resting BP
        trestbps, err = _render_number_field(
            "Resting BP (mm Hg)",
            form_data.get("trestbps", 120),
            f"{form_key}_trestbps",
            "trestbps",
            step=1,
        )
        form_data["trestbps"] = int(trestbps)
        if err:
            errors["trestbps"] = err
        else:
            errors.pop("trestbps", None)
        
        # Cholesterol
        chol, err = _render_number_field(
            "Cholesterol (mg/dl)",
            form_data.get("chol", 200),
            f"{form_key}_chol",
            "chol",
            step=1,
        )
        form_data["chol"] = int(chol)
        if err:
            errors["chol"] = err
        else:
            errors.pop("chol", None)
    
    with col2:
        # Max Heart Rate
        thalach, err = _render_number_field(
            "Max Heart Rate (bpm)",
            form_data.get("thalach", 150),
            f"{form_key}_thalach",
            "thalach",
            step=1,
        )
        form_data["thalach"] = int(thalach)
        if err:
            errors["thalach"] = err
        else:
            errors.pop("thalach", None)
        
        # Fasting Blood Sugar
        fbs, err = _render_select_field(
            "Fasting Blood Sugar > 120 mg/dl",
            form_data.get("fbs", 0),
            f"{form_key}_fbs",
            "fbs",
            {0: "No", 1: "Yes"},
        )
        form_data["fbs"] = fbs
        if err:
            errors["fbs"] = err
        else:
            errors.pop("fbs", None)
    
    return form_data, errors


def _render_clinical_section(
    form_data: Dict[str, Any],
    errors: Dict[str, str],
    form_key: str,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Render clinical findings section."""
    st.markdown(
        """
        <div class="form-section">
            <div class="form-section-title">🩺 Clinical Findings</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Chest Pain Type
    cp, err = _render_select_field(
        "Chest Pain Type",
        form_data.get("cp", 0),
        f"{form_key}_cp",
        "cp",
        CHEST_PAIN_LABELS,
    )
    form_data["cp"] = cp
    if err:
        errors["cp"] = err
    else:
        errors.pop("cp", None)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Resting ECG
        restecg, err = _render_select_field(
            "Resting ECG",
            form_data.get("restecg", 0),
            f"{form_key}_restecg",
            "restecg",
            RESTECG_LABELS,
        )
        form_data["restecg"] = restecg
        if err:
            errors["restecg"] = err
        else:
            errors.pop("restecg", None)
        
        # Exercise Angina
        exang, err = _render_select_field(
            "Exercise-Induced Angina",
            form_data.get("exang", 0),
            f"{form_key}_exang",
            "exang",
            {0: "No", 1: "Yes"},
        )
        form_data["exang"] = exang
        if err:
            errors["exang"] = err
        else:
            errors.pop("exang", None)
    
    with col2:
        # ST Depression
        oldpeak, err = _render_number_field(
            "ST Depression (oldpeak)",
            form_data.get("oldpeak", 0.0),
            f"{form_key}_oldpeak",
            "oldpeak",
            step=0.1,
        )
        form_data["oldpeak"] = float(oldpeak)
        if err:
            errors["oldpeak"] = err
        else:
            errors.pop("oldpeak", None)
        
        # ST Slope
        slope, err = _render_select_field(
            "ST Slope",
            form_data.get("slope", 0),
            f"{form_key}_slope",
            "slope",
            SLOPE_LABELS,
        )
        form_data["slope"] = slope
        if err:
            errors["slope"] = err
        else:
            errors.pop("slope", None)
    
    return form_data, errors


def _render_test_results_section(
    form_data: Dict[str, Any],
    errors: Dict[str, str],
    form_key: str,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Render test results section."""
    st.markdown(
        """
        <div class="form-section">
            <div class="form-section-title">🔬 Test Results</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Major Vessels
        ca, err = _render_select_field(
            "Major Vessels (Fluoroscopy)",
            form_data.get("ca", 0),
            f"{form_key}_ca",
            "ca",
            {0: "0", 1: "1", 2: "2", 3: "3"},
        )
        form_data["ca"] = ca
        if err:
            errors["ca"] = err
        else:
            errors.pop("ca", None)
    
    with col2:
        # Thalassemia
        thal, err = _render_select_field(
            "Thalassemia",
            form_data.get("thal", 1),
            f"{form_key}_thal",
            "thal",
            THAL_LABELS,
        )
        form_data["thal"] = thal
        if err:
            errors["thal"] = err
        else:
            errors.pop("thal", None)
    
    # Symptoms (optional)
    symptoms = st.text_area(
        "Symptoms (optional)",
        value=form_data.get("symptoms", ""),
        key=f"{form_key}_symptoms",
        placeholder="Describe any symptoms or clinical notes...",
        height=80,
    )
    form_data["symptoms"] = symptoms
    
    return form_data, errors


def _render_validation_summary(errors: Dict[str, str]) -> bool:
    """Render validation summary and return if form is valid."""
    if not errors:
        st.markdown(
            """
            <div class="validation-summary validation-success">
                ✓ All fields are valid. Ready to submit.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return True
    
    st.markdown(
        f"""
        <div class="validation-summary validation-error">
            ⚠️ Please correct the following errors:
            <br>{errors_to_str(errors).replace(chr(10), '<br>')}
        </div>
        """,
        unsafe_allow_html=True,
    )
    return False


# ─────────────────────────────────────────────
# Main Render Function
# ─────────────────────────────────────────────

def render_data_entry_form(
    form_key: str = "patient_form",
    initial_data: Optional[Dict[str, Any]] = None,
    show_submit: bool = True,
    user_role: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Render the complete patient data entry form.

    Parameters
    ----------
    form_key     : str — unique key for session state
    initial_data : dict — initial form values (optional)
    show_submit  : bool — whether to show submit button
    user_role    : str — current user role for permission checks

    Returns
    -------
    dict with form state:
        {
            "data": dict,
            "errors": dict,
            "is_valid": bool,
            "submitted": bool,
        }
    """
    # Inject CSS
    st.markdown(FORM_CSS, unsafe_allow_html=True)
    
    # Initialize session state
    state_key = f"{form_key}_state"
    if state_key not in st.session_state:
        st.session_state[state_key] = {
            "data": initial_data or {},
            "errors": {},
            "submitted": False,
        }
    
    state = st.session_state[state_key]
    form_data = state["data"]
    errors = state["errors"]
    
    # Load sample button
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("📋 Load Sample", key=f"{form_key}_load_sample"):
            sample = load_sample_input()
            if sample:
                st.session_state[state_key]["data"] = sample
                st.session_state[state_key]["errors"] = {}
                st.rerun()
    
    with col2:
        if st.button("🗑️ Clear Form", key=f"{form_key}_clear"):
            st.session_state[state_key] = {
                "data": {},
                "errors": {},
                "submitted": False,
            }
            st.rerun()
    
    st.markdown("---")
    
    # Render sections
    form_data, errors = _render_demographics_section(form_data, errors, form_key)
    form_data, errors = _render_vitals_section(form_data, errors, form_key)
    form_data, errors = _render_clinical_section(form_data, errors, form_key)
    form_data, errors = _render_test_results_section(form_data, errors, form_key)
    
    # Update state
    state["data"] = form_data
    state["errors"] = errors
    
    # Validation summary
    is_valid = _render_validation_summary(errors)
    
    # Submit button
    submitted = False
    if show_submit:
        can_submit = (
            user_has_permission({"role": user_role}, "enter_vitals")
            if user_role else True
        )
        
        if st.button(
            "🔍 Analyze Patient",
            key=f"{form_key}_submit",
            use_container_width=True,
            type="primary",
            disabled=not is_valid or not can_submit,
        ):
            if is_valid and can_submit:
                state["submitted"] = True
                submitted = True
                st.toast("✅ Form submitted successfully", icon="✅")
            elif not can_submit:
                st.toast("⚠️ You don't have permission to submit", icon="⚠️")
    
    # Preview expander
    with st.expander("👁️ Preview Form Data", expanded=False):
        display_data = patient_to_display(form_data)
        for label, value in display_data.items():
            st.markdown(f"**{label}**: {value}")
    
    return {
        "data": form_data,
        "errors": errors,
        "is_valid": is_valid,
        "submitted": submitted,
    }


def get_form_data(form_key: str = "patient_form") -> Optional[Dict[str, Any]]:
    """
    Get current form data from session state.
    Returns None if form not initialized.
    """
    state_key = f"{form_key}_state"
    if state_key in st.session_state:
        return st.session_state[state_key].get("data")
    return None


def set_form_data(form_key: str, data: Dict[str, Any]) -> None:
    """
    Set form data in session state.
    """
    state_key = f"{form_key}_state"
    if state_key in st.session_state:
        st.session_state[state_key]["data"] = data
        st.session_state[state_key]["errors"] = {}
    else:
        st.session_state[state_key] = {
            "data": data,
            "errors": {},
            "submitted": False,
        }


def reset_form(form_key: str = "patient_form") -> None:
    """
    Reset form to empty state.
    """
    state_key = f"{form_key}_state"
    st.session_state[state_key] = {
        "data": {},
        "errors": {},
        "submitted": False,
    }