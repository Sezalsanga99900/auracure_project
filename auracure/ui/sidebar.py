# =============================================================================
# ui/sidebar.py
# AuraEcho+ — Streamlit Sidebar Component
#
# Responsibility:
#     Render the sidebar with patient input form, mode badge, sync status,
#     user session info, and navigation controls.
#
# Sections:
#     1. App Header & Mode Badge
#     2. User Session Info
#     3. Patient Input Form
#     4. Sync Status & Controls
#     5. Navigation
#     6. System Status (collapsible)
#
# Public API:
#     render_sidebar() → dict (sidebar state)
# =============================================================================

import streamlit as st
from typing import Any, Dict, Optional

from utils.constants import (
    APP_NAME,
    APP_VERSION,
    PAGE_ICON,
    MODE_ONLINE,
    MODE_OFFLINE,
    UI_PRIMARY_COLOR,
    UI_SUCCESS_COLOR,
    UI_WARNING_COLOR,
    UI_DANGER_COLOR,
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
)
from utils.helpers import (
    get_logger,
    format_mode,
    load_sample_input,
    now_str,
)
from core.mode_detector import (
    is_online,
    get_mode_label,
    get_connection_info,
    invalidate_cache,
)
from services.sync_service import (
    get_sync_status,
    force_sync,
    is_sync_active,
    start_auto_sync,
    stop_auto_sync,
)
from services.auth_service import (
    get_session_user,
    logout,
    user_has_permission,
)

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# CSS Styles
# ─────────────────────────────────────────────

SIDEBAR_CSS = """
<style>
    /* Sidebar header */
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1a1a2e;
        margin-bottom: 0.5rem;
    }
    
    /* Mode badge */
    .mode-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 500;
        margin-bottom: 1rem;
    }
    .mode-online {
        background-color: #d4edda;
        color: #155724;
    }
    .mode-offline {
        background-color: #fff3cd;
        color: #856404;
    }
    
    /* Section divider */
    .sidebar-divider {
        border-top: 1px solid #e0e0e0;
        margin: 1rem 0;
    }
    
    /* Status indicator */
    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 0.5rem;
    }
    .status-ok { background-color: #2ecc71; }
    .status-warn { background-color: #f39c12; }
    .status-error { background-color: #e74c3c; }
    
    /* Sync button */
    .sync-button {
        width: 100%;
        padding: 0.5rem;
        border-radius: 6px;
        font-size: 0.85rem;
    }
</style>
"""


# ─────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────

def _render_mode_badge() -> None:
    """Render the online/offline mode badge."""
    mode_info = format_mode(get_connection_info().get("mode", MODE_OFFLINE))
    
    badge_class = "mode-online" if mode_info["is_online"] else "mode-offline"
    
    st.markdown(
        f"""
        <div class="mode-badge {badge_class}">
            {mode_info["icon"]} {mode_info["label"]}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_user_section() -> None:
    """Render user session info if authenticated."""
    token = st.session_state.get("auth_token")
    if not token:
        return

    user = get_session_user(token)
    if not user:
        return

    role = user.get("role", "").upper()
    full_name = user.get("full_name", user.get("username", "User"))

    st.markdown("### 👤 User")
    st.markdown(f"**{full_name}**")
    st.caption(f"Role: {role}")

    # Logout button
    if st.sidebar.button("🚪 Logout", key="logout_btn", use_container_width=True):
        logout(token)
        st.session_state.clear()
        st.rerun()

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)


def _render_patient_form() -> Dict[str, Any]:
    """
    Render the patient input form.
    Returns dict of form values.
    """
    st.markdown("### 🫀 Patient Data")

    # Initialize form state
    if "patient_form" not in st.session_state:
        st.session_state.patient_form = {}

    form_data = st.session_state.patient_form

    # Load sample button
    if st.button("📋 Load Sample", key="load_sample", use_container_width=True):
        sample = load_sample_input()
        if sample:
            st.session_state.patient_form = sample
            st.rerun()

    # Clear button
    if st.button("🗑️ Clear Form", key="clear_form", use_container_width=True):
        st.session_state.patient_form = {}
        st.rerun()

    st.markdown("---")

    # Patient name
    form_data["name"] = st.text_input(
        "Patient Name",
        value=form_data.get("name", ""),
        key="input_name",
        placeholder="Enter patient name",
    )

    # Age
    form_data["age"] = st.number_input(
        "Age (years)",
        min_value=1,
        max_value=120,
        value=form_data.get("age", 50),
        key="input_age",
    )

    # Sex
    sex_options = {"Male": 1, "Female": 0}
    form_data["sex"] = st.radio(
        "Sex",
        options=list(sex_options.keys()),
        index=list(sex_options.values()).index(form_data.get("sex", 1)),
        key="input_sex",
        horizontal=True,
    )
    form_data["sex"] = sex_options[form_data["sex"]]

    # Chest Pain Type
    cp_labels = {v: k for k, v in CHEST_PAIN_LABELS.items()}
    form_data["cp"] = st.selectbox(
        "Chest Pain Type",
        options=list(CHEST_PAIN_LABELS.values()),
        index=cp_labels.get(form_data.get("cp", 0), 0),
        key="input_cp",
        format_func=lambda x: x,
    )
    form_data["cp"] = cp_labels[form_data["cp"]]

    # Resting BP
    form_data["trestbps"] = st.number_input(
        "Resting BP (mm Hg)",
        min_value=60,
        max_value=250,
        value=form_data.get("trestbps", 120),
        key="input_trestbps",
    )

    # Cholesterol
    form_data["chol"] = st.number_input(
        "Cholesterol (mg/dl)",
        min_value=50,
        max_value=700,
        value=form_data.get("chol", 200),
        key="input_chol",
    )

    # Fasting Blood Sugar
    form_data["fbs"] = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        options=["No", "Yes"],
        index=form_data.get("fbs", 0),
        key="input_fbs",
    )
    form_data["fbs"] = 1 if form_data["fbs"] == "Yes" else 0

    # Resting ECG
    ecg_labels = {v: k for k, v in RESTECG_LABELS.items()}
    form_data["restecg"] = st.selectbox(
        "Resting ECG",
        options=list(RESTECG_LABELS.values()),
        index=ecg_labels.get(form_data.get("restecg", 0), 0),
        key="input_restecg",
    )
    form_data["restecg"] = ecg_labels[form_data["restecg"]]

    # Max Heart Rate
    form_data["thalach"] = st.number_input(
        "Max Heart Rate (bpm)",
        min_value=50,
        max_value=250,
        value=form_data.get("thalach", 150),
        key="input_thalach",
    )

    # Exercise Angina
    form_data["exang"] = st.selectbox(
        "Exercise-Induced Angina",
        options=["No", "Yes"],
        index=form_data.get("exang", 0),
        key="input_exang",
    )
    form_data["exang"] = 1 if form_data["exang"] == "Yes" else 0

    # Oldpeak
    form_data["oldpeak"] = st.number_input(
        "ST Depression (oldpeak)",
        min_value=0.0,
        max_value=10.0,
        value=float(form_data.get("oldpeak", 0.0)),
        step=0.1,
        key="input_oldpeak",
    )

    # Slope
    slope_labels = {v: k for k, v in SLOPE_LABELS.items()}
    form_data["slope"] = st.selectbox(
        "ST Slope",
        options=list(SLOPE_LABELS.values()),
        index=slope_labels.get(form_data.get("slope", 0), 0),
        key="input_slope",
    )
    form_data["slope"] = slope_labels[form_data["slope"]]

    # CA (Major Vessels)
    form_data["ca"] = st.selectbox(
        "Major Vessels (0-3)",
        options=[0, 1, 2, 3],
        index=form_data.get("ca", 0),
        key="input_ca",
    )

    # Thalassemia
    thal_labels = {v: k for k, v in THAL_LABELS.items()}
    form_data["thal"] = st.selectbox(
        "Thalassemia",
        options=list(THAL_LABELS.values()),
        index=thal_labels.get(form_data.get("thal", 1), 0),
        key="input_thal",
    )
    form_data["thal"] = thal_labels[form_data["thal"]]

    # Symptoms (optional)
    form_data["symptoms"] = st.text_area(
        "Symptoms (optional)",
        value=form_data.get("symptoms", ""),
        key="input_symptoms",
        placeholder="Describe any symptoms...",
        height=80,
    )

    return form_data


def _render_sync_section() -> None:
    """Render sync status and controls."""
    st.markdown("### 🔄 Sync Status")

    sync_status = get_sync_status()
    online = sync_status.get("online", False)
    cloud_available = sync_status.get("cloud_available", False)
    pending = sync_status.get("pending_count", 0)
    failed = sync_status.get("failed_count", 0)

    # Status indicators
    if online and cloud_available:
        st.markdown(
            '<span class="status-dot status-ok"></span>Cloud Connected',
            unsafe_allow_html=True,
        )
    elif online:
        st.markdown(
            '<span class="status-dot status-warn"></span>Cloud Unavailable',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="status-dot status-error"></span>Offline',
            unsafe_allow_html=True,
        )

    # Pending/Failed counts
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Pending", pending)
    with col2:
        st.metric("Failed", failed)

    # Sync button
    if st.button(
        "🔄 Sync Now",
        key="sync_now_btn",
        use_container_width=True,
        disabled=not online,
    ):
        with st.spinner("Syncing..."):
            result = force_sync()
            if result.get("success"):
                st.toast(
                    f"✅ Synced {result.get('synced_count', 0)} items",
                    icon="✅",
                )
            else:
                st.toast(
                    f"❌ {result.get('message', 'Sync failed')}",
                    icon="❌",
                )
            st.rerun()

    # Auto-sync toggle
    auto_sync = is_sync_active()
    new_auto_sync = st.toggle(
        "Auto-Sync",
        value=auto_sync,
        key="auto_sync_toggle",
        help="Automatically sync when online",
    )
    if new_auto_sync != auto_sync:
        if new_auto_sync:
            start_auto_sync()
            st.toast("Auto-sync enabled", icon="🔄")
        else:
            stop_auto_sync()
            st.toast("Auto-sync disabled", icon="⏸️")
        st.rerun()

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)


def _render_system_status() -> None:
    """Render collapsible system status panel."""
    with st.expander("🔧 System Status", expanded=False):
        # Mode info
        conn_info = get_connection_info()
        st.markdown("**Connectivity**")
        st.caption(f"Mode: {conn_info.get('mode', 'unknown')}")
        st.caption(f"Latency: {conn_info.get('latency_ms', 'N/A')} ms")
        st.caption(f"Host: {conn_info.get('host', 'N/A')}")

        # Refresh button
        if st.button("🔄 Refresh", key="refresh_conn", use_container_width=True):
            invalidate_cache()
            st.rerun()

        # Sync stats
        sync_status = get_sync_status()
        st.markdown("**Sync**")
        st.caption(f"Cycles: {sync_status.get('cycles_run', 0)}")
        st.caption(f"Total Synced: {sync_status.get('total_synced', 0)}")
        st.caption(f"Last Sync: {sync_status.get('last_sync_time', 'Never')}")

        # App version
        st.markdown(f"**{APP_NAME}** v{APP_VERSION}")
        st.caption(f"Time: {now_str()}")


# ─────────────────────────────────────────────
# Main render function
# ─────────────────────────────────────────────

def render_sidebar() -> Dict[str, Any]:
    """
    Render the complete sidebar.

    Returns
    -------
    dict with sidebar state:
        {
            "patient_data": dict,
            "submit_clicked": bool,
        }
    """
    # Inject CSS
    st.markdown(SIDEBAR_CSS, unsafe_allow_html=True)

    # Header
    st.sidebar.markdown(
        f"""
        <div class="sidebar-header">
            {PAGE_ICON} {APP_NAME}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Mode badge
    _render_mode_badge()

    # User section
    _render_user_section()

    # Patient form
    patient_data = _render_patient_form()

    # Submit button
    submit_clicked = st.sidebar.button(
        "🔍 Analyze Patient",
        key="analyze_btn",
        use_container_width=True,
        type="primary",
    )

    st.sidebar.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # Sync section
    _render_sync_section()

    # System status
    _render_system_status()

    return {
        "patient_data": patient_data,
        "submit_clicked": submit_clicked,
    }