# =============================================================================
# ui/role_dashboard.py
# AuraEcho+ — Role-Based Dashboard Router
#
# Responsibility:
#     Render role-specific dashboard layouts based on user permissions.
#     Provides tailored views for Doctors, Nurses, Admins, and Viewers.
#
# Role Views:
#     • Doctor    : Full access — risk, AI, similar cases, save, export, analytics
#     • Nurse     : Vitals entry, view results, limited actions
#     • Admin     : System status, user management, analytics
#     • Viewer    : Read-only results
#
# Public API:
#     render_role_dashboard(user_role, patient_data, risk_result, 
#                           ai_response, similar_cases) → None
# =============================================================================

import streamlit as st
from typing import Any, Dict, List, Optional

from utils.constants import (
    APP_NAME,
    ROLE_DOCTOR,
    ROLE_NURSE,
    ROLE_ADMIN,
    ROLE_VIEWER,
    UI_PRIMARY_COLOR,
    UI_SUCCESS_COLOR,
    UI_WARNING_COLOR,
    UI_DANGER_COLOR,
    ROLE_PERMISSIONS,
)
from utils.helpers import get_logger, now_str
from services.auth_service import user_has_permission, list_users
from ui.dashboard import render_dashboard
from ui.results_panel import render_results_panel
from ui.system_status import render_system_status
from ui.data_entry_form import render_data_entry_form

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# CSS Styles
# ─────────────────────────────────────────────

ROLE_CSS = """
<style>
    /* Role badge */
    .role-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.35rem 0.85rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        color: white;
        margin-bottom: 1rem;
    }
    .role-doctor {
        background-color: #1a73e8;
    }
    .role-nurse {
        background-color: #2ecc71;
    }
    .role-admin {
        background-color: #9b59b6;
    }
    .role-viewer {
        background-color: #7f8c8d;
    }
    
    /* Role header */
    .role-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.5rem;
        flex-wrap: wrap;
        gap: 0.5rem;
    }
    .role-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1a1a2e;
    }
    
    /* Permission denied */
    .permission-denied {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        border-radius: 6px;
        color: #721c24;
        margin-bottom: 1rem;
    }
    
    /* Section card */
    .role-section {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid #e8f0fe;
    }
    
    /* User table */
    .user-table {
        font-size: 0.9rem;
    }
    .user-table th {
        background-color: #f8f9fa;
        font-weight: 600;
    }
</style>
"""


# ─────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────

def _get_role_class(role: str) -> str:
    """Get CSS class for role badge."""
    role_map = {
        ROLE_DOCTOR: "role-doctor",
        ROLE_NURSE: "role-nurse",
        ROLE_ADMIN: "role-admin",
        ROLE_VIEWER: "role-viewer",
    }
    return role_map.get(role, "role-viewer")


def _get_role_label(role: str) -> str:
    """Get display label for role."""
    labels = {
        ROLE_DOCTOR: "👨‍⚕️ Doctor",
        ROLE_NURSE: "👩‍⚕️ Nurse",
        ROLE_ADMIN: "🔧 Admin",
        ROLE_VIEWER: "👁️ Viewer",
    }
    return labels.get(role, f"👤 {role.title()}")


def _render_role_header(role: str, full_name: str) -> None:
    """Render role-aware header."""
    role_class = _get_role_class(role)
    role_label = _get_role_label(role)
    
    st.markdown(
        f"""
        <div class="role-header">
            <div class="role-title">🫀 {APP_NAME} Dashboard</div>
            <div class="role-badge {role_class}">
                {role_label}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.caption(f"Welcome, {full_name} · {now_str()}")
    st.markdown("---")


def _render_permission_denied(permission: str) -> None:
    """Render permission denied message."""
    st.markdown(
        f"""
        <div class="permission-denied">
            ⚠️ <strong>Access Denied</strong><br>
            You don't have permission to view this content.<br>
            Required permission: <code>{permission}</code>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# Role-Specific Renderers
# ─────────────────────────────────────────────

def _render_doctor_dashboard(
    patient_data: Dict[str, Any],
    risk_result: Optional[Dict[str, Any]],
    ai_response: Optional[Dict[str, Any]],
    similar_cases: Optional[List[Dict[str, Any]]],
    user_role: str,
) -> None:
    """
    Render full dashboard for Doctor role.
    Complete access to all features.
    """
    # Tabs for organization
    tab_results, tab_dashboard, tab_analytics = st.tabs([
        "🔍 Diagnosis",
        "📊 Patient Dashboard",
        "📈 Analytics",
    ])
    
    with tab_results:
        render_results_panel(
            patient_data=patient_data,
            risk_result=risk_result,
            ai_response=ai_response,
            similar_cases=similar_cases,
            user_role=user_role,
        )
    
    with tab_dashboard:
        render_dashboard(
            patient_data=patient_data,
            risk_result=risk_result,
        )
    
    with tab_analytics:
        st.markdown("### 📈 Patient Analytics")
        st.info("Analytics dashboard coming soon — patient history trends, cohort analysis.")
        
        # Quick stats
        if risk_result:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Risk Score",
                    f"{risk_result.get('disease_prob', 0) * 100:.1f}%",
                )
            with col2:
                st.metric(
                    "Confidence",
                    f"{risk_result.get('confidence_pct', 0):.1f}%",
                )
            with col3:
                st.metric(
                    "Similar Cases",
                    len(similar_cases) if similar_cases else 0,
                )


def _render_nurse_dashboard(
    patient_data: Dict[str, Any],
    risk_result: Optional[Dict[str, Any]],
    ai_response: Optional[Dict[str, Any]],
    similar_cases: Optional[List[Dict[str, Any]]],
    user_role: str,
) -> None:
    """
    Render dashboard for Nurse role.
    Focus on vitals entry and viewing results.
    Limited actions (no AI regenerate, limited save).
    """
    col_form, col_results = st.columns([1, 2])
    
    with col_form:
        st.markdown("### 📝 Patient Vitals")
        
        # Check permission
        if user_has_permission({"role": user_role}, "enter_vitals"):
            form_result = render_data_entry_form(
                form_key="nurse_form",
                initial_data=patient_data,
                show_submit=True,
                user_role=user_role,
            )
            
            if form_result.get("submitted"):
                st.session_state["nurse_submitted"] = True
                st.toast("✅ Vitals submitted for analysis", icon="✅")
        else:
            _render_permission_denied("enter_vitals")
    
    with col_results:
        st.markdown("### 🔍 Assessment Results")
        
        if risk_result:
            # Simplified results panel for nurses
            render_results_panel(
                patient_data=patient_data,
                risk_result=risk_result,
                ai_response=ai_response,
                similar_cases=similar_cases,
                user_role=user_role,
            )
        else:
            st.info(
                "💡 Submit patient vitals to generate risk assessment. "
                "Results will appear here after analysis."
            )


def _render_admin_dashboard(
    user_role: str,
) -> None:
    """
    Render dashboard for Admin role.
    System status, user management, analytics.
    """
    tab_status, tab_users, tab_settings = st.tabs([
        "🔧 System Status",
        "👥 User Management",
        "⚙️ Settings",
    ])
    
    with tab_status:
        render_system_status(user_role=user_role)
    
    with tab_users:
        st.markdown("### 👥 User Management")
        
        if user_has_permission({"role": user_role}, "manage_users"):
            users = list_users()
            
            if users:
                # Display users table
                st.dataframe(
                    users,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "username": "Username",
                        "role": "Role",
                        "full_name": "Full Name",
                        "email": "Email",
                        "created_at": "Created",
                        "last_login": "Last Login",
                        "failed_attempts": "Failed Attempts",
                    },
                )
                
                # User count stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Users", len(users))
                with col2:
                    doctors = sum(1 for u in users if u.get("role") == ROLE_DOCTOR)
                    st.metric("Doctors", doctors)
                with col3:
                    nurses = sum(1 for u in users if u.get("role") == ROLE_NURSE)
                    st.metric("Nurses", nurses)
            else:
                st.info("No users found in the system.")
        else:
            _render_permission_denied("manage_users")
    
    with tab_settings:
        st.markdown("### ⚙️ System Settings")
        
        if user_has_permission({"role": user_role}, "system_settings"):
            st.markdown("#### Application Configuration")
            
            col1, col2 = st.columns(2)
            with col1:
                st.text_input("App Name", value=APP_NAME, disabled=True)
            with col2:
                st.text_input("Version", value="1.0.0", disabled=True)
            
            st.markdown("#### Feature Flags")
            st.toggle("Enable AI Diagnosis", value=True, disabled=True)
            st.toggle("Enable Cloud Sync", value=True, disabled=True)
            st.toggle("Enable Audit Logging", value=True, disabled=True)
            
            st.info("⚠️ Settings are currently read-only. Edit configuration files to change.")
        else:
            _render_permission_denied("system_settings")


def _render_viewer_dashboard(
    patient_data: Dict[str, Any],
    risk_result: Optional[Dict[str, Any]],
    ai_response: Optional[Dict[str, Any]],
    similar_cases: Optional[List[Dict[str, Any]]],
    user_role: str,
) -> None:
    """
    Render dashboard for Viewer role.
    Read-only access to results.
    """
    st.markdown("### 👁️ Patient Assessment")
    
    if user_has_permission({"role": user_role}, "view_patient"):
        if risk_result:
            # Render read-only results
            render_results_panel(
                patient_data=patient_data,
                risk_result=risk_result,
                ai_response=ai_response,
                similar_cases=similar_cases,
                user_role=user_role,
            )
            
            # Viewer disclaimer
            st.warning(
                "📋 **Viewer Mode**: You have read-only access. "
                "Contact an administrator if you need edit permissions."
            )
        else:
            st.info(
                "No patient assessment available. "
                "Please request analysis from a doctor or nurse."
            )
    else:
        _render_permission_denied("view_patient")


# ─────────────────────────────────────────────
# Main Render Function
# ─────────────────────────────────────────────

def render_role_dashboard(
    user_role: Optional[str] = None,
    full_name: Optional[str] = None,
    patient_data: Optional[Dict[str, Any]] = None,
    risk_result: Optional[Dict[str, Any]] = None,
    ai_response: Optional[Dict[str, Any]] = None,
    similar_cases: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """
    Render the role-based dashboard.

    Parameters
    ----------
    user_role     : str — current user role
    full_name     : str — user's full name
    patient_data  : dict — patient input data
    risk_result   : dict — RiskResult.to_dict()
    ai_response   : dict — AIResponse.to_dict()
    similar_cases : list — SimilarCase.to_dict() list
    """
    # Inject CSS
    st.markdown(ROLE_CSS, unsafe_allow_html=True)
    
    # Handle missing role
    if not user_role:
        st.error("⚠️ No user role detected. Please log in.")
        return
    
    # Normalize role
    role = user_role.lower()
    name = full_name or user_role
    
    # Render header
    _render_role_header(role, name)
    
    # Dispatch to role-specific renderer
    if role == ROLE_DOCTOR:
        _render_doctor_dashboard(
            patient_data or {},
            risk_result,
            ai_response,
            similar_cases,
            role,
        )
    
    elif role == ROLE_NURSE:
        _render_nurse_dashboard(
            patient_data or {},
            risk_result,
            ai_response,
            similar_cases,
            role,
        )
    
    elif role == ROLE_ADMIN:
        _render_admin_dashboard(role)
    
    elif role == ROLE_VIEWER:
        _render_viewer_dashboard(
            patient_data or {},
            risk_result,
            ai_response,
            similar_cases,
            role,
        )
    
    else:
        # Unknown role — render viewer as fallback
        st.warning(f"⚠️ Unknown role '{role}'. Rendering viewer dashboard.")
        _render_viewer_dashboard(
            patient_data or {},
            risk_result,
            ai_response,
            similar_cases,
            role,
        )
    
    # Debug info
    with st.expander("🔧 Role Debug", expanded=False):
        st.json({
            "user_role": user_role,
            "full_name": full_name,
            "permissions": ROLE_PERMISSIONS.get(role, []),
            "has_patient_data": bool(patient_data),
            "has_risk_result": bool(risk_result),
            "has_ai_response": bool(ai_response),
            "similar_cases_count": len(similar_cases) if similar_cases else 0,
        })


def get_role_capabilities(user_role: str) -> Dict[str, Any]:
    """
    Get capabilities summary for a role.
    Useful for onboarding or help pages.
    """
    role = user_role.lower()
    permissions = ROLE_PERMISSIONS.get(role, [])
    
    capabilities = {
        "role": role,
        "label": _get_role_label(role),
        "permissions": permissions,
        "can_view_diagnosis": "view_diagnosis" in permissions,
        "can_edit_patient": "edit_patient" in permissions,
        "can_manage_users": "manage_users" in permissions,
        "can_view_analytics": "view_analytics" in permissions,
        "can_export_reports": "export_reports" in permissions,
    }
    
    return capabilities