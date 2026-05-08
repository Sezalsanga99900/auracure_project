# =============================================================================
# app.py
# AuraEcho+ — Master Streamlit Entry Point
#
# Responsibility:
#     Orchestrate the entire application flow:
#     • Initialize services (DB, sync, auth)
#     • Manage authentication and session state
#     • Route to login or dashboard based on auth status
#     • Handle patient analysis (risk + similarity + AI)
#     • Route AI calls based on online/offline mode
#     • Render role-based dashboards
#
# Usage:
#     streamlit run app.py
# =============================================================================




import streamlit as st
import os
import sys
from typing import Any, Dict, Optional


# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# 1. PAGE CONFIGURATION (Must be the very first Streamlit command)
from utils.constants import PAGE_TITLE, PAGE_ICON, PAGE_LAYOUT

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=PAGE_LAYOUT,
    initial_sidebar_state="expanded",
)

# # Add project root to path for imports
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────────────────────────────────────────────────────
# Imports — Layer by Layer
# ─────────────────────────────────────────────────────────────────────────────

# Layer 1: Utils
from utils.constants import (
    APP_NAME,
    APP_VERSION,
    PAGE_ICON,
    PAGE_TITLE,
    PAGE_LAYOUT,
    ASSETS_CSS_PATH,
    UI_BACKGROUND_COLOR,
    ROLE_DOCTOR,
    ROLE_NURSE,
    ROLE_ADMIN,
)

from utils.helpers import get_logger, now_str, load_sample_input
from utils.validators import validate_patient, errors_to_str

# Layer 2: Core
from core.preprocess import preprocess_patient
from core.risk_model import predict_risk
from core.similarity import find_similar_cases, preload_reference_data
from core.mode_detector import is_online, get_mode

# Layer 3: AI + Database
from ai.offline_ai import analyze_patient as analyze_offline
from ai.online_ai import analyze_patient as analyze_online
from database.local_db import init_db, save_patient, save_prediction
from services.sync_service import init_sync_service, start_auto_sync

# Layer 4: Services
from services.auth_service import (
    init_auth_db,
    login,
    logout,
    validate_session,
    get_session_user,
    create_default_admin,
)

# Layer 5: UI
from ui.sidebar import render_sidebar
from ui.role_dashboard import render_role_dashboard
from ui.data_entry_form import render_data_entry_form

# ─────────────────────────────────────────────────────────────────────────────
# Logger
# ─────────────────────────────────────────────────────────────────────────────

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────────────────────



# ─────────────────────────────────────────────────────────────────────────────
# Load CSS
# ─────────────────────────────────────────────────────────────────────────────

def load_css() -> None:
    """Load global stylesheet."""
    base_dir=os.path.dirname(os.path.abspath(__file__))
    css_path=os.path.join(base_dir,"assets","style.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        logger.warning("CSS file not found at %s", css_path)

load_css()

# ─────────────────────────────────────────────────────────────────────────────
# Initialize Services
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def initialize_services() -> None:
    """
    Initialize all services once at app startup.
    Cached to prevent re-initialization on reruns.
    """
    logger.info("🚀 Initializing AuraEcho+ services...")
    
    try:
        # Database
        init_db()
        logger.info("✅ Local database initialized")
        
        # Auth
        init_auth_db()
        create_default_admin()
        logger.info("✅ Auth service initialized")
        
        # Sync
        init_sync_service()
        start_auto_sync()
        logger.info("✅ Sync service initialized")
        
        # Preload reference data for similarity engine
        preload_reference_data()
        logger.info("✅ Similarity engine preloaded")
        
        logger.info("✨ All services initialized successfully")
        
    except Exception as exc:
        logger.error("❌ Service initialization failed: %s", exc)
        st.error(f"Failed to initialize services: {exc}")
        st.stop()

# Run initialization
initialize_services()

# ─────────────────────────────────────────────────────────────────────────────
# Session State Management
# ─────────────────────────────────────────────────────────────────────────────

def init_session_state() -> None:
    """Initialize session state variables."""
    defaults = {
        "auth_token": None,
        "patient_data": {},
        "risk_result": None,
        "ai_response": None,
        "similar_cases": None,
        "analysis_loading": False,
        "regenerate_ai": False,
        "last_analysis_time": None,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ─────────────────────────────────────────────────────────────────────────────
# Authentication Flow
# ─────────────────────────────────────────────────────────────────────────────

def render_login_page() -> None:
    """Render login form."""
    st.markdown(
        f"""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 style="font-size: 2.5rem; color: #1a73e8;">{PAGE_ICON} {APP_NAME}</h1>
            <p style="color: #5f6368;">AI-Powered Cardiac Risk Assessment</p>
            <p style="font-size: 0.9rem; color: #95a5a6;">v{APP_VERSION}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        with st.form("login_form"):
            st.markdown("### 🔐 Sign In")
            
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            
            submitted = st.form_submit_button("Login", use_container_width=True, type="primary")
            
            if submitted:
                if not username or not password:
                    st.error("Please enter username and password.")
                    return
                
                token = login(username, password)
                
                if token:
                    st.session_state.auth_token = token
                    st.session_state.patient_data = {}
                    st.session_state.risk_result = None
                    st.session_state.ai_response = None
                    st.session_state.similar_cases = None
                    st.rerun()
                else:
                    st.error("Invalid username or password. Please try again.")
        
        # Help text
        st.info(
            "💡 First time? Use the default admin credentials shown in the console "
            "on first run, or contact your system administrator."
        )

def validate_auth() -> Optional[Dict[str, Any]]:
    """
    Validate current session and return user info.
    Returns None if not authenticated.
    """
    token = st.session_state.get("auth_token")
    
    if not token:
        return None
    
    session = validate_session(token)
    
    if not session:
        # Session expired or invalid
        st.session_state.auth_token = None
        st.warning("Session expired. Please log in again.")
        return None
    
    return {
        "username": session.get("username"),
        "role": session.get("role"),
        "full_name": session.get("full_name"),
    }

# ─────────────────────────────────────────────────────────────────────────────
# Analysis Orchestration
# ─────────────────────────────────────────────────────────────────────────────

def run_analysis(patient_data: Dict[str, Any]) -> None:
    """
    Run complete analysis pipeline:
    1. Validate patient data
    2. Predict risk
    3. Find similar cases
    4. Run AI analysis (mode-aware routing)
    5. Save results
    """
    st.session_state.analysis_loading = True
    
    try:
        # Validate
        ok, errors = validate_patient(patient_data)
        if not ok:
            st.error(f"Validation errors:\n{errors_to_str(errors)}")
            st.session_state.analysis_loading = False
            return
        
        # Risk prediction
        with st.spinner("🧠 Analyzing cardiac risk..."):
            risk_result = predict_risk(patient_data)
            st.session_state.risk_result = risk_result.to_dict()
            logger.info(
                "Risk prediction complete: level=%s, prob=%.3f",
                risk_result.risk_level, risk_result.disease_prob,
            )
        
        # Similar cases
        with st.spinner("🔍 Finding similar cases..."):
            similar_cases = find_similar_cases(patient_data, k=5)
            st.session_state.similar_cases = [c.to_dict() for c in similar_cases]
            logger.info("Found %d similar cases", len(similar_cases))
        
        # AI analysis with mode-aware routing
        mode = get_mode()
        logger.info("AI analysis mode: %s", mode)
        
        with st.spinner("🤖 Generating AI diagnosis..."):
            if mode == "online":
                # Try online first, fallback to offline
                ai_response = analyze_online(
                    patient_data,
                    risk_result=st.session_state.risk_result,
                    similar_cases=st.session_state.similar_cases,
                )
                
                # Fallback if online failed
                if not ai_response.success:
                    logger.warning("Online AI failed, falling back to offline")
                    ai_response = analyze_offline(
                        patient_data,
                        risk_result=st.session_state.risk_result,
                        similar_cases=st.session_state.similar_cases,
                    )
            else:
                # Offline mode
                ai_response = analyze_offline(
                    patient_data,
                    risk_result=st.session_state.risk_result,
                    similar_cases=st.session_state.similar_cases,
                )
            
            st.session_state.ai_response = ai_response.to_dict()
            logger.info(
                "AI analysis complete: source=%s, success=%s",
                ai_response.source, ai_response.success,
            )
        
        # Save to database
        try:
            patient_id = save_patient(patient_data)
            save_prediction(
                patient_id,
                st.session_state.risk_result,
                st.session_state.ai_response,
            )
            logger.info("Saved analysis to database: patient_id=%d", patient_id)
        except Exception as db_exc:
            logger.warning("Failed to save to database: %s", db_exc)
        
        st.session_state.last_analysis_time = now_str()
        st.toast("✅ Analysis complete!", icon="✅")
        
    except Exception as exc:
        logger.error("Analysis failed: %s", exc)
        st.error(f"Analysis failed: {exc}")
    
    finally:
        st.session_state.analysis_loading = False

# ─────────────────────────────────────────────────────────────────────────────
# Main Dashboard
# ─────────────────────────────────────────────────────────────────────────────

def render_dashboard(user: Dict[str, Any]) -> None:
    """Render the main dashboard for authenticated users."""
    
    # Render sidebar
    sidebar_state = render_sidebar()
    
    # Handle sidebar actions
    if sidebar_state.get("submit_clicked"):
        patient_data = sidebar_state.get("patient_data", {})
        if patient_data:
            run_analysis(patient_data)
            st.session_state.patient_data = patient_data
    
    # Handle AI regeneration
    if st.session_state.get("regenerate_ai"):
        st.session_state.regenerate_ai = False
        if st.session_state.patient_data:
            run_analysis(st.session_state.patient_data)
    
    # Get current results
    patient_data = st.session_state.get("patient_data", {})
    risk_result = st.session_state.get("risk_result")
    ai_response = st.session_state.get("ai_response")
    similar_cases = st.session_state.get("similar_cases")
    
    # Render role-based dashboard
    render_role_dashboard(
        user_role=user.get("role"),
        full_name=user.get("full_name"),
        patient_data=patient_data,
        risk_result=risk_result,
        ai_response=ai_response,
        similar_cases=similar_cases,
    )
    
    # Show loading state if analysis in progress
    if st.session_state.analysis_loading:
        with st.spinner("🔄 Running analysis..."):
            pass

# ─────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Main application entry point."""
    
    # Validate authentication
    user = validate_auth()
    
    if not user:
        # Not authenticated — show login
        render_login_page()
        return
    
    # Authenticated — show dashboard
    render_dashboard(user)
    
    # Debug info in expander
    if os.getenv("DEBUG", "").lower() == "true":
        with st.expander("🔧 Debug Info", expanded=False):
            st.json({
                "user": user,
                "mode": get_mode(),
                "session_state": {
                    k: v for k, v in st.session_state.items()
                    if k not in ["auth_token"]
                },
            })

if __name__ == "__main__":
    main()