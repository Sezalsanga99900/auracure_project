"""
app.py
──────
AuraEcho+ — Cardiac Clinical Decision Support System
Master entry point.

Run with:
    streamlit run app.py

Architecture:
    app.py is the orchestrator. It:
    1.  Loads environment variables (.env)
    2.  Initialises all services (DB, auth, AI models, sync)
    3.  Renders the login wall if not authenticated
    4.  Routes authenticated users to the correct page
    5.  Manages Streamlit session state across page navigations

Page routing:
    🏠 Home / Dashboard     → ui/dashboard.py
    📋 Patient Entry        → ui/data_entry_form.py
    🔍 Diagnosis View       → ui/diagnosis_view.py
    📊 Analytics            → ui/analytics_dashboard.py
    ⚙️  System Status        → ui/system_status.py
    👥 Role Dashboard       → ui/role_dashboard.py

Session state keys used:
    st.session_state.authenticated   : bool
    st.session_state.session_token   : str
    st.session_state.current_user    : UserRecord
    st.session_state.current_patient : dict
    st.session_state.risk_result     : dict
    st.session_state.ai_response     : dict
    st.session_state.similar_cases   : list
    st.session_state.current_page    : str
    st.session_state.app_initialized : bool
"""

import os
import sys
import time
from pathlib import Path

# ── Load .env before any other project imports ──────────────────────
from dotenv import load_dotenv
load_dotenv()

import streamlit as st

# ── Project imports ─────────────────────────────────────────────────
from utils.constants import (
    APP_NAME,
    APP_VERSION,
    APP_TAGLINE,
    APP_ICON,
    PAGES,
    ROLE_DOCTOR,
    ROLE_NURSE,
    ROLE_PERMISSIONS,
)
from utils.helpers import get_logger

# Services
from services.auth_service import (
    login,
    logout,
    get_current_user,
    is_authenticated,
    get_session,
    get_user_permissions,
    init_auth_db,
)
from services.sync_service import (
    schedule_auto_sync,
    stop_auto_sync,
    get_sync_status,
)

# Core
from core.mode_detector import is_online, get_connection_info
from core.risk_model import load_model
from core.similarity import preload_reference_data

# Database
from database.local_db import init_db

# UI modules
from ui.sidebar import render_sidebar
from ui.results_panel import render_results_panel
from ui.dashboard import render_dashboard

# Extended UI modules (with graceful fallback if not yet created)
def _safe_import_ui():
    """Safely import optional UI modules — returns dict of available renderers."""
    renderers = {}

    try:
        from ui.data_entry_form import render_data_entry_form
        renderers["data_entry"] = render_data_entry_form
    except ImportError:
        renderers["data_entry"] = _placeholder_page("📋 Data Entry Form")

    try:
        from ui.diagnosis_view import render_diagnosis_view
        renderers["diagnosis"] = render_diagnosis_view
    except ImportError:
        renderers["diagnosis"] = _placeholder_page("🔍 Diagnosis View")

    try:
        from ui.analytics_dashboard import render_analytics_dashboard
        renderers["analytics"] = render_analytics_dashboard
    except ImportError:
        renderers["analytics"] = _placeholder_page("📊 Analytics Dashboard")

    try:
        from ui.system_status import render_system_status
        renderers["system_status"] = render_system_status
    except ImportError:
        renderers["system_status"] = _placeholder_page("⚙️ System Status")

    try:
        from ui.role_dashboard import render_role_dashboard
        renderers["role_dashboard"] = render_role_dashboard
    except ImportError:
        renderers["role_dashboard"] = _placeholder_page("👥 Role Dashboard")

    return renderers


logger = get_logger(__name__)


# ─────────────────────────────────────────────
# Streamlit page configuration
# Must be the FIRST Streamlit call in the script
# ─────────────────────────────────────────────

st.set_page_config(
    page_title=f"{APP_NAME} — Cardiac AI",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help":     "https://github.com/yourusername/auraecho-plus",
        "Report a bug": "https://github.com/yourusername/auraecho-plus/issues",
        "About": (
            f"**{APP_NAME} v{APP_VERSION}**\n\n"
            f"{APP_TAGLINE}\n\n"
            "Built for clinical decision support. "
            "Not a substitute for professional medical judgment."
        ),
    },
)


# ─────────────────────────────────────────────
# Custom CSS loader
# ─────────────────────────────────────────────

def _load_css() -> None:
    """Inject custom CSS from assets/styles.css if it exists."""
    css_path = Path("assets/styles.css")
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as fh:
            css = fh.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    else:
        # Inline fallback CSS — blue/white theme, card shadows
        st.markdown("""
        <style>
        /* ── AuraEcho+ Base Theme ── */
        :root {
            --primary:    #1a56db;
            --secondary:  #e8f0fe;
            --danger:     #dc2626;
            --warning:    #d97706;
            --success:    #16a34a;
            --text-dark:  #1e293b;
            --text-muted: #64748b;
            --card-bg:    #ffffff;
            --border:     #e2e8f0;
        }

        /* Main content area */
        .main .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1400px;
        }

        /* Metric cards */
        div[data-testid="metric-container"] {
            background-color: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1rem 1.25rem;
            box-shadow: 0 1px 4px rgba(0,0,0,0.07);
        }

        /* Risk badge styling */
        .risk-high   { color: var(--danger);  font-weight: 700; font-size: 1.1rem; }
        .risk-medium { color: var(--warning); font-weight: 700; font-size: 1.1rem; }
        .risk-low    { color: var(--success); font-weight: 700; font-size: 1.1rem; }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #1e3a5f 100%);
        }
        section[data-testid="stSidebar"] * {
            color: #e2e8f0 !important;
        }

        /* Header banner */
        .app-header {
            background: linear-gradient(135deg, #1a56db, #0e3a8c);
            color: white;
            padding: 1rem 1.5rem;
            border-radius: 12px;
            margin-bottom: 1.5rem;
        }

        /* Cards */
        .aura-card {
            background: white;
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.25rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            margin-bottom: 1rem;
        }

        /* Hide Streamlit branding */
        #MainMenu { visibility: hidden; }
        footer    { visibility: hidden; }
        header    { visibility: hidden; }
        </style>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Placeholder page factory
# ─────────────────────────────────────────────

def _placeholder_page(title: str):
    """Return a no-op page renderer for modules not yet implemented."""
    def _render():
        st.title(title)
        st.info(
            f"🚧 **{title}** is under construction.\n\n"
            "This module will be available in the next build."
        )
    return _render


# ─────────────────────────────────────────────
# Application initialisation (runs once per session)
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner="⚙️ Initialising AuraEcho+ systems...")
def _initialize_app() -> dict:
    """
    One-time application startup.

    Uses st.cache_resource so this runs ONCE per Streamlit server
    process (not on every page rerender).

    Returns
    -------
    dict with initialisation status of each subsystem.
    """
    status = {}
    t0 = time.monotonic()
    logger.info("=" * 60)
    logger.info("AuraEcho+ v%s — Starting up", APP_VERSION)
    logger.info("=" * 60)

    # 1. Local database
    try:
        init_db()
        status["local_db"] = "✅ Ready"
        logger.info("[1/5] Local database: OK")
    except Exception as exc:
        status["local_db"] = f"❌ Failed: {exc}"
        logger.error("[1/5] Local database failed: %s", exc)

    # 2. Auth database
    try:
        init_auth_db()
        status["auth_db"] = "✅ Ready"
        logger.info("[2/5] Auth database: OK")
    except Exception as exc:
        status["auth_db"] = f"❌ Failed: {exc}"
        logger.error("[2/5] Auth database failed: %s", exc)

    # 3. ML Risk model (load or train)
    try:
        load_model()
        status["risk_model"] = "✅ Loaded"
        logger.info("[3/5] Risk model: OK")
    except Exception as exc:
        status["risk_model"] = f"⚠️ Warning: {exc}"
        logger.warning("[3/5] Risk model: %s", exc)

    # 4. KNN Similarity engine
    try:
        preload_reference_data()
        status["similarity"] = "✅ Loaded"
        logger.info("[4/5] Similarity engine: OK")
    except Exception as exc:
        status["similarity"] = f"⚠️ Warning: {exc}"
        logger.warning("[4/5] Similarity engine: %s", exc)

    # 5. Auto-sync (background thread)
    try:
        schedule_auto_sync()
        status["auto_sync"] = "✅ Running"
        logger.info("[5/5] Auto-sync thread: OK")
    except Exception as exc:
        status["auto_sync"] = f"⚠️ Warning: {exc}"
        logger.warning("[5/5] Auto-sync thread: %s", exc)

    elapsed = (time.monotonic() - t0) * 1000
    status["startup_ms"] = round(elapsed, 1)
    logger.info("Startup complete in %.0f ms", elapsed)

    return status


# ─────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────

def _init_session_state() -> None:
    """
    Set default values for all session state keys.
    Called on every page load — only sets values that don't exist yet.
    """
    defaults = {
        # Auth
        "authenticated":    False,
        "session_token":    "",
        "current_user":     None,

        # Patient data pipeline
        "current_patient":  None,    # raw patient dict from form
        "risk_result":      None,    # RiskResult.to_dict()
        "ai_response":      None,    # AIResponse.to_dict()
        "similar_cases":    None,    # List[SimilarCase.to_dict()]
        "clinical_brief":   None,    # ClinicalBrief.to_dict()

        # Navigation
        "current_page":     PAGES[0],  # default to first page

        # UI state
        "diagnosis_running":    False,
        "show_raw_data":        False,
        "selected_patient_id":  None,
        "conversation_history": [],

        # Notifications
        "notifications":    [],

        # App init flag
        "app_initialized":  False,
    }

    for key, default_val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_val


# ─────────────────────────────────────────────
# Login page
# ─────────────────────────────────────────────

def _render_login_page() -> None:
    """
    Render the login wall.
    Sets st.session_state.authenticated = True on success.
    """
    # Center the login form
    col_l, col_m, col_r = st.columns([1, 1.2, 1])

    with col_m:
        # Logo / header
        st.markdown(f"""
        <div style="text-align:center; padding: 2rem 0 1rem 0;">
            <div style="font-size: 3.5rem;">{APP_ICON}</div>
            <h1 style="color: #1a56db; margin: 0.25rem 0;">
                {APP_NAME}
            </h1>
            <p style="color: #64748b; font-size: 0.95rem; margin-bottom: 1.5rem;">
                {APP_TAGLINE}
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Connection status badge
        online = is_online()
        badge_color = "#16a34a" if online else "#dc2626"
        badge_text  = "🟢 Online Mode" if online else "🔴 Offline Mode"
        st.markdown(
            f'<div style="text-align:center; margin-bottom:1rem;">'
            f'<span style="background:{badge_color}22; color:{badge_color}; '
            f'padding:0.3rem 1rem; border-radius:20px; font-weight:600; '
            f'font-size:0.85rem;">{badge_text}</span></div>',
            unsafe_allow_html=True,
        )

        # Login form card
        with st.container(border=True):
            st.subheader("🔐 Clinical Staff Login", divider="blue")

            username = st.text_input(
                "Username",
                placeholder="Enter your username",
                key="login_username",
            )
            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter your password",
                key="login_password",
            )

            st.caption(
                "🔒 Demo credentials — Doctor: `admin_doctor` / `Doctor@123` "
                "| Nurse: `nurse_demo` / `Nurse@123`"
            )

            col_btn, col_info = st.columns([1, 1])
            with col_btn:
                login_clicked = st.button(
                    "Login →",
                    type="primary",
                    use_container_width=True,
                    key="login_btn",
                )

            if login_clicked:
                if not username or not password:
                    st.error("Please enter both username and password.")
                else:
                    with st.spinner("Authenticating..."):
                        session = login(username.strip(), password)

                    if session:
                        user = get_current_user(session.token)
                        st.session_state.authenticated   = True
                        st.session_state.session_token   = session.token
                        st.session_state.current_user    = user
                        st.session_state.current_page    = PAGES[0]

                        logger.info(
                            "User logged in: %s (%s)",
                            user.username if user else "?",
                            user.role if user else "?",
                        )
                        st.success(f"✅ Welcome, {user.display_name if user else username}!")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error(
                            "❌ Invalid username or password. "
                            "Please check your credentials and try again."
                        )

        # Footer
        st.markdown("""
        <div style="text-align:center; margin-top:1.5rem; color:#94a3b8; font-size:0.8rem;">
            AuraEcho+ is a clinical decision support tool.<br>
            Not a substitute for professional medical judgment.<br>
            © 2024 AuraEcho+ Team
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Navigation header
# ─────────────────────────────────────────────

def _render_top_header() -> None:
    """
    Render the top application header bar with:
    - App name + version
    - Connection status
    - Current user + role badge
    - Logout button
    """
    user  = st.session_state.get("current_user")
    online = is_online()

    col_logo, col_status, col_user, col_logout = st.columns([3, 2, 2, 1])

    with col_logo:
        st.markdown(
            f"<h2 style='margin:0; color:#1a56db;'>{APP_ICON} {APP_NAME}</h2>"
            f"<p style='margin:0; color:#64748b; font-size:0.8rem;'>v{APP_VERSION} — {APP_TAGLINE}</p>",
            unsafe_allow_html=True,
        )

    with col_status:
        if online:
            st.success("🟢 Online", icon=None)
        else:
            st.warning("🔴 Offline Mode", icon=None)

        # Pending sync count
        try:
            sync_st = get_sync_status()
            pending = sync_st.get("total_pending", 0)
            if pending > 0:
                st.caption(f"⏳ {pending} records pending sync")
        except Exception:
            pass

    with col_user:
        if user:
            role_color = "#1a56db" if user.role == ROLE_DOCTOR else "#7c3aed"
            role_icon  = "👨‍⚕️" if user.role == ROLE_DOCTOR else "🩺"
            st.markdown(
                f"<div style='line-height:1.3;'>"
                f"<strong>{role_icon} {user.display_name}</strong><br>"
                f"<span style='color:{role_color}; font-size:0.8rem; font-weight:600;'>"
                f"{user.role.upper()}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    with col_logout:
        if st.button("🚪 Logout", key="logout_btn", use_container_width=True):
            token = st.session_state.get("session_token", "")
            if token:
                logout(token)
            # Clear ALL session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    st.divider()


# ─────────────────────────────────────────────
# Page router
# ─────────────────────────────────────────────

def _get_page_nav_items(user) -> list:
    """
    Build navigation items based on user role.
    Doctors see all pages; nurses see restricted set.
    """
    all_pages = [
        {"key": "dashboard",     "label": "🏠 Dashboard",        "nurse_allowed": True},
        {"key": "patient_entry", "label": "📋 Patient Entry",     "nurse_allowed": True},
        {"key": "diagnosis",     "label": "🔍 AI Diagnosis",      "nurse_allowed": False},
        {"key": "analytics",     "label": "📊 Analytics",         "nurse_allowed": True},
        {"key": "system_status", "label": "⚙️  System Status",    "nurse_allowed": False},
        {"key": "role_dashboard","label": "👥 Role Dashboard",    "nurse_allowed": True},
    ]

    if user and user.role == ROLE_NURSE:
        return [p for p in all_pages if p["nurse_allowed"]]
    return all_pages


def _render_page_navigation(user) -> str:
    """
    Render the horizontal page navigation tabs.
    Returns the key of the selected page.
    """
    nav_items = _get_page_nav_items(user)
    labels    = [item["label"] for item in nav_items]
    keys      = [item["key"]   for item in nav_items]

    # Find current page index
    current_key = st.session_state.get("current_page", keys[0])
    if current_key not in keys:
        current_key = keys[0]
    current_idx = keys.index(current_key)

    selected_label = st.radio(
        "Navigation",
        options=labels,
        index=current_idx,
        horizontal=True,
        key="page_nav_radio",
        label_visibility="collapsed",
    )

    # Map label back to key
    selected_key = keys[labels.index(selected_label)]
    return selected_key


def _route_to_page(page_key: str, ui_renderers: dict) -> None:
    """
    Call the correct UI renderer for the selected page.

    Parameters
    ----------
    page_key     : str — the navigation key
    ui_renderers : dict — map of key → render function
    """
    route_map = {
        "dashboard":      _render_main_dashboard,
        "patient_entry":  lambda: _render_patient_entry(ui_renderers),
        "diagnosis":      lambda: _render_diagnosis(ui_renderers),
        "analytics":      ui_renderers.get("analytics", _placeholder_page("📊 Analytics")),
        "system_status":  ui_renderers.get("system_status", _placeholder_page("⚙️ System Status")),
        "role_dashboard": ui_renderers.get("role_dashboard", _placeholder_page("👥 Role Dashboard")),
    }

    renderer = route_map.get(page_key, _placeholder_page(f"❓ Page: {page_key}"))
    renderer()


# ─────────────────────────────────────────────
# Page renderers — compose the UI modules
# ─────────────────────────────────────────────

def _render_main_dashboard() -> None:
    """
    Home dashboard page.
    Shows: summary stats, recent patients, connectivity, quick-start.
    """
    st.title("🏠 Clinical Dashboard")
    st.caption("Welcome to AuraEcho+ — your cardiac decision support system.")

    # Connection info banner
    conn_info = get_connection_info()
    if conn_info.get("online"):
        st.success(
            f"🟢 **Online Mode** — Cloud AI and Firebase sync active "
            f"(latency: {conn_info.get('latency_ms', 0):.0f} ms)",
            icon=None,
        )
    else:
        st.warning(
            "🔴 **Offline Mode** — Using local Ollama AI and SQLite storage. "
            "Records will sync when internet is restored.",
            icon=None,
        )

    # Delegate to the dashboard UI module
    render_dashboard()


def _render_patient_entry(ui_renderers: dict) -> None:
    """
    Patient data entry page.
    Composes: sidebar (patient form) + results panel.
    """
    col_form, col_results = st.columns([1, 1.8])

    with col_form:
        # Sidebar form renders the patient input form
        patient_data, submitted = render_sidebar()

        if submitted and patient_data:
            st.session_state.current_patient = patient_data
            # Trigger analysis pipeline
            _run_analysis_pipeline(patient_data)

    with col_results:
        if st.session_state.get("risk_result"):
            render_results_panel(
                patient      = st.session_state.current_patient,
                risk_result  = st.session_state.risk_result,
                ai_response  = st.session_state.ai_response,
                similar_cases= st.session_state.similar_cases,
            )
        else:
            st.info(
                "👈 **Enter patient data** in the form on the left and click "
                "'Run Analysis' to see the AI cardiac assessment.",
                icon="ℹ️",
            )
            _render_quick_start_guide()


def _render_diagnosis(ui_renderers: dict) -> None:
    """
    Full AI Diagnosis view page.
    Requires a current patient to be loaded.
    """
    render_fn = ui_renderers.get("diagnosis")
    if render_fn is None:
        st.error("Diagnosis view module not available.")
        return

    if st.session_state.get("current_patient") is None:
        st.warning(
            "⚠️ No patient loaded. Go to **📋 Patient Entry** first "
            "and run an analysis.",
            icon="⚠️",
        )
        if st.button("← Go to Patient Entry"):
            st.session_state.current_page = "patient_entry"
            st.rerun()
        return

    render_fn()


# ─────────────────────────────────────────────
# Analysis pipeline
# ─────────────────────────────────────────────

def _run_analysis_pipeline(patient: dict) -> None:
    """
    Orchestrate the full analysis for a patient.

    Pipeline:
    1. Risk scoring (Random Forest)
    2. Similar case finding (KNN)
    3. AI diagnosis (online or offline LLM)
    4. Clinical brief (guidelines + drug checks)
    5. Save to local DB
    6. Trigger sync if online

    All results are stored in st.session_state so any UI module
    can access them without re-running the pipeline.
    """
    from core.risk_model import predict_risk
    from core.similarity import find_similar_cases
    from services.api_service import get_full_clinical_brief
    from database.local_db import save_patient, save_assessment

    progress_bar = st.progress(0, text="Starting analysis...")

    try:
        # Step 1: Risk model
        progress_bar.progress(15, text="🧠 Scoring cardiac risk...")
        risk_result = predict_risk(patient)
        st.session_state.risk_result = risk_result.to_dict()
        logger.info("Risk scored: %s (%.1f%%)", risk_result.risk_level, risk_result.confidence_pct)

        # Step 2: Similar cases
        progress_bar.progress(35, text="🔍 Finding similar cases...")
        similar = find_similar_cases(patient, k=3)
        st.session_state.similar_cases = [c.to_dict() for c in similar]

        # Step 3: AI diagnosis
        progress_bar.progress(55, text="🤖 Running AI diagnosis...")
        online = is_online()

        if online:
            from ai.online_ai import analyze_patient as online_analyze
            ai_resp = online_analyze(
                patient       = patient,
                risk_result   = st.session_state.risk_result,
                similar_cases = st.session_state.similar_cases,
            )
            logger.info("Online AI used: %s", ai_resp.source)
        else:
            from ai.offline_ai import analyze_patient as offline_analyze
            ai_resp = offline_analyze(
                patient       = patient,
                risk_result   = st.session_state.risk_result,
                similar_cases = st.session_state.similar_cases,
            )
            logger.info("Offline AI used: %s", ai_resp.source)

        st.session_state.ai_response = ai_resp.to_dict()

        # Step 4: Clinical brief (guidelines + drugs)
        progress_bar.progress(75, text="📋 Fetching clinical guidelines...")
        brief = get_full_clinical_brief(
            patient    = patient,
            risk_level = risk_result.risk_level,
        )
        st.session_state.clinical_brief = brief.to_dict()

        # Step 5: Save to local DB
        progress_bar.progress(90, text="💾 Saving record...")
        patient_id    = save_patient(patient)
        assessment_id = save_assessment(
            patient_id    = patient_id,
            risk_result   = st.session_state.risk_result,
            ai_response   = st.session_state.ai_response,
            similar_cases = st.session_state.similar_cases,
        )
        st.session_state.selected_patient_id = patient_id
        logger.info("Saved: patient=%s assessment=%s", patient_id, assessment_id)

        progress_bar.progress(100, text="✅ Analysis complete!")
        time.sleep(0.3)
        progress_bar.empty()

        st.success(
            f"✅ Analysis complete — Risk Level: "
            f"**{risk_result.risk_level}** "
            f"({risk_result.confidence_pct:.1f}% confidence)"
        )

    except Exception as exc:
        progress_bar.empty()
        logger.error("Analysis pipeline error: %s", exc)
        st.error(
            f"⚠️ Analysis encountered an error: `{exc}`\n\n"
            "Risk scoring and similar cases may still be available. "
            "Check system logs for details."
        )


# ─────────────────────────────────────────────
# Quick start guide (shown when no patient loaded)
# ─────────────────────────────────────────────

def _render_quick_start_guide() -> None:
    """Render a helpful guide for first-time users."""
    with st.expander("🚀 Quick Start Guide", expanded=True):
        st.markdown("""
        ### How to use AuraEcho+

        **Step 1 — Enter Patient Data** 👈
        Fill in the patient demographics and clinical vitals in the
        form on the left. All fields are explained with normal ranges.

        **Step 2 — Run Analysis** 🔬
        Click the **"Run Analysis"** button. The system will:
        - Score cardiac risk (Low / Medium / High)
        - Find 3 similar historical cases
        - Generate an AI clinical assessment
        - Retrieve ACC/AHA treatment guidelines

        **Step 3 — Review Results** 📋
        Review the diagnosis card, risk indicators, and recommendations.
        Use the **📊 Analytics** tab for historical trends.

        **Step 4 — Save & Sync** ☁️
        Records are auto-saved locally. If online, they sync to the
        cloud automatically for access from any device.

        ---
        💡 **Tip**: Switch to **Offline Mode** (no internet needed) using
        local Llama3 AI — all features still work!
        """)


# ─────────────────────────────────────────────
# Startup status display
# ─────────────────────────────────────────────

def _show_startup_status(init_status: dict) -> None:
    """
    Show a collapsible startup status box (useful for debugging).
    Only shown once per session.
    """
    if st.session_state.get("startup_shown"):
        return

    has_errors = any("❌" in str(v) for v in init_status.values())

    if has_errors:
        with st.expander("⚠️ System Startup Issues — Click to expand", expanded=True):
            for component, status in init_status.items():
                if component == "startup_ms":
                    continue
                icon = "✅" if "✅" in str(status) else "⚠️" if "⚠️" in str(status) else "❌"
                st.markdown(f"{icon} **{component}**: {status}")
            st.caption(
                f"Startup completed in {init_status.get('startup_ms', 0):.0f} ms"
            )

    st.session_state.startup_shown = True


# ─────────────────────────────────────────────
# Main application entry point
# ─────────────────────────────────────────────

def main() -> None:
    """
    Main application function.

    Called on every Streamlit script re-run (every user interaction).
    Uses session state to maintain context between reruns.
    """
    # 1. Load custom CSS
    _load_css()

    # 2. Initialise session state defaults
    _init_session_state()

    # 3. One-time app initialisation (cached across reruns)
    init_status = _initialize_app()

    # 4. Import UI renderers
    ui_renderers = _safe_import_ui()

    # ── AUTH GATE ───────────────────────────────────────────────────
    if not st.session_state.authenticated:
        _render_login_page()
        return

    # ── Validate session is still alive ─────────────────────────────
    token = st.session_state.get("session_token", "")
    if not is_authenticated(token):
        st.warning("⏱️ Your session has expired. Please log in again.")
        st.session_state.authenticated = False
        st.session_state.session_token = ""
        time.sleep(1)
        st.rerun()
        return

    # ── AUTHENTICATED LAYOUT ─────────────────────────────────────────

    # Top header (app name, user info, logout)
    _render_top_header()

    # Show startup issues if any
    _show_startup_status(init_status)

    # Page navigation tabs
    user         = st.session_state.get("current_user")
    selected_page = _render_page_navigation(user)

    # Update current page in session state
    if selected_page != st.session_state.current_page:
        st.session_state.current_page = selected_page

    # Divider between nav and content
    st.markdown("<br>", unsafe_allow_html=True)

    # Route to the selected page
    _route_to_page(selected_page, ui_renderers)


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    main()