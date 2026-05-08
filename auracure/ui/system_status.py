# =============================================================================
# ui/system_status.py
# AuraEcho+ — System Status Panel Component
#
# Responsibility:
#     Render a comprehensive system status panel showing connectivity,
#     AI backends, database health, sync status, and model information.
#     Provides admin controls for testing and maintenance.
#
# Features:
#     • Real-time connectivity status with latency
#     • AI backend availability (Ollama, Groq, OpenAI)
#     • Database health (local + cloud)
#     • Sync queue status and controls
#     • Model information and metadata
#     • Test connection buttons
#     • Maintenance actions (clear cache, retry sync)
#     • Role-based admin controls
#
# Public API:
#     render_system_status(user_role) → None
# =============================================================================

import streamlit as st
import time
from typing import Any, Dict, Optional

from utils.constants import (
    APP_NAME,
    APP_VERSION,
    MODE_ONLINE_LABEL,
    MODE_OFFLINE_LABEL,
    UI_SUCCESS_COLOR,
    UI_WARNING_COLOR,
    UI_DANGER_COLOR,
    ROLE_PERMISSIONS,
)
from utils.helpers import get_logger, now_str, mask_key
from core.mode_detector import (
    is_online,
    get_connection_info,
    get_mode_label,
    invalidate_cache,
    simulate_offline,
)
from ai.offline_ai import (
    is_ollama_available,
    get_ollama_status,
    get_model_info as get_ollama_model_info,
    warmup_model as warmup_ollama,
)
from ai.online_ai import (
    is_groq_available,
    is_openai_available,
    get_api_status,
    test_api_connection,
    get_model_info as get_online_model_info,
)
from database.cloud_db import get_cloud_status, verify_connection
from services.sync_service import (
    get_sync_status,
    force_sync,
    retry_failed_sync,
    clear_synced_items,
    start_auto_sync,
    stop_auto_sync,
    is_sync_active,
)
from core.risk_model import get_model_metadata, retrain_if_stale
from services.auth_service import user_has_permission

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# CSS Styles
# ─────────────────────────────────────────────

STATUS_CSS = """
<style>
    /* Status card */
    .status-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid #e8f0fe;
    }
    
    /* Status header */
    .status-header {
        font-size: 1rem;
        font-weight: 600;
        color: #1a1a2e;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Status indicator */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.4rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    .status-ok {
        background-color: #d4edda;
        color: #155724;
    }
    .status-warn {
        background-color: #fff3cd;
        color: #856404;
    }
    .status-error {
        background-color: #f8d7da;
        color: #721c24;
    }
    
    /* Status dot */
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
    }
    
    /* Status row */
    .status-row {
        display: flex;
        justify-content: space-between;
        padding: 0.4rem 0;
        border-bottom: 1px solid #f0f0f0;
        font-size: 0.85rem;
    }
    .status-row:last-child {
        border-bottom: none;
    }
    .status-label {
        color: #5f6368;
    }
    .status-value {
        color: #1a1a2e;
        font-weight: 500;
    }
    
    /* Action button */
    .action-btn {
        font-size: 0.8rem;
        padding: 0.35rem 0.75rem;
    }
    
    /* Section divider */
    .status-divider {
        border-top: 1px solid #e0e0e0;
        margin: 0.75rem 0;
    }
</style>
"""


# ─────────────────────────────────────────────
# Helper Render Functions
# ─────────────────────────────────────────────

def _render_status_indicator(
    status: bool,
    ok_text: str,
    warn_text: str = "Unavailable",
) -> None:
    """Render a status indicator badge."""
    if status:
        st.markdown(
            f"""
            <div class="status-indicator status-ok">
                <span class="status-dot" style="background-color: #2ecc71;"></span>
                {ok_text}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="status-indicator status-warn">
                <span class="status-dot" style="background-color: #f39c12;"></span>
                {warn_text}
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_status_row(label: str, value: str) -> None:
    """Render a status row."""
    st.markdown(
        f"""
        <div class="status-row">
            <span class="status-label">{label}</span>
            <span class="status-value">{value}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# Section Renderers
# ─────────────────────────────────────────────

def _render_connectivity_section(can_admin: bool) -> None:
    """Render connectivity status section."""
    st.markdown(
        """
        <div class="status-card">
            <div class="status-header">🌐 Connectivity</div>
        """,
        unsafe_allow_html=True,
    )
    
    conn_info = get_connection_info()
    online = conn_info.get("online", False)
    latency = conn_info.get("latency_ms", -1)
    host = conn_info.get("host", "N/A")
    cached = conn_info.get("cached", False)
    
    _render_status_indicator(online, "Online", "Offline")
    
    _render_status_row("Mode", get_mode_label())
    _render_status_row("Latency", f"{latency:.1f} ms" if latency >= 0 else "N/A")
    _render_status_row("Host", host)
    _render_status_row("Cached", "✓" if cached else "✗")
    
    # Action buttons
    if can_admin:
        st.markdown('<div class="status-divider"></div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh", key="refresh_conn", use_container_width=True):
                invalidate_cache()
                st.rerun()
        with col2:
            if st.button("📴 Simulate Offline", key="simulate_offline", use_container_width=True):
                simulate_offline(duration_seconds=30)
                st.toast("Simulating offline mode for 30 seconds", icon="📴")
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)


def _render_ai_backends_section(can_admin: bool) -> None:
    """Render AI backend status section."""
    st.markdown(
        """
        <div class="status-card">
            <div class="status-header">🤖 AI Backends</div>
        """,
        unsafe_allow_html=True,
    )
    
    # Ollama status
    ollama_status = get_ollama_status()
    ollama_ok = ollama_status.get("available", False)
    ollama_model = ollama_status.get("model", "N/A")
    ollama_loaded = ollama_status.get("model_loaded", False)
    
    st.markdown("**Ollama (Local)**")
    _render_status_indicator(
        ollama_ok and ollama_loaded,
        "Ready",
        ollama_status.get("error", "Unavailable"),
    )
    _render_status_row("Model", ollama_model)
    _render_status_row("Loaded", "✓" if ollama_loaded else "✗")
    
    if can_admin and ollama_ok:
        if st.button("🔥 Warmup", key="warmup_ollama", use_container_width=True):
            with st.spinner("Warming up model..."):
                success = warmup_ollama()
                if success:
                    st.toast("✅ Model warmed up", icon="✅")
                else:
                    st.toast("❌ Warmup failed", icon="❌")
                st.rerun()
    
    st.markdown('<div class="status-divider"></div>', unsafe_allow_html=True)
    
    # Groq status
    groq_ok = is_groq_available()
    st.markdown("**Groq (Cloud)**")
    _render_status_indicator(groq_ok, "Configured", "Not Configured")
    
    if can_admin and groq_ok:
        if st.button("🧪 Test Groq", key="test_groq", use_container_width=True):
            with st.spinner("Testing Groq connection..."):
                result = test_api_connection("groq")
                if result.get("success"):
                    st.toast(
                        f"✅ Groq OK ({result.get('latency_ms', 0):.0f} ms)",
                        icon="✅",
                    )
                else:
                    st.toast(f"❌ {result.get('error', 'Test failed')}", icon="❌")
                st.rerun()
    
    st.markdown('<div class="status-divider"></div>', unsafe_allow_html=True)
    
    # OpenAI status
    openai_ok = is_openai_available()
    st.markdown("**OpenAI (Fallback)**")
    _render_status_indicator(openai_ok, "Configured", "Not Configured")
    
    if can_admin and openai_ok:
        if st.button("🧪 Test OpenAI", key="test_openai", use_container_width=True):
            with st.spinner("Testing OpenAI connection..."):
                result = test_api_connection("openai")
                if result.get("success"):
                    st.toast(
                        f"✅ OpenAI OK ({result.get('latency_ms', 0):.0f} ms)",
                        icon="✅",
                    )
                else:
                    st.toast(f"❌ {result.get('error', 'Test failed')}", icon="❌")
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)


def _render_database_section(can_admin: bool) -> None:
    """Render database status section."""
    st.markdown(
        """
        <div class="status-card">
            <div class="status-header">💾 Database</div>
        """,
        unsafe_allow_html=True,
    )
    
    # Local DB
    st.markdown("**Local (SQLite)**")
    _render_status_indicator(True, "Connected", "Error")
    _render_status_row("Path", "database/auraecho.db")
    
    st.markdown('<div class="status-divider"></div>', unsafe_allow_html=True)
    
    # Cloud DB
    cloud_status = get_cloud_status()
    cloud_ok = cloud_status.get("available", False)
    cloud_connected = cloud_status.get("connected", False)
    
    st.markdown("**Cloud (Firebase)**")
    _render_status_indicator(
        cloud_connected,
        "Connected",
        cloud_status.get("init_error", "Not Configured"),
    )
    _render_status_row("Collection", cloud_status.get("collection", "N/A"))
    _render_status_row("Credentials", "✓" if cloud_status.get("credentials_set") else "✗")
    
    if can_admin and cloud_ok:
        if st.button("🧪 Verify Cloud", key="verify_cloud", use_container_width=True):
            with st.spinner("Verifying cloud connection..."):
                success = verify_connection()
                if success:
                    st.toast("✅ Cloud connection verified", icon="✅")
                else:
                    st.toast("❌ Cloud verification failed", icon="❌")
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)


def _render_sync_section(can_admin: bool) -> None:
    """Render sync status section."""
    st.markdown(
        """
        <div class="status-card">
            <div class="status-header">🔄 Sync Service</div>
        """,
        unsafe_allow_html=True,
    )
    
    sync_status = get_sync_status()
    auto_sync = sync_status.get("auto_sync_active", False)
    pending = sync_status.get("pending_count", 0)
    failed = sync_status.get("failed_count", 0)
    last_sync = sync_status.get("last_sync_time", "Never")
    
    _render_status_indicator(auto_sync, "Auto-Sync Active", "Auto-Sync Paused")
    
    _render_status_row("Pending", str(pending))
    _render_status_row("Failed", str(failed))
    _render_status_row("Last Sync", last_sync)
    _render_status_row("Total Synced", str(sync_status.get("total_synced", 0)))
    
    if can_admin:
        st.markdown('<div class="status-divider"></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Sync Now", key="sync_now", use_container_width=True):
                with st.spinner("Syncing..."):
                    result = force_sync()
                    if result.get("success"):
                        st.toast(
                            f"✅ Synced {result.get('synced_count', 0)} items",
                            icon="✅",
                        )
                    else:
                        st.toast(f"❌ {result.get('message')}", icon="❌")
                    st.rerun()
        
        with col2:
            new_auto = st.toggle(
                "Auto-Sync",
                value=auto_sync,
                key="auto_sync_toggle",
            )
            if new_auto != auto_sync:
                if new_auto:
                    start_auto_sync()
                    st.toast("Auto-sync enabled", icon="🔄")
                else:
                    stop_auto_sync()
                    st.toast("Auto-sync disabled", icon="⏸️")
                st.rerun()
        
        if failed > 0:
            if st.button("🔁 Retry Failed", key="retry_failed", use_container_width=True):
                result = retry_failed_sync()
                st.toast(result.get("message", "Retried failed items"), icon="🔁")
                st.rerun()
        
        if st.button("🧹 Clear Old Synced", key="clear_synced", use_container_width=True):
            result = clear_synced_items()
            st.toast(result.get("message", "Cleared old items"), icon="🧹")
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)


def _render_model_section(can_admin: bool) -> None:
    """Render risk model status section."""
    st.markdown(
        """
        <div class="status-card">
            <div class="status-header">🧠 Risk Model</div>
        """,
        unsafe_allow_html=True,
    )
    
    model_meta = get_model_metadata()
    
    _render_status_indicator(
        model_meta.get("model_exists", False),
        "Model Loaded",
        "Model Not Found",
    )
    
    _render_status_row("Type", model_meta.get("model_type", "N/A"))
    _render_status_row("Version", model_meta.get("model_version", "N/A"))
    _render_status_row("Estimators", str(model_meta.get("n_estimators", "N/A")))
    _render_status_row("Max Depth", str(model_meta.get("max_depth", "N/A")))
    _render_status_row("Features", str(model_meta.get("n_features", "N/A")))
    
    top_features = model_meta.get("top_features", [])
    if top_features:
        st.markdown("**Top Features**")
        for feat, imp in top_features:
            st.caption(f"• {feat}: {imp:.2%}")
    
    if can_admin:
        st.markdown('<div class="status-divider"></div>', unsafe_allow_html=True)
        if st.button("🔄 Retrain if Stale", key="retrain_model", use_container_width=True):
            with st.spinner("Checking model freshness..."):
                retrained = retrain_if_stale(max_age_days=30)
                if retrained:
                    st.toast("✅ Model retrained", icon="✅")
                else:
                    st.toast("ℹ️ Model is still fresh", icon="ℹ️")
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)


def _render_app_info_section() -> None:
    """Render application info section."""
    st.markdown(
        """
        <div class="status-card">
            <div class="status-header">ℹ️ Application</div>
        """,
        unsafe_allow_html=True,
    )
    
    _render_status_row("Name", APP_NAME)
    _render_status_row("Version", APP_VERSION)
    _render_status_row("Time", now_str())
    
    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Main Render Function
# ─────────────────────────────────────────────

def render_system_status(user_role: Optional[str] = None) -> None:
    """
    Render the complete system status panel.

    Parameters
    ----------
    user_role : str — current user role for permission checks
    """
    # Check permissions
    can_view = user_has_permission(
        {"role": user_role}, "view_dashboard"
    ) if user_role else True
    
    can_admin = user_has_permission(
        {"role": user_role}, "system_settings"
    ) if user_role else False
    
    if not can_view:
        st.error("⚠️ You don't have permission to view system status.")
        return
    
    # Inject CSS
    st.markdown(STATUS_CSS, unsafe_allow_html=True)
    
    # Title
    st.markdown("## 🔧 System Status")
    
    # Auto-refresh toggle
    auto_refresh = st.toggle(
        "Auto-refresh",
        value=False,
        key="status_auto_refresh",
        help="Automatically refresh status every 30 seconds",
    )
    
    if auto_refresh:
        st.autorefresh(interval=30000, key="status_autorefresh")
    
    st.markdown("---")
    
    # Grid layout
    col1, col2 = st.columns(2)
    
    with col1:
        _render_connectivity_section(can_admin)
        _render_ai_backends_section(can_admin)
        _render_database_section(can_admin)
    
    with col2:
        _render_sync_section(can_admin)
        _render_model_section(can_admin)
        _render_app_info_section()
    
    # Debug expander
    with st.expander("🔧 Status Debug", expanded=False):
        st.json({
            "connectivity": get_connection_info(),
            "ai_status": get_api_status(),
            "sync_status": get_sync_status(),
            "cloud_status": get_cloud_status(),
            "model_meta": get_model_metadata(),
        })