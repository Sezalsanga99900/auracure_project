# =============================================================================
# ui/diagnosis_view.py
# AuraEcho+ — AI Diagnosis Card Component
#
# Responsibility:
#     Render the AI diagnosis card with structured sections, risk summary,
#     source attribution, and action buttons. Handles all states including
#     loading, error, success, and rule-based fallback.
#
# Features:
#     • Structured section display (Assessment, Indicators, Recommendations)
#     • Risk badge integration with color coding
#     • Source attribution (Local/Cloud AI model info)
#     • Latency and token usage display
#     • Fallback handling for offline/unavailable AI
#     • Action buttons (Regenerate, Copy, Save)
#     • Role-based permission checks
#     • Clinical disclaimer
#
# Public API:
#     render_diagnosis_view(risk_result, ai_response, patient_data) → None
# =============================================================================

import streamlit as st
import json
from typing import Any, Dict, List, Optional

from utils.constants import (
    APP_NAME,
    RISK_LABELS,
    RISK_COLORS,
    RISK_ICONS,
    UI_PRIMARY_COLOR,
    UI_SUCCESS_COLOR,
    UI_WARNING_COLOR,
    UI_DANGER_COLOR,
    UI_CARD_COLOR,
    UI_SHADOW,
    ROLE_PERMISSIONS,
)
from utils.helpers import (
    get_logger,
    format_score,
    now_str,
    truncate,
)
from services.auth_service import user_has_permission

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# CSS Styles
# ─────────────────────────────────────────────

DIAGNOSIS_CSS = """
<style>
    /* Diagnosis card */
    .diagnosis-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid #e8f0fe;
    }
    
    /* Header */
    .diagnosis-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 1rem;
        flex-wrap: wrap;
        gap: 0.5rem;
    }
    .diagnosis-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1a1a2e;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Risk badge */
    .risk-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        color: white;
    }
    
    /* Source badge */
    .source-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.6rem;
        border-radius: 10px;
        font-size: 0.75rem;
        font-weight: 500;
        background-color: #e8f0fe;
        color: #1a73e8;
    }
    .source-badge.local {
        background-color: #d4edda;
        color: #155724;
    }
    .source-badge.fallback {
        background-color: #fff3cd;
        color: #856404;
    }
    
    /* Section */
    .diagnosis-section {
        margin-bottom: 1rem;
    }
    .section-title {
        font-weight: 600;
        color: #1a73e8;
        margin-bottom: 0.5rem;
        font-size: 0.95rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .section-content {
        color: #333;
        line-height: 1.6;
        font-size: 0.95rem;
        background-color: #f8f9fa;
        padding: 0.75rem;
        border-radius: 6px;
        border-left: 3px solid #e0e0e0;
    }
    .section-content.warning {
        border-left-color: #ffc107;
        background-color: #fffdf5;
    }
    .section-content.critical {
        border-left-color: #dc3545;
        background-color: #fff5f5;
    }
    
    /* Metadata */
    .diagnosis-meta {
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
        margin-top: 1rem;
        padding-top: 1rem;
        border-top: 1px solid #e0e0e0;
        font-size: 0.8rem;
        color: #5f6368;
    }
    .meta-item {
        display: flex;
        align-items: center;
        gap: 0.25rem;
    }
    
    /* Disclaimer */
    .disclaimer {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 0.75rem;
        border-radius: 6px;
        font-size: 0.85rem;
        color: #856404;
        margin-top: 1rem;
    }
    
    /* Error state */
    .error-state {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        border-radius: 6px;
        color: #721c24;
    }
    
    /* Loading state */
    .loading-state {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 2rem;
        color: #5f6368;
    }
</style>
"""


# ─────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────

def _get_source_info(source: str, model: str) -> Dict[str, str]:
    """
    Get source display info based on source type.
    """
    source_map = {
        "offline_llama3": {
            "label": "🖥️ Local AI",
            "class": "local",
            "tooltip": "Running locally via Ollama — 100% private",
        },
        "online_groq": {
            "label": "☁️ Groq",
            "class": "",
            "tooltip": "Cloud inference via Groq API",
        },
        "online_openai": {
            "label": "☁️ OpenAI",
            "class": "",
            "tooltip": "Cloud inference via OpenAI API",
        },
        "offline_fallback": {
            "label": "📋 Rule-Based",
            "class": "fallback",
            "tooltip": "AI unavailable — using rule-based guidance",
        },
        "rule_based_fallback": {
            "label": "📋 Rule-Based",
            "class": "fallback",
            "tooltip": "AI unavailable — using rule-based guidance",
        },
    }
    
    info = source_map.get(source, {
        "label": f"🤖 {source}",
        "class": "",
        "tooltip": source,
    })
    info["model"] = model
    return info


def _get_section_icon(section_name: str) -> str:
    """Get icon for section name."""
    icons = {
        "Clinical Assessment": "🔍",
        "Key Risk Indicators": "⚠️",
        "Potential Future Symptoms": "🔮",
        "Treatment Recommendations": "💊",
        "Referral & Follow-up": "🏥",
        "Patient Education Points": "📋",
        "Immediate Actions": "🚨",
        "Emergency Referral": "🚑",
        "Critical Monitoring": "📊",
        "Red Flag Symptoms": "🚩",
    }
    return icons.get(section_name, "📝")


def _get_section_class(section_name: str, content: str) -> str:
    """Determine CSS class for section based on content."""
    content_lower = content.lower()
    if "urgent" in content_lower or "immediate" in content_lower or "emergency" in content_lower:
        return "critical"
    if "warning" in content_lower or "caution" in content_lower:
        return "warning"
    return ""


# ─────────────────────────────────────────────
# Render Functions
# ─────────────────────────────────────────────

def _render_risk_badge(risk_result: Dict[str, Any]) -> None:
    """Render risk level badge."""
    if not risk_result:
        return
    
    risk_level = risk_result.get("risk_level", "MEDIUM")
    risk_label = risk_result.get("risk_label", RISK_LABELS.get(risk_level, risk_level))
    risk_color = risk_result.get("badge_color", RISK_COLORS.get(risk_level, "#95a5a6"))
    risk_icon = risk_result.get("badge_icon", RISK_ICONS.get(risk_level, "❔"))
    
    st.markdown(
        f"""
        <div class="risk-badge" style="background-color: {risk_color};">
            {risk_icon} {risk_label}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_source_badge(source_info: Dict[str, str]) -> None:
    """Render source attribution badge."""
    css_class = source_info.get("class", "")
    label = source_info["label"]
    model = source_info.get("model", "")
    tooltip = source_info.get("tooltip", "")
    
    model_text = f" · {model}" if model else ""
    
    st.markdown(
        f"""
        <div class="source-badge {css_class}" title="{tooltip}">
            {label}{model_text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_section(name: str, content: str) -> None:
    """Render a single diagnosis section."""
    icon = _get_section_icon(name)
    css_class = _get_section_class(name, content)
    
    st.markdown(
        f"""
        <div class="diagnosis-section">
            <div class="section-title">{icon} {name}</div>
            <div class="section-content {css_class}">{content}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_metadata(ai_response: Dict[str, Any]) -> None:
    """Render metadata footer."""
    latency = ai_response.get("latency_ms", 0)
    prompt_tokens = ai_response.get("prompt_tokens", 0)
    output_tokens = ai_response.get("output_tokens", 0)
    
    st.markdown(
        f"""
        <div class="diagnosis-meta">
            <div class="meta-item">⏱️ {latency:.0f} ms</div>
            <div class="meta-item">📥 {prompt_tokens} tokens</div>
            <div class="meta-item">📤 {output_tokens} tokens</div>
            <div class="meta-item">🕐 {now_str()}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_action_buttons(
    risk_result: Dict[str, Any],
    ai_response: Dict[str, Any],
    patient_data: Dict[str, Any],
    user_role: Optional[str] = None,
) -> None:
    """Render action buttons with permission checks."""
    col1, col2, col3 = st.columns(3)
    
    # Regenerate button
    can_regenerate = user_has_permission(
        {"role": user_role}, "view_ai_insights"
    ) if user_role else True
    
    with col1:
        if st.button(
            "🔄 Regenerate",
            key="diagnosis_regenerate",
            use_container_width=True,
            disabled=not can_regenerate,
        ):
            if can_regenerate:
                st.session_state["regenerate_ai"] = True
                st.rerun()
            else:
                st.toast("⚠️ No permission to regenerate", icon="⚠️")
    
    # Copy button
    with col2:
        content = ai_response.get("content", "")
        if st.button(
            "📋 Copy",
            key="diagnosis_copy",
            use_container_width=True,
        ):
            st.code(content, language="text")
            st.toast("✅ Copied to clipboard", icon="✅")
    
    # Export button
    with col3:
        if st.button(
            "📤 Export",
            key="diagnosis_export",
            use_container_width=True,
        ):
            export_data = {
                "timestamp": now_str(),
                "patient": patient_data,
                "risk_assessment": risk_result,
                "ai_diagnosis": ai_response,
            }
            json_str = json.dumps(export_data, indent=2, default=str)
            st.download_button(
                label="⬇️ Download JSON",
                data=json_str,
                file_name=f"diagnosis_{now_str().replace(' ', '_').replace(':', '-')}.json",
                mime="application/json",
                key="diagnosis_download",
                use_container_width=True,
            )


def _render_disclaimer() -> None:
    """Render medical disclaimer."""
    st.markdown(
        """
        <div class="disclaimer">
            ⚠️ <strong>Medical Disclaimer:</strong> This assessment is AI-generated and 
            intended for clinical decision support only. It does not constitute a medical 
            diagnosis. Always verify findings with a licensed physician and follow your 
            institution's clinical protocols.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# State Renderers
# ─────────────────────────────────────────────

def _render_loading_state() -> None:
    """Render loading state."""
    st.markdown(
        """
        <div class="diagnosis-card">
            <div class="loading-state">
                🤖 Analyzing patient data...
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_error_state(error: str) -> None:
    """Render error state."""
    st.markdown(
        f"""
        <div class="diagnosis-card">
            <div class="error-state">
                ❌ <strong>AI Analysis Failed</strong><br>
                {error}<br><br>
                Please check your connection and try again.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_empty_state() -> None:
    """Render empty state."""
    st.info(
        "👈 Enter patient data and click **Analyze Patient** to generate AI diagnosis."
    )


# ─────────────────────────────────────────────
# Main Render Function
# ─────────────────────────────────────────────

def render_diagnosis_view(
    risk_result: Optional[Dict[str, Any]] = None,
    ai_response: Optional[Dict[str, Any]] = None,
    patient_data: Optional[Dict[str, Any]] = None,
    user_role: Optional[str] = None,
    loading: bool = False,
) -> None:
    """
    Render the complete diagnosis view.

    Parameters
    ----------
    risk_result  : dict — RiskResult.to_dict() or None
    ai_response  : dict — AIResponse.to_dict() or None
    patient_data : dict — patient input data or None
    user_role    : str — current user role for permissions
    loading      : bool — whether analysis is in progress
    """
    # Inject CSS
    st.markdown(DIAGNOSIS_CSS, unsafe_allow_html=True)
    
    # Handle loading state
    if loading:
        _render_loading_state()
        return
    
    # Handle empty state
    if not ai_response and not risk_result:
        _render_empty_state()
        return
    
    # Handle error state
    if ai_response and not ai_response.get("success", True):
        error = ai_response.get("error", "Unknown error")
        _render_error_state(error)
        
        # Still show fallback content if available
        content = ai_response.get("content", "")
        if content:
            st.markdown('<div class="diagnosis-card">', unsafe_allow_html=True)
            st.markdown("### 📋 Fallback Guidance")
            st.markdown(content)
            _render_disclaimer()
            st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Render diagnosis card
    st.markdown('<div class="diagnosis-card">', unsafe_allow_html=True)
    
    # Header
    st.markdown(
        """
        <div class="diagnosis-header">
            <div class="diagnosis-title">🤖 AI Diagnosis</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Risk badge and source info
    col_risk, col_source = st.columns([2, 1])
    with col_risk:
        _render_risk_badge(risk_result)
    
    with col_source:
        if ai_response:
            source = ai_response.get("source", "unknown")
            model = ai_response.get("model", "")
            source_info = _get_source_info(source, model)
            _render_source_badge(source_info)
    
    # Sections
    if ai_response:
        sections = ai_response.get("sections", {})
        if sections:
            for section_name, section_content in sections.items():
                _render_section(section_name, section_content)
        else:
            # Fallback: render raw content
            content = ai_response.get("content", "")
            if content:
                st.markdown(content)
    
    # Metadata
    if ai_response:
        _render_metadata(ai_response)
    
    # Action buttons
    st.markdown("---")
    _render_action_buttons(
        risk_result or {},
        ai_response or {},
        patient_data or {},
        user_role,
    )
    
    # Disclaimer
    _render_disclaimer()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Debug expander
    with st.expander("🔧 Diagnosis Debug", expanded=False):
        st.json({
            "risk_result": risk_result,
            "ai_response": ai_response,
        })