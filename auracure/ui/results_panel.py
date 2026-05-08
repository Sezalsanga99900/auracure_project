# =============================================================================
# ui/results_panel.py
# AuraEcho+ — Results Panel Component
#
# Responsibility:
#     Display cardiac risk assessment results, AI diagnosis, similar cases,
#     and feature contributions. Provides action buttons for save, export,
#     and regenerate.
#
# Sections:
#     1. Risk Summary Card
#     2. AI Diagnosis Panel
#     3. Similar Cases Carousel
#     4. Feature Contributions Chart
#     5. Action Buttons
#
# Public API:
#     render_results_panel(risk_result, ai_response, similar_cases) → None
# =============================================================================

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
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
    RESULTS_TOP_N_CASES,
    ROLE_PERMISSIONS,
)
from utils.helpers import (
    get_logger,
    format_score,
    format_similarity,
    patient_to_display,
    now_str,
)
from services.auth_service import user_has_permission
from database.local_db import save_patient, save_prediction

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# CSS Styles
# ─────────────────────────────────────────────

RESULTS_CSS = """
<style>
    /* Result card */
    .result-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid #e8f0fe;
    }
    
    /* Risk badge */
    .risk-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 1rem;
        color: white;
        margin-bottom: 0.5rem;
    }
    
    /* Section header */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1a1a2e;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* AI response section */
    .ai-section {
        margin-bottom: 1rem;
    }
    .ai-section-title {
        font-weight: 600;
        color: #1a73e8;
        margin-bottom: 0.5rem;
        font-size: 0.95rem;
    }
    .ai-section-content {
        color: #333;
        line-height: 1.6;
        font-size: 0.95rem;
    }
    
    /* Similar case card */
    .similar-case {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        border-left: 4px solid #1a73e8;
    }
    .similar-case-header {
        font-weight: 600;
        margin-bottom: 0.5rem;
        display: flex;
        justify-content: space-between;
    }
    .similar-case-details {
        font-size: 0.85rem;
        color: #555;
    }
    
    /* Feature bar */
    .feature-bar-label {
        font-size: 0.85rem;
        color: #333;
        margin-bottom: 0.25rem;
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
    
    /* Source badge */
    .source-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 10px;
        font-size: 0.7rem;
        font-weight: 500;
        background-color: #e8f0fe;
        color: #1a73e8;
        margin-left: 0.5rem;
    }
</style>
"""


# ─────────────────────────────────────────────
# Helper render functions
# ─────────────────────────────────────────────

def _render_risk_badge(risk_result: Dict[str, Any]) -> None:
    """Render the risk level badge with score and confidence."""
    risk_level = risk_result.get("risk_level", "MEDIUM")
    risk_label = risk_result.get("risk_label", RISK_LABELS.get(risk_level, risk_level))
    risk_color = RISK_COLORS.get(risk_level, "#95a5a6")
    risk_icon = risk_result.get("badge_icon", RISK_ICONS.get(risk_level, "❔"))
    disease_prob = risk_result.get("disease_prob", 0)
    confidence = risk_result.get("confidence_pct", 0)

    st.markdown(
        f"""
        <div class="risk-badge" style="background-color: {risk_color};">
            {risk_icon} {risk_label}
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Disease Probability", format_score(disease_prob))
    with col2:
        st.metric("Confidence", f"{confidence:.1f}%")


def _render_ai_response(ai_response: Optional[Dict[str, Any]]) -> None:
    """Render AI diagnosis response with sections."""
    if not ai_response:
        st.warning("⚠️ AI analysis unavailable. Using rule-based guidance.")
        return

    if not ai_response.get("success", True):
        st.error(f"❌ AI Error: {ai_response.get('error', 'Unknown error')}")
        return

    content = ai_response.get("content", "")
    sections = ai_response.get("sections", {})
    source = ai_response.get("source", "unknown")
    model = ai_response.get("model", "")

    # Source badge
    source_label = {
        "offline_llama3": "🖥️ Local AI",
        "online_groq": "☁️ Groq",
        "online_openai": "☁️ OpenAI",
        "offline_fallback": "📋 Rule-Based",
    }.get(source, source)

    st.markdown(
        f"""
        <div class="section-header">
            🤖 AI Diagnosis
            <span class="source-badge">{source_label} · {model}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Render sections
    section_icons = {
        "Clinical Assessment": "🔍",
        "Key Risk Indicators": "⚠️",
        "Potential Future Symptoms": "🔮",
        "Treatment Recommendations": "💊",
        "Referral & Follow-up": "🏥",
        "Patient Education Points": "📋",
    }

    for section_name, section_content in sections.items():
        icon = section_icons.get(section_name, "📝")
        st.markdown(
            f"""
            <div class="ai-section">
                <div class="ai-section-title">{icon} {section_name}</div>
                <div class="ai-section-content">{section_content}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Latency info
    latency = ai_response.get("latency_ms", 0)
    st.caption(f"⏱️ Generated in {latency:.0f} ms")


def _render_similar_cases(similar_cases: Optional[List[Dict[str, Any]]]) -> None:
    """Render similar historical cases."""
    st.markdown(
        """
        <div class="section-header">
            👥 Similar Historical Cases
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not similar_cases:
        st.info("No similar cases found in the reference database.")
        return

    for case in similar_cases[:RESULTS_TOP_N_CASES]:
        rank = case.get("rank", "?")
        sim_pct = case.get("similarity_pct", 0)
        age = case.get("age", "?")
        sex = case.get("sex", "?")
        outcome = case.get("outcome", "?")
        risk_label = case.get("risk_label", "?")
        risk_color = case.get("risk_color", "#95a5a6")
        risk_icon = case.get("risk_icon", "❔")

        outcome_icon = "🔴" if outcome == "Disease" else "🟢"

        st.markdown(
            f"""
            <div class="similar-case" style="border-left-color: {risk_color};">
                <div class="similar-case-header">
                    <span>#{rank} Match · {format_similarity(sim_pct/100)}</span>
                    <span>{risk_icon} {risk_label}</span>
                </div>
                <div class="similar-case-details">
                    {age}yr {sex} · {outcome_icon} {outcome}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_feature_contributions(risk_result: Dict[str, Any]) -> None:
    """Render feature importance chart."""
    st.markdown(
        """
        <div class="section-header">
            📊 Key Contributing Factors
        </div>
        """,
        unsafe_allow_html=True,
    )

    contributions = risk_result.get("feature_contributions", [])
    if not contributions:
        st.info("Feature contributions not available.")
        return

    # Prepare data
    features = [c["feature"] for c in contributions[:7]]  # Top 7
    importances = [c["importance"] for c in contributions[:7]]

    # Create horizontal bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=features,
        x=importances,
        orientation='h',
        marker_color=UI_PRIMARY_COLOR,
        text=[f"{v:.1%}" for v in importances],
        textposition='outside',
    ))

    fig.update_layout(
        height=300,
        margin=dict(l=150, r=20, t=20, b=20),
        xaxis=dict(title="Importance", range=[0, max(importances) * 1.2]),
        yaxis=dict(autorange="reversed"),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _render_action_buttons(
    patient_data: Dict[str, Any],
    risk_result: Dict[str, Any],
    ai_response: Optional[Dict[str, Any]],
    user_role: Optional[str] = None,
) -> None:
    """Render action buttons based on user permissions."""
    st.markdown("### 🛠️ Actions")

    col1, col2, col3 = st.columns(3)

    # Save button
    can_save = user_has_permission({"role": user_role}, "edit_patient") if user_role else True
    with col1:
        if st.button("💾 Save", key="save_btn", use_container_width=True, disabled=not can_save):
            if can_save:
                try:
                    patient_id = save_patient(patient_data)
                    save_prediction(patient_id, risk_result, ai_response)
                    st.toast("✅ Patient and prediction saved", icon="✅")
                    logger.info("Saved patient id=%d", patient_id)
                except Exception as exc:
                    st.toast(f"❌ Save failed: {exc}", icon="❌")
                    logger.error("Save failed: %s", exc)
            else:
                st.toast("⚠️ You don't have permission to save", icon="⚠️")

    # Export button
    with col2:
        if st.button("📤 Export", key="export_btn", use_container_width=True):
            # Create export data
            export_data = {
                "timestamp": now_str(),
                "patient": patient_data,
                "risk_assessment": risk_result,
                "ai_diagnosis": ai_response,
            }
            import json
            json_str = json.dumps(export_data, indent=2, default=str)
            st.download_button(
                label="⬇️ Download JSON",
                data=json_str,
                file_name=f"auraecho_report_{now_str().replace(' ', '_').replace(':', '-')}.json",
                mime="application/json",
                key="download_json",
                use_container_width=True,
            )

    # Regenerate AI button
    can_regenerate = user_has_permission({"role": user_role}, "view_ai_insights") if user_role else True
    with col3:
        if st.button("🔄 Regenerate AI", key="regen_btn", use_container_width=True, disabled=not can_regenerate):
            if can_regenerate:
                st.session_state["regenerate_ai"] = True
                st.rerun()
            else:
                st.toast("⚠️ You don't have permission to regenerate AI", icon="⚠️")


def _render_disclaimer() -> None:
    """Render AI disclaimer."""
    st.markdown(
        """
        <div class="disclaimer">
            ⚠️ <strong>Medical Disclaimer:</strong> This assessment is AI-generated and intended 
            for clinical decision support only. It does not constitute a medical diagnosis. 
            Always verify findings with a licensed physician and follow your institution's 
            clinical protocols.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# Main render function
# ─────────────────────────────────────────────

def render_results_panel(
    patient_data: Dict[str, Any],
    risk_result: Optional[Dict[str, Any]] = None,
    ai_response: Optional[Dict[str, Any]] = None,
    similar_cases: Optional[List[Dict[str, Any]]] = None,
    user_role: Optional[str] = None,
) -> None:
    """
    Render the complete results panel.

    Parameters
    ----------
    patient_data  : dict — patient input data
    risk_result   : dict — RiskResult.to_dict() or None
    ai_response   : dict — AIResponse.to_dict() or None
    similar_cases : list — SimilarCase.to_dict() list or None
    user_role     : str — current user role for permission checks
    """
    # Inject CSS
    st.markdown(RESULTS_CSS, unsafe_allow_html=True)

    # Check if we have results
    if not risk_result:
        st.info("👈 Enter patient data in the sidebar and click **Analyze Patient** to see results.")
        return

    # Risk summary card
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    _render_risk_badge(risk_result)
    st.markdown('</div>', unsafe_allow_html=True)

    # Two-column layout for AI and similar cases
    col_ai, col_sim = st.columns([2, 1])

    with col_ai:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        _render_ai_response(ai_response)
        st.markdown('</div>', unsafe_allow_html=True)

        # Feature contributions
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        _render_feature_contributions(risk_result)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_sim:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        _render_similar_cases(similar_cases)
        st.markdown('</div>', unsafe_allow_html=True)

    # Action buttons
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    _render_action_buttons(patient_data, risk_result, ai_response, user_role)
    st.markdown('</div>', unsafe_allow_html=True)

    # Disclaimer
    _render_disclaimer()

    # Debug expander
    with st.expander("🔧 Debug Info", expanded=False):
        st.json({
            "risk_result": risk_result,
            "ai_response": ai_response,
            "similar_cases_count": len(similar_cases) if similar_cases else 0,
        })