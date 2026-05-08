# =============================================================================
# ui/dashboard.py
# AuraEcho+ — Patient Dashboard Component
#
# Responsibility:
#     Render interactive visualizations of patient vitals, risk assessment,
#     and clinical indicators using Plotly. Provides at-a-glance clinical
#     overview with threshold-aware color coding.
#
# Sections:
#     1. Patient Header Card
#     2. Risk Gauge
#     3. Vitals Gauges Grid
#     4. Feature Radar Chart
#     5. Categorical Summary
#
# Public API:
#     render_dashboard(patient_data, risk_result) → None
# =============================================================================

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Any, Dict, List, Optional, Tuple

from utils.constants import (
    APP_NAME,
    FEATURE_COLUMNS,
    FEATURE_LABELS,
    FEATURE_RANGES,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    RISK_LEVELS,
    RISK_LABELS,
    RISK_COLORS,
    RISK_ICONS,
    UI_PRIMARY_COLOR,
    UI_SUCCESS_COLOR,
    UI_WARNING_COLOR,
    UI_DANGER_COLOR,
    UI_CARD_COLOR,
    CHART_THEME,
    CHART_FONT_FAMILY,
    CHEST_PAIN_LABELS,
    THAL_LABELS,
    SLOPE_LABELS,
    RESTECG_LABELS,
)
from utils.helpers import get_logger, format_score, patient_to_display

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# CSS Styles
# ─────────────────────────────────────────────

DASHBOARD_CSS = """
<style>
    /* Dashboard card */
    .dash-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid #e8f0fe;
        height: 100%;
    }
    
    /* Patient header */
    .patient-header {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    .patient-name {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1a1a2e;
    }
    .patient-meta {
        font-size: 0.9rem;
        color: #5f6368;
    }
    
    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    .status-normal {
        background-color: #d4edda;
        color: #155724;
    }
    .status-elevated {
        background-color: #fff3cd;
        color: #856404;
    }
    .status-high {
        background-color: #f8d7da;
        color: #721c24;
    }
    
    /* Chart container */
    .chart-container {
        margin-bottom: 1rem;
    }
    
    /* Section title */
    .dash-section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1a1a2e;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Categorical grid */
    .cat-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
        gap: 0.75rem;
    }
    .cat-item {
        background-color: #f8f9fa;
        padding: 0.75rem;
        border-radius: 8px;
        border-left: 3px solid #1a73e8;
    }
    .cat-label {
        font-size: 0.8rem;
        color: #5f6368;
        margin-bottom: 0.25rem;
    }
    .cat-value {
        font-size: 0.95rem;
        font-weight: 500;
        color: #1a1a2e;
    }
</style>
"""


# ─────────────────────────────────────────────
# Clinical Threshold Helpers
# ─────────────────────────────────────────────

# Clinical thresholds for status determination
_CLINICAL_THRESHOLDS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "trestbps": {
        "normal": (0, 120),
        "elevated": (120, 130),
        "high": (130, 300),
    },
    "chol": {
        "normal": (0, 200),
        "elevated": (200, 240),
        "high": (240, 700),
    },
    "thalach": {
        "low": (0, 100),
        "normal": (100, 180),
        "high": (180, 300),
    },
    "oldpeak": {
        "normal": (0, 1.0),
        "elevated": (1.0, 2.0),
        "high": (2.0, 10.0),
    },
}


def _get_vital_status(feature: str, value: float) -> Tuple[str, str]:
    """
    Determine vital status and color based on clinical thresholds.

    Returns
    -------
    (status_label, color)
        status_label: "Normal" | "Elevated" | "High" | "Low"
        color: hex color code
    """
    if feature not in _CLINICAL_THRESHOLDS:
        return "Normal", UI_SUCCESS_COLOR

    thresholds = _CLINICAL_THRESHOLDS[feature]

    for status, (lo, hi) in thresholds.items():
        if lo <= value < hi:
            if status == "normal":
                return "Normal", UI_SUCCESS_COLOR
            elif status == "elevated":
                return "Elevated", UI_WARNING_COLOR
            elif status in ("high", "low"):
                return status.capitalize(), UI_DANGER_COLOR

    return "Unknown", "#95a5a6"


# ─────────────────────────────────────────────
# Chart Builders
# ─────────────────────────────────────────────

def _create_risk_gauge(risk_result: Dict[str, Any]) -> go.Figure:
    """Create risk score gauge chart."""
    score = risk_result.get("disease_prob", 0)
    risk_level = risk_result.get("risk_level", "MEDIUM")
    risk_label = risk_result.get("risk_label", RISK_LABELS.get(risk_level, risk_level))
    risk_color = RISK_COLORS.get(risk_level, "#95a5a6")
    confidence = risk_result.get("confidence_pct", 0)

    # Define gauge steps
    steps = [
        {"range": [0, 0.35], "color": RISK_COLORS["LOW"]},
        {"range": [0.35, 0.65], "color": RISK_COLORS["MEDIUM"]},
        {"range": [0.65, 1.0], "color": RISK_COLORS["HIGH"]},
    ]

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score * 100,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Cardiac Risk Score", "font": {"size": 16}},
        delta={"reference": 50, "suffix": "%"},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": risk_color},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "#e0e0e0",
            "steps": steps,
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": 90,
            },
        },
        number={"suffix": "%", "font": {"size": 28}},
    ))

    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": CHART_FONT_FAMILY},
        annotations=[
            dict(
                text=f"{risk_label}<br>Confidence: {confidence:.0f}%",
                x=0.5,
                y=-0.1,
                xref="paper",
                yref="paper",
                showarrow=False,
                font={"size": 12, "color": "#555"},
            )
        ],
    )

    return fig


def _create_vital_gauge(
    value: float,
    feature: str,
    label: str,
) -> go.Figure:
    """Create a single vital sign gauge."""
    status, color = _get_vital_status(feature, value)
    min_val, max_val = FEATURE_RANGES.get(feature, (0, 100))

    # Determine unit
    units = {
        "trestbps": "mm Hg",
        "chol": "mg/dl",
        "thalach": "bpm",
        "oldpeak": "",
        "age": "yr",
    }
    unit = units.get(feature, "")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": label, "font": {"size": 12}},
        gauge={
            "axis": {"range": [min_val, max_val], "tickwidth": 1},
            "bar": {"color": color},
            "bgcolor": "white",
            "borderwidth": 1,
            "bordercolor": "#e0e0e0",
            "steps": [
                {"range": [min_val, max_val], "color": "#f0f0f0"},
            ],
        },
        number={"suffix": f" {unit}", "font": {"size": 20}},
    ))

    fig.update_layout(
        height=180,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": CHART_FONT_FAMILY},
        annotations=[
            dict(
                text=status,
                x=0.5,
                y=-0.1,
                xref="paper",
                yref="paper",
                showarrow=False,
                font={"size": 10, "color": color, "weight": "bold"},
            )
        ],
    )

    return fig


def _create_feature_radar(risk_result: Dict[str, Any]) -> go.Figure:
    """Create radar chart for top contributing features."""
    contributions = risk_result.get("feature_contributions", [])
    if not contributions:
        return None

    # Top 6 features
    top_features = contributions[:6]
    features = [c["feature"] for c in top_features]
    importances = [c["importance"] for c in top_features]

    # Normalize to 0-100 for radar
    max_imp = max(importances) if importances else 1
    values = [(v / max_imp) * 100 for v in importances]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=features,
        fill="toself",
        fillcolor=UI_PRIMARY_COLOR + "40",
        line={"color": UI_PRIMARY_COLOR, "width": 2},
        name="Feature Importance",
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickmode="array",
                tickvals=[0, 25, 50, 75, 100],
                ticktext=["0", "25", "50", "75", "100"],
            ),
            angularaxis=dict(
                tickfont={"size": 10},
            ),
        ),
        showlegend=False,
        height=300,
        margin=dict(l=40, r=40, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": CHART_FONT_FAMILY},
    )

    return fig


# ─────────────────────────────────────────────
# Render Functions
# ─────────────────────────────────────────────

def _render_patient_header(patient_data: Dict[str, Any]) -> None:
    """Render patient header card."""
    name = patient_data.get("name", "Unknown Patient")
    age = patient_data.get("age", "?")
    sex = patient_data.get("sex", 1)
    sex_label = "Male" if sex == 1 else "Female"

    st.markdown(
        f"""
        <div class="dash-card">
            <div class="patient-header">
                <div style="font-size: 2.5rem;">🫀</div>
                <div>
                    <div class="patient-name">{name}</div>
                    <div class="patient-meta">
                        {age} years · {sex_label}
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_risk_section(risk_result: Dict[str, Any]) -> None:
    """Render risk gauge section."""
    st.markdown(
        """
        <div class="dash-section-title">
            📊 Cardiac Risk Assessment
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="dash-card">', unsafe_allow_html=True)
    fig = _create_risk_gauge(risk_result)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)


def _render_vitals_section(patient_data: Dict[str, Any]) -> None:
    """Render vitals gauges grid."""
    st.markdown(
        """
        <div class="dash-section-title">
            💓 Vital Signs
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Numerical features to display
    display_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]

    cols = st.columns(len(display_features))
    for i, feature in enumerate(display_features):
        with cols[i]:
            value = patient_data.get(feature, 0)
            label = FEATURE_LABELS.get(feature, feature)
            # Clean label for gauge
            label_short = label.split("(")[0].strip()

            fig = _create_vital_gauge(value, feature, label_short)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _render_radar_section(risk_result: Dict[str, Any]) -> None:
    """Render feature importance radar chart."""
    st.markdown(
        """
        <div class="dash-section-title">
            🎯 Key Risk Contributors
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="dash-card">', unsafe_allow_html=True)
    fig = _create_feature_radar(risk_result)
    if fig:
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("Feature contributions not available.")
    st.markdown('</div>', unsafe_allow_html=True)


def _render_categorical_section(patient_data: Dict[str, Any]) -> None:
    """Render categorical features summary."""
    st.markdown(
        """
        <div class="dash-section-title">
            📋 Clinical Findings
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="dash-card">', unsafe_allow_html=True)

    # Decode maps
    decode_maps = {
        "cp": CHEST_PAIN_LABELS,
        "restecg": RESTECG_LABELS,
        "slope": SLOPE_LABELS,
        "thal": THAL_LABELS,
        "fbs": {0: "Normal", 1: "Elevated"},
        "exang": {0: "Absent", 1: "Present"},
    }

    st.markdown('<div class="cat-grid">', unsafe_allow_html=True)

    for feature in ["cp", "restecg", "slope", "thal", "fbs", "exang"]:
        value = patient_data.get(feature, 0)
        label = FEATURE_LABELS.get(feature, feature)
        decode = decode_maps.get(feature, {})
        display_value = decode.get(int(value), str(value))

        st.markdown(
            f"""
            <div class="cat-item">
                <div class="cat-label">{label}</div>
                <div class="cat-value">{display_value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # CA (major vessels)
    ca = patient_data.get("ca", 0)
    st.markdown(
        f"""
        <div class="cat-item">
            <div class="cat-label">{FEATURE_LABELS.get("ca", "Major Vessels")}</div>
            <div class="cat-value">{ca} vessel(s)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Main Render Function
# ─────────────────────────────────────────────

def render_dashboard(
    patient_data: Dict[str, Any],
    risk_result: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Render the complete patient dashboard.

    Parameters
    ----------
    patient_data : dict — patient input data
    risk_result  : dict — RiskResult.to_dict() or None
    """
    # Inject CSS
    st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)

    # Check for patient data
    if not patient_data:
        st.info("👈 Enter patient data in the sidebar to view the dashboard.")
        return

    # Patient header
    _render_patient_header(patient_data)

    # Layout based on risk result availability
    if risk_result:
        # Two-column layout with risk and radar
        col_risk, col_radar = st.columns([1, 1])

        with col_risk:
            _render_risk_section(risk_result)

        with col_radar:
            _render_radar_section(risk_result)

        # Vitals section
        _render_vitals_section(patient_data)

        # Categorical section
        _render_categorical_section(patient_data)

    else:
        # No risk result — show vitals and categorical only
        _render_vitals_section(patient_data)
        _render_categorical_section(patient_data)

        st.info(
            "💡 Click **Analyze Patient** in the sidebar to generate risk assessment."
        )

    # Debug expander
    with st.expander("🔧 Dashboard Debug", expanded=False):
        st.json({
            "patient_data": patient_data,
            "risk_result": risk_result,
        })