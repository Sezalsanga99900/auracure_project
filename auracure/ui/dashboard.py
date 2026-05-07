"""
ui/dashboard.py
─────────────────────────────────────────────────────────────────
AuraCure — Analytics Dashboard (Online Mode Only)
Purpose : Renders interactive Plotly charts using saved patient
          records from the database. Only shown when the system
          detects an internet connection (is_online = True).

What gets shown here
  1. Risk distribution donut chart
  2. Vital trends line chart (BP, HR over saved patients)
  3. Symptom frequency bar chart
  4. Age vs cholesterol scatter plot
  5. Diagnosis breakdown horizontal bar chart

Dependencies used here
  • streamlit — layout and rendering
  • plotly    — all interactive charts
  • pandas    — data manipulation for charts
  • Data comes from database/local_db.py (passed in as DataFrame)
─────────────────────────────────────────────────────────────────
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


# ── Chart colour palette (matches the blue/white healthcare theme) ───────────

PALETTE = {
    "blue"       : "#3B5BDB",
    "light_blue" : "#93C5FD",
    "green"      : "#22C55E",
    "amber"      : "#F59E0B",
    "red"        : "#F43F5E",
    "purple"     : "#8B5CF6",
    "teal"       : "#14B8A6",
    "gray"       : "#9CA3AF",
}

RISK_COLORS = {
    "Low"    : PALETTE["green"],
    "Medium" : PALETTE["amber"],
    "High"   : PALETTE["red"],
}

PLOTLY_LAYOUT = dict(
    font_family = "DM Sans, sans-serif",
    font_color  = "#1F2937",
    paper_bgcolor = "rgba(0,0,0,0)",   # transparent background
    plot_bgcolor  = "rgba(0,0,0,0)",
    margin = dict(l=16, r=16, t=40, b=16),
    legend = dict(
        bgcolor     = "rgba(255,255,255,0.8)",
        bordercolor = "#E0E7FF",
        borderwidth = 1,
        font_size   = 12,
    ),
)

DASHBOARD_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600;700&display=swap');

.dashboard-header {
    background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 100%);
    border-radius: 16px;
    padding: 24px 30px;
    color: white;
    margin-bottom: 24px;
}
.dashboard-header h2 {
    font-size: 22px !important;
    font-weight: 700 !important;
    margin: 0 0 6px 0 !important;
    color: white !important;
}
.dashboard-header p { margin:0; font-size:13px; opacity:0.75; }

.chart-card {
    background: white;
    border: 1.5px solid #E0E7FF;
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 2px 10px rgba(59,91,219,0.05);
}
.chart-title {
    font-size: 13px; font-weight: 700;
    color: #1E3A8A; letter-spacing: 0.05em;
    text-transform: uppercase; margin-bottom: 14px;
}
.stat-card {
    background: white;
    border: 1.5px solid #E0E7FF;
    border-radius: 12px;
    padding: 18px 20px;
    text-align: center;
}
.stat-value {
    font-size: 32px; font-weight: 800;
    color: #1E3A8A; line-height: 1;
}
.stat-label {
    font-size: 11px; font-weight: 600;
    color: #6B7AB8; text-transform: uppercase;
    letter-spacing: 0.08em; margin-top: 6px;
}
.stat-delta {
    font-size: 12px; font-weight: 600;
    margin-top: 4px;
}
.stat-delta.up   { color: #22C55E; }
.stat-delta.down { color: #F43F5E; }

.no-data-box {
    text-align: center; padding: 60px 20px;
    color: #9CA3AF; background: #F8FAFF;
    border-radius: 14px; border: 1.5px dashed #C7D2FE;
}
</style>
"""


# ── Dummy data generator (used when DB has < 10 records for demo) ──────────

def _generate_demo_df(n: int = 80) -> pd.DataFrame:
    """
    Creates a realistic synthetic DataFrame for chart demo purposes.
    In production this is replaced by real records from local_db.py.
    """
    import numpy as np
    rng = np.random.default_rng(42)

    ages         = rng.integers(28, 82, n)
    bp_sys       = rng.integers(100, 190, n)
    heart_rate   = rng.integers(55, 130, n)
    cholesterol  = rng.integers(150, 320, n)
    glucose      = rng.integers(80, 250, n)

    risks = rng.choice(["Low", "Medium", "High"],
                       size=n, p=[0.35, 0.40, 0.25])

    diagnoses = rng.choice([
        "Coronary Artery Disease",
        "Hypertensive Heart Disease",
        "Arrhythmia",
        "Heart Failure",
        "Stable Angina",
        "Myocardial Infarction",
    ], size=n)

    symptoms_pool = [
        "Chest pain", "Shortness of breath",
        "Palpitations", "Fatigue",
        "Dizziness", "Swelling",
    ]
    symptoms = [
        ", ".join(rng.choice(symptoms_pool,
                             size=rng.integers(1, 4), replace=False).tolist())
        for _ in range(n)
    ]

    dates = pd.date_range(end=pd.Timestamp.today(), periods=n, freq="3D")

    return pd.DataFrame({
        "date"        : dates,
        "age"         : ages,
        "bp_systolic" : bp_sys,
        "heart_rate"  : heart_rate,
        "cholesterol" : cholesterol,
        "glucose"     : glucose,
        "risk_level"  : risks,
        "diagnosis"   : diagnoses,
        "symptoms"    : symptoms,
    })


# ── Individual chart builders ─────────────────────────────────────────────────

def _chart_risk_donut(df: pd.DataFrame) -> go.Figure:
    """Risk level distribution donut chart."""
    counts = df["risk_level"].value_counts().reindex(
        ["Low", "Medium", "High"], fill_value=0
    )
    fig = go.Figure(go.Pie(
        labels  = counts.index.tolist(),
        values  = counts.values.tolist(),
        hole    = 0.60,
        marker  = dict(
            colors = [RISK_COLORS[r] for r in counts.index],
            line   = dict(color="white", width=2),
        ),
        textinfo     = "label+percent",
        textfont     = dict(size=12),
        hovertemplate= "<b>%{label}</b><br>Patients: %{value}<br>Share: %{percent}<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        showlegend  = False,
        height      = 280,
        annotations = [dict(
            text      = f"<b>{len(df)}</b><br><span style='font-size:11px'>patients</span>",
            x=0.5, y=0.5, font_size=18, showarrow=False,
            font_color="#1E3A8A",
        )],
    )
    return fig


def _chart_vitals_trend(df: pd.DataFrame) -> go.Figure:
    """BP systolic and heart rate over time."""
    df_sorted = df.sort_values("date")
    # Rolling average for smoother lines
    df_sorted["bp_ma"]  = df_sorted["bp_systolic"].rolling(5, min_periods=1).mean()
    df_sorted["hr_ma"]  = df_sorted["heart_rate"].rolling(5, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_sorted["date"], y=df_sorted["bp_ma"],
        name="Systolic BP", mode="lines",
        line=dict(color=PALETTE["blue"], width=2.5),
        hovertemplate="BP: %{y:.0f} mmHg<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df_sorted["date"], y=df_sorted["hr_ma"],
        name="Heart Rate", mode="lines",
        line=dict(color=PALETTE["red"], width=2.5, dash="dash"),
        hovertemplate="HR: %{y:.0f} bpm<extra></extra>",
    ))
    # Reference lines
    fig.add_hline(y=140, line_dash="dot",
                  line_color=PALETTE["amber"], opacity=0.5,
                  annotation_text="BP alert (140)", annotation_font_size=10)
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=280,
        xaxis=dict(showgrid=False, title=""),
        yaxis=dict(showgrid=True, gridcolor="#F3F4F6",
                   title="Value", title_font_size=11),
    )
    return fig


def _chart_symptom_frequency(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of most common symptoms."""
    from collections import Counter
    all_symptoms = []
    for row in df["symptoms"].dropna():
        all_symptoms.extend([s.strip() for s in str(row).split(",")])

    counts = Counter(all_symptoms).most_common(8)
    if not counts:
        return None

    labels = [c[0] for c in reversed(counts)]
    values = [c[1] for c in reversed(counts)]

    fig = go.Figure(go.Bar(
        x           = values,
        y           = labels,
        orientation = "h",
        marker      = dict(
            color     = values,
            colorscale= [[0, PALETTE["light_blue"]], [1, PALETTE["blue"]]],
            showscale = False,
            line      = dict(color="white", width=0.5),
        ),
        hovertemplate = "<b>%{y}</b><br>Patients: %{x}<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=300,
        xaxis=dict(showgrid=True, gridcolor="#F3F4F6", title="Patient count"),
        yaxis=dict(showgrid=False, automargin=True),
    )
    return fig


def _chart_age_cholesterol(df: pd.DataFrame) -> go.Figure:
    """Scatter: age vs cholesterol, coloured by risk level."""
    fig = go.Figure()
    for risk, color in RISK_COLORS.items():
        subset = df[df["risk_level"] == risk]
        fig.add_trace(go.Scatter(
            x    = subset["age"],
            y    = subset["cholesterol"],
            mode = "markers",
            name = f"{risk} Risk",
            marker= dict(
                color   = color,
                size    = 9,
                opacity = 0.75,
                line    = dict(color="white", width=1),
            ),
            hovertemplate=(
                f"<b>{risk} Risk</b><br>"
                "Age: %{x}<br>Cholesterol: %{y} mg/dL<extra></extra>"
            ),
        ))
    # Threshold lines
    fig.add_vline(x=50, line_dash="dot", line_color=PALETTE["gray"],
                  opacity=0.5, annotation_text="Age 50", annotation_font_size=10)
    fig.add_hline(y=240, line_dash="dot", line_color=PALETTE["amber"],
                  opacity=0.5, annotation_text="High chol.", annotation_font_size=10)
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=300,
        xaxis=dict(showgrid=True, gridcolor="#F3F4F6", title="Age (years)"),
        yaxis=dict(showgrid=True, gridcolor="#F3F4F6", title="Cholesterol (mg/dL)"),
    )
    return fig


def _chart_diagnosis_breakdown(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar — how many patients per diagnosis."""
    counts = df["diagnosis"].value_counts().head(7)
    colors = [PALETTE["blue"], PALETTE["teal"], PALETTE["purple"],
              PALETTE["amber"], PALETTE["red"], PALETTE["green"],
              PALETTE["light_blue"]]

    fig = go.Figure(go.Bar(
        x           = counts.values,
        y           = counts.index,
        orientation = "h",
        marker_color= colors[:len(counts)],
        hovertemplate = "<b>%{y}</b><br>Count: %{x}<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=300,
        xaxis=dict(showgrid=True, gridcolor="#F3F4F6", title="Patient count"),
        yaxis=dict(showgrid=False, automargin=True),
    )
    return fig


# ── Summary stats row ─────────────────────────────────────────────────────────

def _render_stat_cards(df: pd.DataFrame):
    """4 KPI cards above the charts."""
    total     = len(df)
    high_risk = int((df["risk_level"] == "High").sum())
    avg_bp    = int(df["bp_systolic"].mean())
    avg_age   = int(df["age"].mean())

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{total}</div>
            <div class="stat-label">Total Patients</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value" style="color:#F43F5E">{high_risk}</div>
            <div class="stat-label">High Risk Cases</div>
            <div class="stat-delta {'up' if high_risk > total*0.2 else 'down'}">
                {high_risk/total*100:.0f}% of total
            </div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{avg_bp}</div>
            <div class="stat-label">Avg Systolic BP</div>
            <div class="stat-delta {'up' if avg_bp > 130 else 'down'}">
                {'↑ Above normal' if avg_bp > 130 else '✓ Normal range'}
            </div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{avg_age}</div>
            <div class="stat-label">Average Age</div>
        </div>""", unsafe_allow_html=True)


# ── Main render function ──────────────────────────────────────────────────────

def render_dashboard(records_df: pd.DataFrame = None):
    """
    Main entry point for the analytics dashboard tab.
    Called from app.py only when is_online is True.

    Parameters
    ----------
    records_df : pd.DataFrame | None
        Patient records from database/local_db.py.
        If None or too few rows, demo data is used automatically.
    """
    st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)

    # Use real data if available, else demo
    if records_df is None or len(records_df) < 5:
        df = _generate_demo_df(80)
        st.info(
            "📊 Showing **demo analytics** — based on synthetic data. "
            "Real patient records will appear here as doctors use the system.",
            icon="ℹ️"
        )
    else:
        df = records_df

    # Header banner
    st.markdown(f"""
    <div class="dashboard-header">
        <h2>📈 Cardiac Analytics Dashboard</h2>
        <p>Real-time insights from {len(df)} patient records · Online mode active</p>
    </div>
    """, unsafe_allow_html=True)

    # KPI stat cards
    _render_stat_cards(df)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1: Donut + Vitals trend ─────────────────────────────────────────
    col1, col2 = st.columns([1, 1.6])

    with col1:
        st.markdown('<div class="chart-card">'
                    '<div class="chart-title">🎯 Risk Distribution</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(_chart_risk_donut(df),
                        use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="chart-card">'
                    '<div class="chart-title">📈 Vitals Trend Over Time</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(_chart_vitals_trend(df),
                        use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Row 2: Symptom freq + Age vs Cholesterol ────────────────────────────
    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="chart-card">'
                    '<div class="chart-title">🩺 Symptom Frequency</div>',
                    unsafe_allow_html=True)
        fig_sym = _chart_symptom_frequency(df)
        if fig_sym:
            st.plotly_chart(fig_sym, use_container_width=True,
                            config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="chart-card">'
                    '<div class="chart-title">🔬 Age vs Cholesterol by Risk</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(_chart_age_cholesterol(df),
                        use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Row 3: Diagnosis breakdown (full width) ─────────────────────────────
    st.markdown('<div class="chart-card">'
                '<div class="chart-title">🏥 Diagnosis Breakdown</div>',
                unsafe_allow_html=True)
    st.plotly_chart(_chart_diagnosis_breakdown(df),
                    use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Footer note ─────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; font-size:11px; color:#9CA3AF; padding:16px 0;">
        AuraCure Analytics · Data shown is for clinical decision support purposes only.<br>
        All patient data is stored locally and synced securely when online.
    </div>
    """, unsafe_allow_html=True) 
