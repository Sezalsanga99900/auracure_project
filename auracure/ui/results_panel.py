"""
ui/results_panel.py
─────────────────────────────────────────────────────────────────
AuraCure — Results Display Panel
Purpose : Renders the main right-side content area. Takes the
          output from AI + ML layers and presents it to the
          doctor in a clean, readable card layout.

What gets displayed here
  1. Patient summary header
  2. Cardiac Risk Badge  (from core/risk_model.py)
  3. AI Diagnosis        (from ai/offline_ai.py or online_ai.py)
  4. Top 3 Similar Cases (from core/similarity.py)
  5. Treatment guidance  (from AI response)
  6. Online-only extras  (drug check, guidelines — if is_online)

Dependencies used here
  • streamlit — all rendering
  • Outputs passed in as Python dicts (no direct imports from
    other project files — results_panel is purely presentational)
─────────────────────────────────────────────────────────────────
"""

import streamlit as st
import time


# ── CSS for the results panel ─────────────────────────────────────────────────

RESULTS_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');

/* ── Base ── */
.main .block-container { padding-top: 1.5rem !important; }
*, .stMarkdown { font-family: 'DM Sans', sans-serif !important; }

/* ── Page header ── */
.page-header {
    background: linear-gradient(135deg, #1E3A8A 0%, #1D4ED8 60%, #2563EB 100%);
    border-radius: 16px;
    padding: 28px 32px;
    color: white;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.page-header::before {
    content: '🫀';
    position: absolute;
    right: 28px; top: 50%;
    transform: translateY(-50%);
    font-size: 64px;
    opacity: 0.15;
}
.page-header h1 {
    font-family: 'DM Serif Display', serif !important;
    font-size: 28px !important; font-weight: 400 !important;
    margin: 0 0 6px 0 !important; color: white !important;
}
.page-header p { margin:0; font-size:13px; opacity:0.8; }
.header-badge {
    display:inline-block; background:rgba(255,255,255,0.2);
    border:1px solid rgba(255,255,255,0.3);
    border-radius:20px; padding:3px 12px;
    font-size:11px; font-weight:600; letter-spacing:0.05em;
    margin-top:10px;
}

/* ── Patient summary card ── */
.patient-summary {
    background: #F8FAFF;
    border: 1.5px solid #DBEAFE;
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 20px;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 16px;
}
.vital-item { text-align: center; }
.vital-value {
    font-size: 22px; font-weight: 700;
    color: #1E3A8A; line-height: 1;
}
.vital-label {
    font-size: 10px; font-weight: 600;
    color: #6B7AB8; letter-spacing: 0.08em;
    text-transform: uppercase; margin-top: 4px;
}

/* ── Risk badge ── */
.risk-card {
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 20px;
}
.risk-card.low    { background:#F0FDF4; border:2px solid #86EFAC; }
.risk-card.medium { background:#FFFBEB; border:2px solid #FCD34D; }
.risk-card.high   { background:#FFF1F2; border:2px solid #FDA4AF; }
.risk-icon { font-size: 42px; }
.risk-title { font-size: 12px; font-weight:600; letter-spacing:0.08em;
              text-transform:uppercase; opacity:0.7; }
.risk-level {
    font-size: 28px; font-weight: 800;
    line-height: 1.1;
}
.risk-card.low    .risk-level { color:#15803D; }
.risk-card.medium .risk-level { color:#B45309; }
.risk-card.high   .risk-level { color:#BE123C; }
.risk-desc { font-size: 12px; color: #4B5563; margin-top: 4px; }
.risk-score-bar {
    flex: 1;
    background: rgba(0,0,0,0.06);
    border-radius: 99px;
    height: 8px;
    overflow: hidden;
}
.risk-score-fill {
    height: 100%; border-radius: 99px;
    transition: width 1s ease;
}
.risk-card.low    .risk-score-fill { background:#22C55E; }
.risk-card.medium .risk-score-fill { background:#F59E0B; }
.risk-card.high   .risk-score-fill { background:#F43F5E; }

/* ── Section title ── */
.section-title {
    font-size: 12px; font-weight: 700;
    letter-spacing: 0.10em; text-transform: uppercase;
    color: #3B5BDB; margin: 24px 0 12px 0;
    display: flex; align-items: center; gap: 8px;
}
.section-title::after {
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(to right, #C7D2FE, transparent);
}

/* ── AI diagnosis card ── */
.ai-card {
    background: white;
    border: 1.5px solid #E0E7FF;
    border-radius: 14px;
    padding: 22px 26px;
    margin-bottom: 20px;
    box-shadow: 0 2px 12px rgba(59,91,219,0.06);
}
.ai-card-header {
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 14px;
    padding-bottom: 12px;
    border-bottom: 1px solid #EEF2FF;
}
.ai-source-badge {
    font-size: 10px; font-weight: 700;
    letter-spacing: 0.08em; text-transform: uppercase;
    padding: 3px 10px; border-radius: 20px;
}
.ai-source-badge.online  { background:#DBEAFE; color:#1D4ED8; }
.ai-source-badge.offline { background:#FEE2E2; color:#DC2626; }
.ai-response-text {
    font-size: 14px; color: #1F2937;
    line-height: 1.75; white-space: pre-wrap;
}

/* ── Similar cases ── */
.case-card {
    background: white;
    border: 1.5px solid #E5E7EB;
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 12px;
    transition: border-color 0.2s, box-shadow 0.2s;
    position: relative;
}
.case-card:hover {
    border-color: #93C5FD;
    box-shadow: 0 4px 16px rgba(59,91,219,0.1);
}
.case-rank {
    position: absolute; top: -10px; left: 16px;
    background: #3B5BDB; color: white;
    font-size: 10px; font-weight: 700;
    padding: 2px 10px; border-radius: 20px;
    letter-spacing: 0.06em;
}
.case-similarity {
    font-size: 22px; font-weight: 800;
    color: #3B5BDB; float: right;
}
.case-diagnosis {
    font-size: 15px; font-weight: 600;
    color: #111827; margin: 8px 0 4px 0;
}
.case-meta {
    font-size: 12px; color: #6B7AB8;
    display: flex; gap: 16px; flex-wrap: wrap;
}
.case-treatment {
    font-size: 12px; color: #374151;
    background: #F0F4FF; border-radius: 8px;
    padding: 8px 12px; margin-top: 10px;
    line-height: 1.5;
}
.case-outcome {
    display: inline-block;
    font-size: 10px; font-weight: 600;
    padding: 2px 10px; border-radius: 20px;
    margin-top: 8px;
}
.case-outcome.recovered  { background:#DCFCE7; color:#15803D; }
.case-outcome.stable     { background:#DBEAFE; color:#1D4ED8; }
.case-outcome.monitoring { background:#FEF3C7; color:#B45309; }

/* ── Online extras ── */
.online-extra-card {
    background: linear-gradient(135deg, #EFF6FF 0%, #F0FDF4 100%);
    border: 1.5px solid #BFDBFE;
    border-radius: 14px;
    padding: 18px 22px;
    margin-bottom: 14px;
}
.online-extra-title {
    font-size: 12px; font-weight: 700;
    color: #1D4ED8; letter-spacing: 0.06em;
    text-transform: uppercase; margin-bottom: 10px;
}
.guideline-item {
    font-size: 13px; color: #1F2937;
    padding: 6px 0; border-bottom: 1px solid #DBEAFE;
    line-height: 1.5;
}
.guideline-item:last-child { border-bottom: none; }
.drug-tag {
    display: inline-block;
    background: #FEF3C7; color: #92400E;
    border: 1px solid #FDE68A;
    border-radius: 6px;
    font-size: 11px; font-weight: 600;
    padding: 3px 10px; margin: 3px 4px 3px 0;
}
.drug-interaction {
    font-size: 12px; color: #B91C1C;
    background: #FFF1F2; border-radius: 8px;
    padding: 8px 12px; margin-top: 8px;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 80px 40px;
    color: #9CA3AF;
}
.empty-state-icon { font-size: 72px; margin-bottom: 16px; }
.empty-state-title { font-size: 20px; font-weight: 600; color: #4B5563; }
.empty-state-sub   { font-size: 14px; margin-top: 8px; }

/* ── Loading animation ── */
.analysis-loading {
    text-align: center; padding: 40px;
    background: #F8FAFF; border-radius: 14px;
    border: 1.5px dashed #C7D2FE;
}
</style>
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _risk_config(risk_level: str) -> dict:
    """Map risk level string to display config."""
    configs = {
        "Low": {
            "cls": "low", "icon": "💚",
            "desc": "Routine monitoring recommended. Maintain healthy lifestyle.",
            "score_pct": "25%",
        },
        "Medium": {
            "cls": "medium", "icon": "🟡",
            "desc": "Further investigation advised. Schedule specialist review.",
            "score_pct": "60%",
        },
        "High": {
            "cls": "high", "icon": "🔴",
            "desc": "Urgent cardiac evaluation required. Consider immediate intervention.",
            "score_pct": "90%",
        },
    }
    return configs.get(risk_level, configs["Medium"])


def _outcome_class(outcome: str) -> str:
    """Map outcome string to CSS class."""
    o = outcome.lower()
    if "recover" in o:  return "recovered"
    if "stable" in o:   return "stable"
    return "monitoring"


# ── Public render functions ───────────────────────────────────────────────────

def render_empty_state():
    """
    Show the welcome / empty state before any patient is submitted.
    Called from app.py when no results are available yet.
    """
    st.markdown(RESULTS_CSS, unsafe_allow_html=True)
    st.markdown("""
    <div class="page-header">
        <h1>AuraCure</h1>
        <p>Hybrid AI-Powered Cardiac Decision Support System</p>
        <span class="header-badge">Clinical AI · Offline + Online</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="empty-state">
        <div class="empty-state-icon">🫀</div>
        <div class="empty-state-title">No patient analysed yet</div>
        <div class="empty-state-sub">
            Fill in the patient details on the left and click<br>
            <strong>Analyse Patient</strong> to get AI-powered cardiac insights.
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_patient_summary(patient: dict):
    """
    Render a compact vitals grid at the top of results.

    Parameters
    ----------
    patient : dict
        The validated patient dict returned by ui/sidebar.py
    """
    bp_str = f"{patient['bp_systolic']}/{patient['bp_diastolic']}"
    symptoms_short = (
        patient["symptoms"][0].split("or")[0].strip()
        if patient["symptoms"] else "None"
    )

    st.markdown(f"""
    <div class="patient-summary">
        <div class="vital-item">
            <div class="vital-value">{patient['age']}</div>
            <div class="vital-label">Age · {patient['gender'][0]}</div>
        </div>
        <div class="vital-item">
            <div class="vital-value">{bp_str}</div>
            <div class="vital-label">Blood Pressure</div>
        </div>
        <div class="vital-item">
            <div class="vital-value">{patient['heart_rate']}</div>
            <div class="vital-label">Heart Rate (bpm)</div>
        </div>
        <div class="vital-item">
            <div class="vital-value">{patient['cholesterol']}</div>
            <div class="vital-label">Cholesterol</div>
        </div>
        <div class="vital-item">
            <div class="vital-value">{patient['glucose']}</div>
            <div class="vital-label">Glucose</div>
        </div>
        <div class="vital-item">
            <div class="vital-value">{len(patient['symptoms'])}</div>
            <div class="vital-label">Symptoms</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_risk_badge(risk_level: str, risk_score: float = None):
    """
    Display the cardiac risk level prominently.

    Parameters
    ----------
    risk_level  : str   "Low", "Medium", or "High"
    risk_score  : float Optional 0–100 numeric score for the bar
    """
    cfg = _risk_config(risk_level)
    score_pct = f"{int(risk_score)}%" if risk_score is not None else cfg["score_pct"]

    st.markdown(f"""
    <div class="risk-card {cfg['cls']}">
        <div class="risk-icon">{cfg['icon']}</div>
        <div style="flex:1">
            <div class="risk-title">Cardiac Risk Level</div>
            <div class="risk-level">{risk_level.upper()} RISK</div>
            <div class="risk-desc">{cfg['desc']}</div>
            <div style="margin-top:10px">
                <div class="risk-score-bar">
                    <div class="risk-score-fill" style="width:{score_pct}"></div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_ai_diagnosis(ai_response: str, is_online: bool):
    """
    Display the AI-generated diagnosis and treatment text.

    Parameters
    ----------
    ai_response : str   Full text response from offline_ai or online_ai
    is_online   : bool  Controls which source badge is shown
    """
    source_cls   = "online"  if is_online else "offline"
    source_label = "Cloud AI · Enhanced" if is_online else "Local AI · Llama 3"

    st.markdown('<div class="section-title">🧠 AI Diagnosis & Recommendations</div>',
                unsafe_allow_html=True)

    st.markdown(f"""
    <div class="ai-card">
        <div class="ai-card-header">
            <span style="font-size:20px">🤖</span>
            <span style="font-weight:600; color:#111827">AI Assessment</span>
            <span class="ai-source-badge {source_cls}">{source_label}</span>
        </div>
        <div class="ai-response-text">{ai_response}</div>
    </div>
    """, unsafe_allow_html=True)


def render_similar_cases(cases: list[dict]):
    """
    Display top 3 similar historical heart patients.

    Parameters
    ----------
    cases : list[dict]
        Each dict should have keys:
        rank, similarity_pct, diagnosis, age, gender,
        treatment, outcome, bp, cholesterol
        (returned by core/similarity.py)
    """
    if not cases:
        st.info("No similar cases found in the dataset.")
        return

    st.markdown('<div class="section-title">📂 Top Similar Historical Cases</div>',
                unsafe_allow_html=True)

    rank_labels = ["#1 Closest Match", "#2 Similar Case", "#3 Similar Case"]

    for i, case in enumerate(cases[:3]):
        rank_label = rank_labels[i] if i < len(rank_labels) else f"#{i+1}"
        outcome_cls = _outcome_class(case.get("outcome", "monitoring"))

        st.markdown(f"""
        <div class="case-card">
            <div class="case-rank">{rank_label}</div>
            <div class="case-similarity">{case.get('similarity_pct', '--')}%</div>
            <div class="case-diagnosis">{case.get('diagnosis', 'Unknown diagnosis')}</div>
            <div class="case-meta">
                <span>👤 {case.get('age', '?')}y {case.get('gender', '')}</span>
                <span>💉 BP {case.get('bp', '?')}</span>
                <span>🧪 Chol. {case.get('cholesterol', '?')}</span>
            </div>
            <div class="case-treatment">
                💊 <strong>Treatment:</strong> {case.get('treatment', 'Not recorded')}
            </div>
            <div>
                <span class="case-outcome {outcome_cls}">
                    ✦ {case.get('outcome', 'Monitoring')}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_online_extras(guidelines: list[str] = None,
                          drug_info: dict = None):
    """
    Online-only section: cardiology guidelines + drug interaction info.
    Only called from app.py when is_online is True.

    Parameters
    ----------
    guidelines : list[str]  Fetched from services/api_service.py
    drug_info  : dict       Drug interaction result from api_service.py
    """
    st.markdown('<div class="section-title">🌐 Live Cardiology Intelligence</div>',
                unsafe_allow_html=True)

    # Guidelines panel
    if guidelines:
        lines_html = "".join(
            f'<div class="guideline-item">📋 {g}</div>' for g in guidelines
        )
        st.markdown(f"""
        <div class="online-extra-card">
            <div class="online-extra-title">📖 Latest Cardiology Guidelines</div>
            {lines_html}
        </div>
        """, unsafe_allow_html=True)

    # Drug interaction panel
    if drug_info:
        drugs_html = "".join(
            f'<span class="drug-tag">{d}</span>'
            for d in drug_info.get("drugs", [])
        )
        interaction_note = drug_info.get("interaction", "No major interactions detected.")
        st.markdown(f"""
        <div class="online-extra-card">
            <div class="online-extra-title">💊 Cardiac Drug Interaction Check</div>
            <div>{drugs_html}</div>
            <div class="drug-interaction">⚠️ {interaction_note}</div>
        </div>
        """, unsafe_allow_html=True)


def render_results_header(patient_name: str, is_online: bool):
    """Top blue banner shown when results are available."""
    mode_label = "Online · Advanced AI" if is_online else "Offline · Local AI"
    st.markdown(f"""
    <div class="page-header">
        <h1>Analysis: {patient_name}</h1>
        <p>Cardiac decision support report generated by AuraCure AI</p>
        <span class="header-badge">{mode_label}</span>
    </div>
    """, unsafe_allow_html=True)


def render_loading():
    """Progress bar shown while AI + ML pipeline runs."""
    st.markdown("""
    <div class="analysis-loading">
        <div style="font-size:36px; margin-bottom:12px">🔬</div>
        <div style="font-size:15px; font-weight:600; color:#3B5BDB">
            Running cardiac analysis…
        </div>
        <div style="font-size:12px; color:#6B7AB8; margin-top:6px">
            Similarity search · Risk scoring · AI reasoning
        </div>
    </div>
    """, unsafe_allow_html=True)
    progress = st.progress(0)
    for pct in range(0, 101, 5):
        time.sleep(0.04)
        progress.progress(pct)
    progress.empty()
