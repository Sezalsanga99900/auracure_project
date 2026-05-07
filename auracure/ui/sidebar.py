"""
ui/sidebar.py
─────────────────────────────────────────────────────────────────
AuraCure — Patient Input Sidebar
Purpose : Renders the left-side panel where the doctor fills in
          all patient details. Also shows the Online / Offline
          mode badge at the top so the doctor always knows which
          mode the system is operating in.

Dependencies used here
  • streamlit          — draws every widget (sliders, inputs, etc.)
  • utils/validators   — checks inputs before submission
  • core/mode_detector — tells us if internet is available
─────────────────────────────────────────────────────────────────
"""

import streamlit as st


# ── Inline CSS injected once ──────────────────────────────────────────────────

SIDEBAR_CSS = """
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

/* ── Global sidebar reset ── */
section[data-testid="stSidebar"] {
    background: #F0F4FF !important;
    border-right: 1.5px solid #D6E0FF;
}
section[data-testid="stSidebar"] * {
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Mode badge ── */
.mode-badge {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 16px;
    border-radius: 10px;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.04em;
    margin-bottom: 20px;
}
.mode-badge.online  { background:#E6F9F0; color:#0A7A46; border:1.5px solid #A8EDCB; }
.mode-badge.offline { background:#FFF2F2; color:#B91C1C; border:1.5px solid #FECACA; }
.mode-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
    animation: pulse 2s ease-in-out infinite;
}
.mode-badge.online  .mode-dot { background:#22C55E; box-shadow:0 0 0 3px #BBF7D0; }
.mode-badge.offline .mode-dot { background:#EF4444; box-shadow:0 0 0 3px #FECACA; animation:none; }
@keyframes pulse {
    0%,100% { box-shadow:0 0 0 3px #BBF7D0; }
    50%      { box-shadow:0 0 0 6px #DCFCE7; }
}

/* ── Section headers ── */
.form-section-title {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    color: #6B7AB8;
    margin: 18px 0 8px 0;
    padding-bottom: 6px;
    border-bottom: 1px solid #D6E0FF;
}

/* ── Logo area ── */
.sidebar-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 22px;
    padding-bottom: 16px;
    border-bottom: 1.5px solid #D6E0FF;
}
.logo-icon {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, #3B5BDB, #228BE6);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
}
.logo-text { font-size: 18px; font-weight: 700; color: #1E3A8A; letter-spacing:-0.02em; }
.logo-sub  { font-size: 10px; color: #6B7AB8; font-weight:500; letter-spacing:0.05em; }

/* ── Submit button ── */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #3B5BDB 0%, #228BE6 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    width: 100% !important;
    margin-top: 10px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(59,91,219,0.35) !important;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(59,91,219,0.45) !important;
}

/* ── Streamlit widget tweaks ── */
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label,
div[data-testid="stTextInput"] label,
div[data-testid="stNumberInput"] label,
div[data-testid="stMultiSelect"] label {
    font-size: 12px !important;
    font-weight: 600 !important;
    color: #374151 !important;
    letter-spacing: 0.01em !important;
}
div[data-testid="stSelectbox"] > div > div,
div[data-testid="stTextInput"] > div > div > input {
    border-radius: 8px !important;
    border: 1.5px solid #C7D2FE !important;
    background: #FFFFFF !important;
    font-size: 13px !important;
}
div[data-testid="stSelectbox"] > div > div:focus-within,
div[data-testid="stTextInput"] > div > div:focus-within {
    border-color: #3B5BDB !important;
    box-shadow: 0 0 0 3px rgba(59,91,219,0.12) !important;
}

/* ── Disclaimer box ── */
.disclaimer-box {
    background: #FFF7ED;
    border: 1px solid #FED7AA;
    border-left: 3px solid #F97316;
    border-radius: 8px;
    padding: 10px 12px;
    font-size: 11px;
    color: #9A3412;
    line-height: 1.5;
    margin-top: 16px;
}
</style>
"""


# ── Mode badge HTML builder ───────────────────────────────────────────────────

def _build_mode_badge(is_online: bool) -> str:
    """Return the HTML string for the online / offline badge."""
    if is_online:
        return """
        <div class="mode-badge online">
            <div class="mode-dot"></div>
            <span>🟢 Online Mode — Advanced AI Active</span>
        </div>"""
    return """
    <div class="mode-badge offline">
        <div class="mode-dot"></div>
        <span>🔴 Offline Mode — Local AI Active</span>
    </div>"""


# ── Symptom options ───────────────────────────────────────────────────────────

SYMPTOM_OPTIONS = [
    "Chest pain or tightness",
    "Shortness of breath",
    "Palpitations / irregular heartbeat",
    "Dizziness or lightheadedness",
    "Fatigue or extreme tiredness",
    "Swelling in legs or ankles",
    "Nausea or cold sweats",
    "Pain radiating to arm or jaw",
    "Fainting (syncope)",
    "No symptoms (routine check)",
]

GENDER_OPTIONS   = ["Male", "Female", "Other"]
SMOKING_OPTIONS  = ["Never", "Former smoker", "Current smoker"]
DIABETES_OPTIONS = ["No", "Type 1", "Type 2", "Pre-diabetic"]


# ── Main render function ──────────────────────────────────────────────────────

def render_sidebar(is_online: bool) -> dict | None:
    """
    Renders the complete sidebar UI inside st.sidebar.

    Parameters
    ----------
    is_online : bool
        Result from core/mode_detector.py. Controls which badge
        is displayed and which features are enabled.

    Returns
    -------
    dict | None
        A dictionary of validated patient data when the doctor
        clicks Submit. Returns None if not yet submitted or if
        validation fails.
    """

    # Inject styles once
    st.sidebar.markdown(SIDEBAR_CSS, unsafe_allow_html=True)

    # Logo
    st.sidebar.markdown("""
    <div class="sidebar-logo">
        <div class="logo-icon">🫀</div>
        <div>
            <div class="logo-text">AuraCure</div>
            <div class="logo-sub">CARDIAC DECISION SUPPORT</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Online / Offline badge
    st.sidebar.markdown(_build_mode_badge(is_online), unsafe_allow_html=True)

    # ── Section 1: Patient Identity ──────────────────────────────────────────
    st.sidebar.markdown('<div class="form-section-title">👤 Patient Identity</div>',
                        unsafe_allow_html=True)

    col1, col2 = st.sidebar.columns(2)
    with col1:
        age = st.number_input(
            "Age (years)", min_value=1, max_value=120,
            value=50, step=1, key="age"
        )
    with col2:
        gender = st.selectbox("Gender", GENDER_OPTIONS, key="gender")

    patient_name = st.sidebar.text_input(
        "Patient ID / Name (optional)", placeholder="e.g. P-00142",
        key="patient_name"
    )

    # ── Section 2: Symptoms ──────────────────────────────────────────────────
    st.sidebar.markdown('<div class="form-section-title">🩺 Presenting Symptoms</div>',
                        unsafe_allow_html=True)

    symptoms = st.sidebar.multiselect(
        "Select all symptoms present",
        options=SYMPTOM_OPTIONS,
        default=[],
        key="symptoms"
    )

    symptom_duration = st.sidebar.selectbox(
        "Duration of symptoms",
        ["< 24 hours", "1–7 days", "1–4 weeks", "> 1 month", "Not applicable"],
        key="symptom_duration"
    )

    # ── Section 3: Vitals ────────────────────────────────────────────────────
    st.sidebar.markdown('<div class="form-section-title">📊 Vitals</div>',
                        unsafe_allow_html=True)

    col3, col4 = st.sidebar.columns(2)
    with col3:
        bp_systolic = st.number_input(
            "BP Systolic (mmHg)", min_value=60, max_value=250,
            value=120, step=1, key="bp_sys"
        )
    with col4:
        bp_diastolic = st.number_input(
            "BP Diastolic (mmHg)", min_value=40, max_value=150,
            value=80, step=1, key="bp_dia"
        )

    heart_rate = st.sidebar.slider(
        "Heart Rate (bpm)", min_value=30, max_value=200,
        value=75, step=1, key="heart_rate"
    )

    col5, col6 = st.sidebar.columns(2)
    with col5:
        cholesterol = st.number_input(
            "Cholesterol (mg/dL)", min_value=50, max_value=600,
            value=200, step=1, key="cholesterol"
        )
    with col6:
        glucose = st.number_input(
            "Glucose (mg/dL)", min_value=50, max_value=600,
            value=100, step=1, key="glucose"
        )

    # ── Section 4: Risk Factors ──────────────────────────────────────────────
    st.sidebar.markdown('<div class="form-section-title">⚠️ Risk Factors</div>',
                        unsafe_allow_html=True)

    smoking   = st.sidebar.selectbox("Smoking status", SMOKING_OPTIONS, key="smoking")
    diabetes  = st.sidebar.selectbox("Diabetes", DIABETES_OPTIONS, key="diabetes")

    col7, col8 = st.sidebar.columns(2)
    with col7:
        family_history = st.checkbox("Family history\nof heart disease", key="family_history")
    with col8:
        hypertension = st.checkbox("Known\nhypertension", key="hypertension")

    # ── Submit button ────────────────────────────────────────────────────────
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    submitted = st.sidebar.button("🔍 Analyse Patient", use_container_width=True)

    # ── Medical disclaimer ───────────────────────────────────────────────────
    st.sidebar.markdown("""
    <div class="disclaimer-box">
        ⚠️ <strong>Clinical Decision Support Only.</strong><br>
        AuraCure assists — it does not replace — qualified medical
        professionals. All output must be reviewed by a licensed physician.
    </div>
    """, unsafe_allow_html=True)

    # ── Return payload only on submission ────────────────────────────────────
    if not submitted:
        return None

    # Basic inline validation before passing to validators.py
    if not symptoms:
        st.sidebar.error("⚠️ Please select at least one symptom.")
        return None

    if bp_systolic <= bp_diastolic:
        st.sidebar.error("⚠️ Systolic BP must be greater than Diastolic BP.")
        return None

    return {
        "patient_name"      : patient_name or f"Patient-{age}{gender[0]}",
        "age"               : age,
        "gender"            : gender,
        "symptoms"          : symptoms,
        "symptom_duration"  : symptom_duration,
        "bp_systolic"       : bp_systolic,
        "bp_diastolic"      : bp_diastolic,
        "heart_rate"        : heart_rate,
        "cholesterol"       : cholesterol,
        "glucose"           : glucose,
        "smoking"           : smoking,
        "diabetes"          : diabetes,
        "family_history"    : family_history,
        "hypertension"      : hypertension,
        "is_online"         : is_online,
    }
