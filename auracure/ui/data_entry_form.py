"""
ui/data_entry_form.py
─────────────────────────────────────────────────────────────────
AuraCure — Structured Patient Data Entry Form
─────────────────────────────────────────────────────────────────
PURPOSE:
    A dedicated full-page patient data entry form with:
    - Personal details (name, age, gender, contact)
    - Symptom checklist with severity sliders
    - Vitals input (BP, HR, cholesterol, glucose, BMI)
    - Medical history (smoking, diabetes, family history)
    - Medical report upload section (PDF/image)
    - Form progress indicator
    - Save as draft + Submit buttons

USED BY:
    app.py — rendered as a separate tab/page
    Passes validated patient dict → core/ pipeline

IMPORTS FROM:
    utils/validators.py  — validate_patient()
    utils/constants.py   — field ranges, symptom list
─────────────────────────────────────────────────────────────────
"""

import streamlit as st
import json
from datetime import date, datetime


# ─────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────

DATA_ENTRY_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&display=swap');

/* Reset */
.main .block-container { padding-top: 1.2rem !important; }
*, .stMarkdown { font-family: 'DM Sans', sans-serif !important; }

/* ── Page title ── */
.form-page-header {
    background: white;
    border: 1.5px solid #E0E7FF;
    border-left: 5px solid #3B5BDB;
    border-radius: 12px;
    padding: 20px 26px;
    margin-bottom: 28px;
    display: flex;
    align-items: center;
    gap: 16px;
}
.form-page-icon  { font-size: 36px; }
.form-page-title { font-size: 20px; font-weight: 700; color: #1E3A8A; margin:0; }
.form-page-sub   { font-size: 12px; color: #6B7AB8; margin-top:3px; }

/* ── Progress bar ── */
.form-progress-wrap {
    margin-bottom: 24px;
}
.form-progress-label {
    display: flex; justify-content: space-between;
    font-size: 12px; font-weight: 600;
    color: #6B7AB8; margin-bottom: 8px;
}
.form-progress-bar {
    height: 6px; background: #E0E7FF;
    border-radius: 99px; overflow: hidden;
}
.form-progress-fill {
    height: 100%; border-radius: 99px;
    background: linear-gradient(90deg, #3B5BDB, #60A5FA);
    transition: width 0.4s ease;
}

/* ── Section card ── */
.form-section-card {
    background: white;
    border: 1.5px solid #E5E7EB;
    border-radius: 14px;
    padding: 24px 28px;
    margin-bottom: 20px;
    box-shadow: 0 1px 8px rgba(59,91,219,0.04);
}
.form-section-header {
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 20px;
    padding-bottom: 12px;
    border-bottom: 1.5px solid #F3F4F6;
}
.form-section-icon  { font-size: 22px; }
.form-section-title {
    font-size: 14px; font-weight: 700;
    color: #1E3A8A; letter-spacing: 0.02em;
}
.form-section-num {
    margin-left: auto;
    background: #EEF2FF; color: #3B5BDB;
    font-size: 11px; font-weight: 700;
    padding: 2px 10px; border-radius: 20px;
}

/* ── Symptom chip grid ── */
.symptom-grid {
    display: flex; flex-wrap: wrap; gap: 8px;
    margin-bottom: 12px;
}
.symptom-chip {
    display: inline-flex; align-items: center; gap: 6px;
    background: #F0F4FF; border: 1.5px solid #C7D2FE;
    color: #3B5BDB; border-radius: 8px;
    font-size: 12px; font-weight: 500;
    padding: 5px 12px; cursor: pointer;
    transition: all 0.15s ease;
}
.symptom-chip.selected {
    background: #3B5BDB; color: white;
    border-color: #3B5BDB;
}

/* ── Vital input styling ── */
div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"]  label,
div[data-testid="stTextInput"]  label,
div[data-testid="stSlider"]     label,
div[data-testid="stTextArea"]   label {
    font-size: 12px !important;
    font-weight: 600 !important;
    color: #374151 !important;
}
div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"]   input {
    border-radius: 8px !important;
    border: 1.5px solid #D1D5DB !important;
    font-size: 14px !important;
    font-weight: 500 !important;
}
div[data-testid="stNumberInput"] input:focus,
div[data-testid="stTextInput"]   input:focus {
    border-color: #3B5BDB !important;
    box-shadow: 0 0 0 3px rgba(59,91,219,0.1) !important;
}

/* ── Upload zone ── */
.upload-zone {
    border: 2px dashed #C7D2FE;
    border-radius: 14px;
    padding: 32px;
    text-align: center;
    background: #F8FAFF;
    margin: 8px 0;
}
.upload-zone-icon  { font-size: 40px; margin-bottom: 8px; }
.upload-zone-title { font-size: 14px; font-weight: 600; color: #3B5BDB; }
.upload-zone-sub   { font-size: 12px; color: #9CA3AF; margin-top: 4px; }

/* ── Vital reference row ── */
.vital-ref-row {
    background: #F0F9FF;
    border: 1px solid #BAE6FD;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 11px; color: #0369A1;
    margin-top: 12px; line-height: 1.7;
}

/* ── Action buttons ── */
.form-actions {
    display: flex; gap: 12px; margin-top: 8px;
}
div[data-testid="stButton"] > button {
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
    transition: all 0.2s !important;
}
.primary-btn > div[data-testid="stButton"] > button {
    background: linear-gradient(135deg,#3B5BDB,#2563EB) !important;
    color: white !important; border: none !important;
    box-shadow: 0 4px 14px rgba(59,91,219,0.3) !important;
}

/* ── Severity slider labels ── */
.severity-label {
    display: flex; justify-content: space-between;
    font-size: 10px; color: #9CA3AF;
    margin-top: -8px; margin-bottom: 8px;
    padding: 0 4px;
}

/* ── Required asterisk ── */
.required { color: #EF4444; margin-left: 2px; }

/* ── Draft saved badge ── */
.draft-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: #F0FDF4; border: 1px solid #86EFAC;
    color: #15803D; border-radius: 20px;
    font-size: 11px; font-weight: 600;
    padding: 4px 12px;
}
</style>
"""


# ─────────────────────────────────────────────────────────────────
# Constants (mirrors utils/constants.py — kept local for UI)
# ─────────────────────────────────────────────────────────────────

SYMPTOMS_LIST = [
    "Chest pain / tightness",
    "Shortness of breath",
    "Palpitations",
    "Dizziness / lightheadedness",
    "Fatigue / weakness",
    "Leg / ankle swelling",
    "Nausea / cold sweats",
    "Arm / jaw pain",
    "Syncope (fainting)",
    "Cough (persistent)",
    "Orthopnea (breathless lying flat)",
    "No symptoms",
]

GENDER_OPTIONS    = ["Male", "Female", "Other", "Prefer not to say"]
BLOOD_GROUPS      = ["A+", "A−", "B+", "B−", "AB+", "AB−", "O+", "O−", "Unknown"]
SMOKING_OPTIONS   = ["Never smoked", "Former smoker", "Current smoker (< 10/day)",
                     "Current smoker (≥ 10/day)"]
DIABETES_OPTIONS  = ["None", "Type 1 Diabetes", "Type 2 Diabetes",
                     "Pre-diabetic / IGT", "Gestational diabetes"]
ACTIVITY_OPTIONS  = ["Sedentary", "Light (1–2 days/week)", "Moderate (3–5 days/week)",
                     "Active (6–7 days/week)"]
CHEST_PAIN_TYPES  = ["No chest pain", "Typical angina", "Atypical angina",
                     "Non-anginal pain", "Asymptomatic"]
ECG_OPTIONS       = ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy",
                     "Not done"]


# ─────────────────────────────────────────────────────────────────
# Helper: calculate progress
# ─────────────────────────────────────────────────────────────────

def _calc_progress(ss: dict) -> int:
    """Return 0–100 indicating how complete the form is."""
    fields = [
        ss.get("def_age"),
        ss.get("def_gender"),
        ss.get("def_symptoms"),
        ss.get("def_bp_sys"),
        ss.get("def_heart_rate"),
        ss.get("def_cholesterol"),
    ]
    filled = sum(1 for f in fields if f)
    return int((filled / len(fields)) * 100)


# ─────────────────────────────────────────────────────────────────
# Section renderers
# ─────────────────────────────────────────────────────────────────

def _section_personal() -> dict:
    """Section 1 — Personal details."""
    st.markdown("""
    <div class="form-section-card">
        <div class="form-section-header">
            <span class="form-section-icon">👤</span>
            <span class="form-section-title">Personal Information</span>
            <span class="form-section-num">Section 1 / 5</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            name = st.text_input(
                "Full Name / Patient ID ✱",
                placeholder="e.g. Rajveer Singh or P-00142",
                key="def_name"
            )
        with c2:
            age = st.number_input(
                "Age (years) ✱", min_value=1, max_value=120,
                value=None, placeholder="e.g. 52", key="def_age"
            )
        with c3:
            gender = st.selectbox("Gender ✱", GENDER_OPTIONS, key="def_gender")

        c4, c5, c6 = st.columns(3)
        with c4:
            blood_group = st.selectbox("Blood Group", BLOOD_GROUPS, key="def_blood")
        with c5:
            dob = st.date_input(
                "Date of Birth",
                value=None,
                min_value=date(1900, 1, 1),
                max_value=date.today(),
                key="def_dob"
            )
        with c6:
            contact = st.text_input(
                "Contact / Room No.",
                placeholder="e.g. +91-9876543210",
                key="def_contact"
            )

        notes = st.text_area(
            "Clinical Notes (optional)",
            placeholder="Add any additional context the doctor should know...",
            height=80,
            key="def_notes"
        )

    return {
        "patient_name" : name,
        "age"          : age,
        "gender"       : gender,
        "blood_group"  : blood_group,
        "dob"          : str(dob) if dob else None,
        "contact"      : contact,
        "notes"        : notes,
    }


def _section_symptoms() -> dict:
    """Section 2 — Symptom checklist with severity."""
    st.markdown("""
    <div class="form-section-card">
        <div class="form-section-header">
            <span class="form-section-icon">🩺</span>
            <span class="form-section-title">Presenting Symptoms</span>
            <span class="form-section-num">Section 2 / 5</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    symptoms = st.multiselect(
        "Select all symptoms present ✱",
        options=SYMPTOMS_LIST,
        key="def_symptoms",
        help="Select every symptom the patient reports"
    )

    chest_pain_type = st.selectbox(
        "Chest pain type (if applicable)",
        options=CHEST_PAIN_TYPES,
        key="def_cp_type"
    )

    col1, col2 = st.columns(2)
    with col1:
        symptom_duration = st.selectbox(
            "Duration of main symptom",
            ["< 24 hours", "1–3 days", "4–7 days",
             "1–4 weeks", "> 1 month", "Chronic / ongoing", "Not applicable"],
            key="def_duration"
        )
    with col2:
        symptom_severity = st.slider(
            "Overall symptom severity",
            min_value=1, max_value=10, value=5,
            key="def_severity"
        )
        st.markdown(
            '<div class="severity-label"><span>Mild (1)</span>'
            '<span>Moderate (5)</span><span>Severe (10)</span></div>',
            unsafe_allow_html=True
        )

    exercise_angina = st.checkbox(
        "Symptoms worsen with physical exertion (exercise-induced angina)",
        key="def_exang"
    )

    return {
        "symptoms"        : symptoms,
        "chest_pain_type" : chest_pain_type,
        "symptom_duration": symptom_duration,
        "symptom_severity": symptom_severity,
        "exercise_angina" : exercise_angina,
    }


def _section_vitals() -> dict:
    """Section 3 — Vitals with normal range references."""
    st.markdown("""
    <div class="form-section-card">
        <div class="form-section-header">
            <span class="form-section-icon">📊</span>
            <span class="form-section-title">Vitals & Lab Values</span>
            <span class="form-section-num">Section 3 / 5</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Blood Pressure
    c1, c2, c3 = st.columns(3)
    with c1:
        bp_sys = st.number_input(
            "Systolic BP (mmHg) ✱",
            min_value=60, max_value=250,
            value=None, placeholder="e.g. 128",
            key="def_bp_sys"
        )
    with c2:
        bp_dia = st.number_input(
            "Diastolic BP (mmHg) ✱",
            min_value=40, max_value=150,
            value=None, placeholder="e.g. 84",
            key="def_bp_dia"
        )
    with c3:
        heart_rate = st.number_input(
            "Heart Rate (bpm) ✱",
            min_value=30, max_value=220,
            value=None, placeholder="e.g. 78",
            key="def_heart_rate"
        )

    c4, c5, c6 = st.columns(3)
    with c4:
        cholesterol = st.number_input(
            "Total Cholesterol (mg/dL) ✱",
            min_value=50, max_value=700,
            value=None, placeholder="e.g. 210",
            key="def_cholesterol"
        )
    with c5:
        glucose = st.number_input(
            "Fasting Glucose (mg/dL)",
            min_value=50, max_value=600,
            value=None, placeholder="e.g. 100",
            key="def_glucose"
        )
    with c6:
        max_hr = st.number_input(
            "Max HR Achieved (exercise test)",
            min_value=50, max_value=250,
            value=None, placeholder="e.g. 150",
            key="def_thalach"
        )

    c7, c8 = st.columns(2)
    with c7:
        st_depression = st.number_input(
            "ST Depression (oldpeak)",
            min_value=0.0, max_value=10.0,
            value=0.0, step=0.1,
            key="def_oldpeak",
            help="ST depression induced by exercise relative to rest"
        )
    with c8:
        st_slope = st.selectbox(
            "ST Slope",
            ["Upsloping", "Flat", "Downsloping"],
            key="def_slope"
        )

    # Quick reference
    st.markdown("""
    <div class="vital-ref-row">
        📌 <strong>Normal ranges:</strong>
        BP 90–120 / 60–80 mmHg &nbsp;·&nbsp;
        HR 60–100 bpm &nbsp;·&nbsp;
        Total Cholesterol &lt; 200 mg/dL &nbsp;·&nbsp;
        Fasting Glucose 70–100 mg/dL
    </div>
    """, unsafe_allow_html=True)

    return {
        "bp_systolic"  : bp_sys,
        "bp_diastolic" : bp_dia,
        "heart_rate"   : heart_rate,
        "cholesterol"  : cholesterol,
        "glucose"      : glucose,
        "thalach"      : max_hr,
        "oldpeak"      : st_depression,
        "slope"        : st_slope,
    }


def _section_history() -> dict:
    """Section 4 — Medical history and risk factors."""
    st.markdown("""
    <div class="form-section-card">
        <div class="form-section-header">
            <span class="form-section-icon">📋</span>
            <span class="form-section-title">Medical History & Risk Factors</span>
            <span class="form-section-num">Section 4 / 5</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        smoking  = st.selectbox("Smoking status", SMOKING_OPTIONS, key="def_smoking")
        diabetes = st.selectbox("Diabetes status", DIABETES_OPTIONS, key="def_diabetes")
    with c2:
        activity  = st.selectbox("Physical activity", ACTIVITY_OPTIONS, key="def_activity")
        ecg       = st.selectbox("Resting ECG", ECG_OPTIONS, key="def_restecg")
    with c3:
        vessels = st.selectbox(
            "Major vessels coloured (fluoroscopy)",
            ["0", "1", "2", "3", "Not done"],
            key="def_ca"
        )
        thal = st.selectbox(
            "Thalassemia",
            ["Normal", "Fixed defect", "Reversible defect", "Not tested"],
            key="def_thal"
        )

    st.markdown("**Comorbidities** — check all that apply")
    ch1, ch2, ch3, ch4 = st.columns(4)
    with ch1:
        hypertension   = st.checkbox("Hypertension",     key="def_htn")
        prev_mi        = st.checkbox("Previous MI",       key="def_prev_mi")
    with ch2:
        heart_failure  = st.checkbox("Heart failure",     key="def_hf")
        stroke         = st.checkbox("Prior stroke/TIA",  key="def_stroke")
    with ch3:
        family_history = st.checkbox("Family hx of CAD", key="def_fam")
        obesity        = st.checkbox("Obesity (BMI>30)",  key="def_obesity")
    with ch4:
        ckd            = st.checkbox("Chronic kidney dis.",key="def_ckd")
        hyperlipidemia = st.checkbox("Hyperlipidemia",    key="def_hl")

    current_meds = st.text_area(
        "Current Medications (comma-separated)",
        placeholder="e.g. Metoprolol 50mg, Aspirin 75mg, Atorvastatin 20mg",
        height=70, key="def_meds"
    )
    allergies = st.text_input(
        "Known Drug Allergies",
        placeholder="e.g. Penicillin, Sulfa drugs",
        key="def_allergies"
    )

    return {
        "smoking"        : smoking,
        "diabetes"       : diabetes,
        "activity"       : activity,
        "ecg"            : ecg,
        "vessels"        : int(vessels) if vessels.isdigit() else 0,
        "thal"           : thal,
        "hypertension"   : hypertension,
        "prev_mi"        : prev_mi,
        "heart_failure"  : heart_failure,
        "stroke"         : stroke,
        "family_history" : family_history,
        "obesity"        : obesity,
        "ckd"            : ckd,
        "hyperlipidemia" : hyperlipidemia,
        "medications"    : current_meds,
        "allergies"      : allergies,
    }


def _section_upload() -> dict:
    """Section 5 — Medical report upload."""
    st.markdown("""
    <div class="form-section-card">
        <div class="form-section-header">
            <span class="form-section-icon">📎</span>
            <span class="form-section-title">Medical Reports & Documents</span>
            <span class="form-section-num">Section 5 / 5</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="upload-zone">
        <div class="upload-zone-icon">📄</div>
        <div class="upload-zone-title">Upload medical reports</div>
        <div class="upload-zone-sub">ECG strips, echo reports, lab reports — PDF or image</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Select files",
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="def_reports",
        label_visibility="collapsed"
    )

    report_type = st.selectbox(
        "Primary report type uploaded",
        ["None", "ECG / EKG", "Echocardiogram",
         "Stress Test", "Lab Report", "Angiogram", "Other"],
        key="def_report_type"
    )

    file_names = [f.name for f in uploaded_files] if uploaded_files else []

    if file_names:
        st.success(f"✅ {len(file_names)} file(s) attached: {', '.join(file_names)}")

    return {
        "report_files" : file_names,
        "report_type"  : report_type,
    }


# ─────────────────────────────────────────────────────────────────
# Main render
# ─────────────────────────────────────────────────────────────────

def render_data_entry_form() -> dict | None:
    """
    Render the full structured patient data entry form.

    Returns
    -------
    dict | None
        Complete patient record when submitted and validated.
        None if not yet submitted or validation fails.
    """
    st.markdown(DATA_ENTRY_CSS, unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div class="form-page-header">
        <div class="form-page-icon">📝</div>
        <div>
            <div class="form-page-title">New Patient Entry</div>
            <div class="form-page-sub">
                Fill all sections marked ✱ before submitting for AI analysis
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Progress bar (reads from session_state)
    progress = _calc_progress(st.session_state)
    st.markdown(f"""
    <div class="form-progress-wrap">
        <div class="form-progress-label">
            <span>Form completion</span>
            <span>{progress}%</span>
        </div>
        <div class="form-progress-bar">
            <div class="form-progress-fill" style="width:{progress}%"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Collect all sections ─────────────────────────────────────
    personal = _section_personal()
    symptoms = _section_symptoms()
    vitals   = _section_vitals()
    history  = _section_history()
    uploads  = _section_upload()

    # ── Action buttons ───────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    col_save, col_submit, col_clear = st.columns([1, 1, 0.5])

    with col_save:
        if st.button("💾 Save as Draft", use_container_width=True):
            st.session_state["draft_patient"] = {
                **personal, **symptoms, **vitals, **history, **uploads
            }
            st.markdown(
                '<div class="draft-badge">✓ Draft saved successfully</div>',
                unsafe_allow_html=True
            )

    with col_submit:
        submitted = st.button(
            "🔍 Submit for AI Analysis",
            use_container_width=True,
            type="primary"
        )

    with col_clear:
        if st.button("🗑 Clear", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key.startswith("def_"):
                    del st.session_state[key]
            st.rerun()

    # ── Validation on submit ─────────────────────────────────────
    if not submitted:
        return None

    errors = []
    if not personal.get("age"):
        errors.append("Age is required.")
    if not personal.get("gender"):
        errors.append("Gender is required.")
    if not symptoms.get("symptoms"):
        errors.append("At least one symptom must be selected.")
    if not vitals.get("bp_systolic"):
        errors.append("Systolic BP is required.")
    if not vitals.get("heart_rate"):
        errors.append("Heart rate is required.")
    if not vitals.get("cholesterol"):
        errors.append("Cholesterol is required.")
    if (vitals.get("bp_systolic") and vitals.get("bp_diastolic") and
            vitals["bp_systolic"] <= vitals["bp_diastolic"]):
        errors.append("Systolic BP must be greater than Diastolic BP.")

    if errors:
        for err in errors:
            st.error(f"⚠️ {err}")
        return None

    # ── Merge all sections into one record ───────────────────────
    patient_record = {
        **personal,
        **symptoms,
        **vitals,
        **history,
        **uploads,
        "submitted_at": datetime.now().isoformat(),
        "is_online"   : st.session_state.get("is_online", False),
    }

    return patient_record