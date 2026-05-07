"""
services/api_service.py
───────────────────────
Clinical reference API layer for AuraEcho+.

Responsibility:
    Provide structured access to cardiology clinical guidelines,
    drug interaction checking, ICD-10 code lookup, and lab reference
    ranges. Acts as a mock/local implementation of what would be a
    real clinical API (like ClinicalTrials.gov or a hospital formulary).

Why a mock layer?
    In a hackathon/MVP context, we cannot integrate live hospital
    APIs (HL7 FHIR, Epic, Cerner). This module provides:
        1. Realistic clinical content (ACC/AHA guideline summaries)
        2. The same interface a real API would expose
        3. Easy future swap — replace mock data with real HTTP calls
           without changing any calling code

What's in here:
    • get_guidelines(risk_level)      — ACC/AHA treatment guidelines
    • get_drug_interactions(drugs)    — common cardiac drug interactions
    • get_icd10_codes(risk_level)     — relevant ICD-10 billing codes
    • get_lab_reference_ranges()      — normal ranges for cardiac labs
    • get_lifestyle_recommendations() — evidence-based lifestyle advice
    • get_emergency_criteria()        — when to call 911 / emergency referral

Public API:
    get_guidelines(risk_level)              → GuidelineResult
    check_drug_interactions(drug_list)      → List[InteractionAlert]
    get_icd10(risk_level, symptoms)         → List[ICD10Code]
    get_lab_ranges()                        → Dict[str, LabRange]
    get_lifestyle_advice(risk_level)        → List[str]
    get_emergency_criteria()                → List[str]
    get_full_clinical_brief(patient, risk)  → ClinicalBrief
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from utils.constants import RISK_LOW, RISK_MEDIUM, RISK_HIGH
from utils.helpers import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────

@dataclass
class GuidelineResult:
    """
    Structured ACC/AHA guideline output for a given risk level.

    Attributes
    ----------
    risk_level        : str
    guideline_class   : str  "Class I" | "Class IIa" | "Class IIb" | "Class III"
    evidence_level    : str  "Level A" | "Level B" | "Level C"
    recommendations   : List[str]  — ordered list of clinical actions
    medications       : List[str]  — first-line medications
    monitoring        : List[str]  — what to track and how often
    referral          : str        — when/who to refer to
    source            : str        — guideline document name + year
    """
    risk_level:      str
    guideline_class: str
    evidence_level:  str
    recommendations: List[str]    = field(default_factory=list)
    medications:     List[str]    = field(default_factory=list)
    monitoring:      List[str]    = field(default_factory=list)
    referral:        str          = ""
    source:          str          = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_level":      self.risk_level,
            "guideline_class": self.guideline_class,
            "evidence_level":  self.evidence_level,
            "recommendations": self.recommendations,
            "medications":     self.medications,
            "monitoring":      self.monitoring,
            "referral":        self.referral,
            "source":          self.source,
        }


@dataclass
class InteractionAlert:
    """
    A drug-drug interaction warning.

    Attributes
    ----------
    drug_a      : str
    drug_b      : str
    severity    : str  "Major" | "Moderate" | "Minor"
    description : str  — what the interaction causes
    action      : str  — what the clinician should do
    """
    drug_a:      str
    drug_b:      str
    severity:    str
    description: str
    action:      str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "drug_a":      self.drug_a,
            "drug_b":      self.drug_b,
            "severity":    self.severity,
            "description": self.description,
            "action":      self.action,
        }


@dataclass
class ICD10Code:
    """An ICD-10-CM diagnostic code."""
    code:        str
    description: str
    category:    str

    def to_dict(self) -> Dict[str, Any]:
        return {"code": self.code, "description": self.description, "category": self.category}


@dataclass
class LabRange:
    """Normal reference range for a lab value."""
    name:    str
    low:     float
    high:    float
    unit:    str
    notes:   str = ""

    def is_normal(self, value: float) -> bool:
        return self.low <= value <= self.high

    def flag(self, value: float) -> str:
        if value < self.low:   return "LOW ↓"
        if value > self.high:  return "HIGH ↑"
        return "Normal"


@dataclass
class ClinicalBrief:
    """
    Complete clinical reference package for one patient.
    Combines guidelines + drugs + ICD-10 + labs + lifestyle.
    """
    risk_level:     str
    guidelines:     GuidelineResult
    interactions:   List[InteractionAlert]  = field(default_factory=list)
    icd10_codes:    List[ICD10Code]         = field(default_factory=list)
    lab_flags:      Dict[str, str]          = field(default_factory=dict)
    lifestyle:      List[str]               = field(default_factory=list)
    emergency_flags: List[str]              = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_level":      self.risk_level,
            "guidelines":      self.guidelines.to_dict(),
            "interactions":    [i.to_dict() for i in self.interactions],
            "icd10_codes":     [c.to_dict() for c in self.icd10_codes],
            "lab_flags":       self.lab_flags,
            "lifestyle":       self.lifestyle,
            "emergency_flags": self.emergency_flags,
        }


# ─────────────────────────────────────────────
# Static clinical data (mock / embedded)
# ─────────────────────────────────────────────

# ACC/AHA Heart Failure and Coronary Artery Disease Guidelines 2023
_GUIDELINES: Dict[str, Dict[str, Any]] = {
    RISK_LOW: {
        "guideline_class": "Class I",
        "evidence_level":  "Level B",
        "source":          "ACC/AHA Guideline on Primary Prevention of CVD (2023)",
        "recommendations": [
            "Lifestyle modification as first-line intervention",
            "Annual cardiovascular risk assessment",
            "Blood pressure monitoring every 6 months",
            "Fasting lipid panel annually",
            "Diabetes screening every 3 years if risk factors present",
            "Encourage physical activity: ≥150 min/week moderate intensity",
            "Dietary counselling: Mediterranean or DASH diet",
            "Smoking cessation counselling if applicable",
        ],
        "medications": [
            "Aspirin 81mg/day — only if 10-year ASCVD risk ≥10% after shared decision-making",
            "Statin therapy if LDL-C ≥190 mg/dL or 10-year risk ≥7.5%",
            "ACE inhibitor — consider if hypertension present",
        ],
        "monitoring": [
            "Blood pressure: every 6 months",
            "Lipid panel: annually",
            "HbA1c: annually if pre-diabetic",
            "Weight/BMI: every visit",
            "12-lead ECG: baseline, then as clinically indicated",
        ],
        "referral": "Primary care follow-up in 3-6 months. Cardiology only if symptoms develop.",
    },
    RISK_MEDIUM: {
        "guideline_class": "Class IIa",
        "evidence_level":  "Level B",
        "source":          "ACC/AHA CAD Management Guideline (2023)",
        "recommendations": [
            "Initiate or intensify statin therapy (moderate-to-high intensity)",
            "Blood pressure control target: <130/80 mmHg",
            "Consider stress testing (exercise treadmill or pharmacologic)",
            "Order coronary artery calcium (CAC) score if decision uncertain",
            "Antiplatelet therapy discussion with physician",
            "Cardiac rehabilitation referral if CAD confirmed",
            "Optimise diabetes management if HbA1c >7%",
            "Sleep apnoea screening (OSA worsens cardiac outcomes)",
        ],
        "medications": [
            "High-intensity statin: Atorvastatin 40-80mg or Rosuvastatin 20-40mg",
            "ACE inhibitor or ARB: if hypertension or reduced EF",
            "Beta-blocker: if prior MI, HF, or angina symptoms",
            "Aspirin 81mg/day: if established ASCVD",
            "Consider SGLT-2 inhibitor if T2DM co-existing",
        ],
        "monitoring": [
            "Blood pressure: every 1-3 months until controlled",
            "Lipid panel: 4-6 weeks after statin initiation, then annually",
            "HbA1c: every 3-6 months if diabetic",
            "Renal function + electrolytes: annually (ACE/ARB use)",
            "12-lead ECG: every 6-12 months",
            "Echocardiogram: baseline EF assessment",
        ],
        "referral": "Cardiology within 2-4 weeks. Consider stress echo or nuclear perfusion study.",
    },
    RISK_HIGH: {
        "guideline_class": "Class I",
        "evidence_level":  "Level A",
        "source":          "ACC/AHA ACS Management Guideline (2023) + ESC Guidelines",
        "recommendations": [
            "🚨 URGENT: Immediate cardiology evaluation",
            "Rule out acute coronary syndrome (ACS) — obtain serial troponins",
            "12-lead ECG immediately and at 3-6 hours",
            "Admit for monitoring if ACS cannot be excluded",
            "Dual antiplatelet therapy (DAPT) if ACS confirmed",
            "Coronary angiography — consider within 24-72 hours",
            "Echocardiogram to assess LV function",
            "High-intensity statin therapy regardless of baseline LDL",
            "IV anticoagulation if STEMI or high-risk NSTEMI",
            "Revascularisation (PCI or CABG) based on anatomy",
        ],
        "medications": [
            "Aspirin 325mg loading dose, then 81mg/day",
            "P2Y12 inhibitor: Ticagrelor 180mg loading or Clopidogrel 600mg loading",
            "High-intensity statin: Atorvastatin 80mg STAT",
            "Beta-blocker: Metoprolol succinate — if haemodynamically stable",
            "ACE inhibitor: Lisinopril — if EF <40% or anterior MI",
            "Anticoagulation: Heparin IV or LMWH per ACS protocol",
            "Nitroglycerin: for ongoing chest pain (avoid if hypotensive)",
            "Consider ezetimibe + PCSK9 inhibitor if LDL >70 mg/dL on max statin",
        ],
        "monitoring": [
            "Continuous cardiac monitoring (telemetry)",
            "Troponin every 3-6 hours × 3 sets",
            "BMP, CBC, coagulation studies STAT",
            "Blood pressure every 15-30 minutes (acute phase)",
            "Urine output monitoring",
            "Serial ECGs every 6-8 hours",
            "Daily echocardiogram if EF impaired",
        ],
        "referral": "🚨 EMERGENCY: Immediate cardiology / interventional cardiology. Consider cath lab activation.",
    },
}

# Common cardiac drug-drug interactions
_DRUG_INTERACTIONS: List[Dict[str, Any]] = [
    {
        "drugs":       ("warfarin", "aspirin"),
        "severity":    "Major",
        "description": "Increased bleeding risk — additive anticoagulant effect",
        "action":      "Monitor INR closely; use lowest effective aspirin dose; consider PPI prophylaxis",
    },
    {
        "drugs":       ("simvastatin", "amlodipine"),
        "severity":    "Moderate",
        "description": "Amlodipine inhibits CYP3A4, increasing simvastatin exposure — myopathy risk",
        "action":      "Limit simvastatin to 20mg/day; prefer rosuvastatin or pravastatin",
    },
    {
        "drugs":       ("metoprolol", "verapamil"),
        "severity":    "Major",
        "description": "Additive negative chronotropic and dromotropic effects — risk of AV block / bradycardia",
        "action":      "Avoid combination; if necessary, start at very low doses with continuous monitoring",
    },
    {
        "drugs":       ("lisinopril", "spironolactone"),
        "severity":    "Major",
        "description": "Both increase serum potassium — risk of life-threatening hyperkalaemia",
        "action":      "Monitor electrolytes weekly initially; consider dose reduction; target K+ <5.5",
    },
    {
        "drugs":       ("clopidogrel", "omeprazole"),
        "severity":    "Moderate",
        "description": "Omeprazole inhibits CYP2C19, reducing clopidogrel activation by up to 47%",
        "action":      "Switch to pantoprazole or famotidine for GI protection",
    },
    {
        "drugs":       ("atorvastatin", "clarithromycin"),
        "severity":    "Major",
        "description": "CYP3A4 inhibition dramatically increases atorvastatin levels — severe myopathy / rhabdomyolysis",
        "action":      "Withhold atorvastatin during clarithromycin course; resume after antibiotic completed",
    },
    {
        "drugs":       ("digoxin", "amiodarone"),
        "severity":    "Major",
        "description": "Amiodarone inhibits P-gp and CYP3A4, increasing digoxin levels 70-100%",
        "action":      "Reduce digoxin dose by 50% when starting amiodarone; monitor levels closely",
    },
    {
        "drugs":       ("heparin", "nsaids"),
        "severity":    "Moderate",
        "description": "NSAIDs inhibit platelet function and may increase GI bleeding risk with heparin",
        "action":      "Avoid NSAIDs; use acetaminophen for pain; add PPI if combination unavoidable",
    },
]

# ICD-10-CM codes for cardiac diagnoses
_ICD10_CODES: Dict[str, List[Dict[str, str]]] = {
    RISK_LOW: [
        {"code": "Z13.6",   "description": "Encounter for screening for cardiovascular disorders", "category": "Screening"},
        {"code": "Z82.49",  "description": "Family history of ischaemic heart disease and other diseases", "category": "Family History"},
        {"code": "I10",     "description": "Essential (primary) hypertension", "category": "Hypertension"},
    ],
    RISK_MEDIUM: [
        {"code": "I25.10",  "description": "Atherosclerotic heart disease of native coronary artery", "category": "CAD"},
        {"code": "I25.2",   "description": "Old myocardial infarction", "category": "CAD"},
        {"code": "I20.0",   "description": "Unstable angina", "category": "Angina"},
        {"code": "I20.9",   "description": "Angina pectoris, unspecified", "category": "Angina"},
        {"code": "R07.9",   "description": "Chest pain, unspecified", "category": "Symptom"},
        {"code": "I10",     "description": "Essential (primary) hypertension", "category": "Hypertension"},
    ],
    RISK_HIGH: [
        {"code": "I21.9",   "description": "Acute myocardial infarction, unspecified", "category": "ACS"},
        {"code": "I21.3",   "description": "ST elevation (STEMI) myocardial infarction", "category": "ACS"},
        {"code": "I21.4",   "description": "Non-ST elevation (NSTEMI) myocardial infarction", "category": "ACS"},
        {"code": "I20.0",   "description": "Unstable angina", "category": "Angina"},
        {"code": "I50.9",   "description": "Heart failure, unspecified", "category": "Heart Failure"},
        {"code": "I47.2",   "description": "Ventricular tachycardia", "category": "Arrhythmia"},
        {"code": "I46.9",   "description": "Cardiac arrest, cause unspecified", "category": "Emergency"},
    ],
}

# Lab reference ranges for cardiac workup
_LAB_RANGES: Dict[str, Dict[str, Any]] = {
    "Total Cholesterol":    {"low": 0,   "high": 200,  "unit": "mg/dL",  "notes": ">240 = High"},
    "LDL Cholesterol":      {"low": 0,   "high": 100,  "unit": "mg/dL",  "notes": "<70 target for high-risk"},
    "HDL Cholesterol (M)":  {"low": 40,  "high": 200,  "unit": "mg/dL",  "notes": "<40 = Low (Men)"},
    "HDL Cholesterol (F)":  {"low": 50,  "high": 200,  "unit": "mg/dL",  "notes": "<50 = Low (Women)"},
    "Triglycerides":        {"low": 0,   "high": 150,  "unit": "mg/dL",  "notes": ">500 = Pancreatitis risk"},
    "Troponin I (hs)":      {"low": 0,   "high": 0.04, "unit": "ng/mL",  "notes": ">0.04 = Myocardial injury"},
    "BNP":                  {"low": 0,   "high": 100,  "unit": "pg/mL",  "notes": ">400 = Likely HF"},
    "NT-proBNP":            {"low": 0,   "high": 125,  "unit": "pg/mL",  "notes": "Age-adjusted thresholds"},
    "HbA1c":                {"low": 0,   "high": 5.6,  "unit": "%",      "notes": "5.7-6.4% = Pre-diabetic"},
    "Fasting Glucose":      {"low": 70,  "high": 100,  "unit": "mg/dL",  "notes": "100-125 = Pre-diabetic"},
    "Creatinine":           {"low": 0.7, "high": 1.2,  "unit": "mg/dL",  "notes": "Adjust for sex/age"},
    "eGFR":                 {"low": 60,  "high": 120,  "unit": "mL/min", "notes": "<60 = CKD stage 3"},
    "Potassium":            {"low": 3.5, "high": 5.0,  "unit": "mEq/L",  "notes": "Critical: <3.0 or >6.0"},
    "Sodium":               {"low": 136, "high": 145,  "unit": "mEq/L",  "notes": "<130 worsens HF prognosis"},
    "INR (on warfarin)":    {"low": 2.0, "high": 3.0,  "unit": "ratio",  "notes": "Therapeutic range for AF/DVT"},
}

# Lifestyle recommendations by risk level
_LIFESTYLE: Dict[str, List[str]] = {
    RISK_LOW: [
        "150+ minutes/week of moderate aerobic exercise (brisk walking, swimming, cycling)",
        "DASH or Mediterranean diet: ↑ fruits, vegetables, whole grains; ↓ sodium, saturated fats",
        "Maintain BMI 18.5–24.9 kg/m²; waist circumference <40 in (M) or <35 in (F)",
        "Limit alcohol: ≤1 drink/day (women), ≤2 drinks/day (men)",
        "Complete smoking cessation — even 1 cigarette/day doubles cardiovascular risk",
        "Stress management: mindfulness, yoga, adequate sleep (7-9 hours/night)",
        "Monitor blood pressure at home monthly",
    ],
    RISK_MEDIUM: [
        "Cardiac rehabilitation programme — supervised exercise therapy (evidence Level A)",
        "Sodium restriction: <2,300 mg/day; target <1,500 mg/day if hypertensive",
        "Saturated fat <7% of total calories; eliminate trans fats completely",
        "Maintain LDL-C <70 mg/dL through diet + medication combination",
        "Daily weight monitoring — >2 lbs gain in 24hrs or >5 lbs in 1 week: call cardiologist",
        "Blood pressure home monitoring daily; log readings for physician review",
        "Limit caffeine; avoid energy drinks",
        "Sexual activity guidance — discuss with cardiologist based on exercise tolerance",
    ],
    RISK_HIGH: [
        "🚨 MEDICAL CLEARANCE REQUIRED before any exercise programme",
        "Complete bed rest during acute phase — no exertion until cardiologist clears",
        "Strict sodium restriction: <1,500 mg/day (fluid restriction if indicated)",
        "Daily weight: alert care team if >2 lbs/day increase (fluid retention)",
        "Zero alcohol — alcohol worsens cardiomyopathy and arrhythmias",
        "Zero smoking — immediate cessation; Varenicline or NRT with physician guidance",
        "Supervised cardiac rehabilitation (Phase II) — mandatory after discharge",
        "Driving restriction — discuss with cardiologist (typically 1-4 weeks post-event)",
        "Psychological support — depression affects 20-30% of post-MI patients",
        "Medical alert bracelet recommended for severe cases",
    ],
}

# Emergency referral criteria
_EMERGENCY_CRITERIA = [
    "🚨 Chest pain/pressure radiating to arm, jaw, back, or neck",
    "🚨 Sudden severe shortness of breath at rest",
    "🚨 Syncope (loss of consciousness) or near-syncope",
    "🚨 Palpitations with haemodynamic instability (hypotension, diaphoresis)",
    "🚨 Heart rate >150 bpm or <40 bpm with symptoms",
    "🚨 Blood pressure >180/120 mmHg (hypertensive emergency) or <90/60 mmHg (shock)",
    "🚨 ST elevation >1mm in ≥2 contiguous leads on ECG",
    "🚨 New left bundle branch block with chest pain",
    "🚨 Troponin rising on serial measurements",
    "🚨 Acute pulmonary oedema (frothy sputum, severe orthopnoea)",
    "🚨 Sudden neurological deficit (stroke can mimic/accompany cardiac events)",
    "🚨 Signs of cardiac tamponade: hypotension + JVD + muffled heart sounds",
]


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def get_guidelines(risk_level: str) -> GuidelineResult:
    """
    Return ACC/AHA treatment guidelines for a given risk level.

    Parameters
    ----------
    risk_level : str  "Low" | "Medium" | "High"

    Returns
    -------
    GuidelineResult with recommendations, medications, monitoring, referral.
    """
    data = _GUIDELINES.get(risk_level, _GUIDELINES[RISK_MEDIUM])
    result = GuidelineResult(
        risk_level=risk_level,
        guideline_class=data["guideline_class"],
        evidence_level=data["evidence_level"],
        recommendations=data["recommendations"],
        medications=data["medications"],
        monitoring=data["monitoring"],
        referral=data["referral"],
        source=data["source"],
    )
    logger.debug("Guidelines fetched for risk_level=%s", risk_level)
    return result


def check_drug_interactions(drug_list: List[str]) -> List[InteractionAlert]:
    """
    Check for known drug-drug interactions among a list of medications.

    Parameters
    ----------
    drug_list : List[str]  — medication names (case-insensitive)

    Returns
    -------
    List[InteractionAlert] — empty list if no interactions found.

    Example
    -------
    alerts = check_drug_interactions(["warfarin", "aspirin", "lisinopril"])
    """
    if not drug_list or len(drug_list) < 2:
        return []

    # Normalise to lowercase for matching
    normalised = [d.lower().strip() for d in drug_list]
    alerts: List[InteractionAlert] = []

    for interaction in _DRUG_INTERACTIONS:
        drug_a, drug_b = interaction["drugs"]
        # Check if both drugs in the interaction are in the patient's list
        if drug_a in normalised and drug_b in normalised:
            alerts.append(InteractionAlert(
                drug_a=drug_a.title(),
                drug_b=drug_b.title(),
                severity=interaction["severity"],
                description=interaction["description"],
                action=interaction["action"],
            ))

    # Sort by severity: Major first
    severity_order = {"Major": 0, "Moderate": 1, "Minor": 2}
    alerts.sort(key=lambda a: severity_order.get(a.severity, 99))

    logger.debug(
        "Drug interaction check: %d drugs → %d alerts", len(drug_list), len(alerts)
    )
    return alerts


def get_icd10(
    risk_level: str,
    include_all_levels: bool = False,
) -> List[ICD10Code]:
    """
    Return relevant ICD-10-CM codes for the given risk level.

    Parameters
    ----------
    risk_level          : str — "Low" | "Medium" | "High"
    include_all_levels  : bool — if True, include codes from all risk levels

    Returns
    -------
    List[ICD10Code]
    """
    if include_all_levels:
        all_codes: List[Dict] = []
        for codes in _ICD10_CODES.values():
            all_codes.extend(codes)
        code_list = all_codes
    else:
        code_list = _ICD10_CODES.get(risk_level, _ICD10_CODES[RISK_MEDIUM])

    result = [
        ICD10Code(
            code=c["code"],
            description=c["description"],
            category=c["category"],
        )
        for c in code_list
    ]
    logger.debug("ICD-10 lookup: risk=%s → %d codes", risk_level, len(result))
    return result


def get_lab_ranges() -> Dict[str, LabRange]:
    """
    Return all cardiac lab reference ranges.

    Returns
    -------
    Dict mapping lab name → LabRange object.
    """
    return {
        name: LabRange(
            name=name,
            low=vals["low"],
            high=vals["high"],
            unit=vals["unit"],
            notes=vals.get("notes", ""),
        )
        for name, vals in _LAB_RANGES.items()
    }


def flag_patient_labs(patient: Dict[str, Any]) -> Dict[str, str]:
    """
    Flag any patient lab values outside normal reference ranges.

    Checks the clinical features present in the patient dict against
    the embedded lab ranges. Returns only the flagged values.

    Parameters
    ----------
    patient : dict  — patient record with clinical feature values

    Returns
    -------
    Dict mapping feature name → flag string ("HIGH ↑", "LOW ↓", "Normal")
    """
    flags: Dict[str, str] = {}
    ranges = get_lab_ranges()

    # Map patient features to lab names where applicable
    feature_to_lab = {
        "chol":     "Total Cholesterol",
        "trestbps": None,   # Blood pressure — check separately
        "thalach":  None,   # Heart rate — check separately
    }

    # Cholesterol check
    chol = patient.get("chol")
    if chol is not None:
        try:
            chol_val = float(chol)
            lab = ranges.get("Total Cholesterol")
            if lab:
                flag = lab.flag(chol_val)
                if flag != "Normal":
                    flags["Total Cholesterol"] = f"{flag} ({chol_val} mg/dL)"
        except (ValueError, TypeError):
            pass

    # Blood pressure check (trestbps)
    trestbps = patient.get("trestbps")
    if trestbps is not None:
        try:
            bp_val = float(trestbps)
            if bp_val > 140:
                flags["Resting BP"] = f"HIGH ↑ ({bp_val} mmHg — Stage 2 Hypertension threshold: 140)"
            elif bp_val > 130:
                flags["Resting BP"] = f"ELEVATED ({bp_val} mmHg — Stage 1 Hypertension)"
        except (ValueError, TypeError):
            pass

    # Heart rate check (thalach)
    thalach = patient.get("thalach")
    if thalach is not None:
        try:
            hr_val = float(thalach)
            if hr_val > 100:
                flags["Max Heart Rate"] = f"HIGH ↑ ({hr_val} bpm — tachycardia threshold: 100)"
            elif hr_val < 60:
                flags["Max Heart Rate"] = f"LOW ↓ ({hr_val} bpm — bradycardia threshold: 60)"
        except (ValueError, TypeError):
            pass

    return flags


def get_lifestyle_advice(risk_level: str) -> List[str]:
    """
    Return evidence-based lifestyle recommendations for a risk level.

    Returns
    -------
    List[str]
    """
    return list(_LIFESTYLE.get(risk_level, _LIFESTYLE[RISK_MEDIUM]))


def get_emergency_criteria() -> List[str]:
    """
    Return the list of criteria that mandate emergency referral.

    Returns
    -------
    List[str] — each item is an actionable emergency trigger.
    """
    return list(_EMERGENCY_CRITERIA)


def check_emergency_flags(patient: Dict[str, Any]) -> List[str]:
    """
    Automatically flag emergency conditions from patient data.

    Checks hard thresholds from the patient vitals and returns
    any emergency criteria that are met.

    Parameters
    ----------
    patient : dict

    Returns
    -------
    List[str] — triggered emergency conditions (empty = no flags)
    """
    flags: List[str] = []

    try:
        # Blood pressure emergency
        trestbps = float(patient.get("trestbps", 0))
        if trestbps >= 180:
            flags.append(f"🚨 Resting BP = {trestbps} mmHg — Hypertensive Emergency threshold")

        # Severe tachycardia
        thalach = float(patient.get("thalach", 0))
        if thalach > 150:
            flags.append(f"🚨 Max Heart Rate = {thalach} bpm — Severe tachycardia")

        # Exercise-induced angina with high ST depression
        exang   = int(float(patient.get("exang", 0)))
        oldpeak = float(patient.get("oldpeak", 0))
        if exang == 1 and oldpeak > 3.0:
            flags.append(
                f"🚨 Exercise-induced angina + ST depression {oldpeak} mm — "
                "Significant ischaemic burden"
            )

        # Multiple vessel disease indicators
        ca = int(float(patient.get("ca", 0)))
        if ca >= 3:
            flags.append(f"🚨 {ca} major vessels affected — likely multivessel disease")

        # Reversible thalassemia defect (worst prognosis)
        thal = int(float(patient.get("thal", 0)))
        if thal == 2:
            flags.append("🚨 Reversible thalassemia defect — indicates active myocardial ischaemia")

    except (ValueError, TypeError) as exc:
        logger.warning("Emergency flag check error: %s", exc)

    return flags


def get_full_clinical_brief(
    patient:     Dict[str, Any],
    risk_level:  str,
    drug_list:   Optional[List[str]] = None,
) -> ClinicalBrief:
    """
    Generate a complete clinical reference package for one patient.

    Combines guidelines + drug interactions + ICD-10 + lab flags +
    lifestyle + emergency flags into a single ClinicalBrief object.

    Parameters
    ----------
    patient    : dict  — patient record
    risk_level : str   — from risk_model.py
    drug_list  : list  — current medications (optional)

    Returns
    -------
    ClinicalBrief
    """
    guidelines   = get_guidelines(risk_level)
    interactions = check_drug_interactions(drug_list or [])
    icd10_codes  = get_icd10(risk_level)
    lab_flags    = flag_patient_labs(patient)
    lifestyle    = get_lifestyle_advice(risk_level)
    emergency    = check_emergency_flags(patient)

    logger.info(
        "Clinical brief generated: risk=%s | %d interactions | %d lab flags | %d emergency flags",
        risk_level, len(interactions), len(lab_flags), len(emergency),
    )

    return ClinicalBrief(
        risk_level=risk_level,
        guidelines=guidelines,
        interactions=interactions,
        icd10_codes=icd10_codes,
        lab_flags=lab_flags,
        lifestyle=lifestyle,
        emergency_flags=emergency,
    )