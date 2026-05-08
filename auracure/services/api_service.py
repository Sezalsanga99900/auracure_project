# =============================================================================
# services/api_service.py
# AuraEcho+ — Cardiology Guidelines & Mock API Service
#
# Responsibility:
#     Provide structured cardiology guidelines, treatment protocols, and
#     reference data. Acts as a mock API for clinical decision support.
#     All data is local — no external network calls.
#
# Public API:
#     get_guidelines(risk_level)           → dict
#     get_treatment_recommendations(...)   → dict
#     get_reference_data()                 → dict
#     get_risk_factor_info(factor)         → dict
#     get_emergency_protocols()            → dict
#     mock_api_call(endpoint, params)      → dict
# =============================================================================

from typing import Any, Dict, List, Optional

from utils.constants import (
    RISK_LEVELS,
    RISK_LABELS,
    FEATURE_LABELS,
    CHEST_PAIN_LABELS,
    THAL_LABELS,
    RESTECG_LABELS,
)
from utils.helpers import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# Clinical Guidelines Database
# Based on ACC/AHA guidelines (simplified for demonstration)
# ─────────────────────────────────────────────

_GUIDELINES: Dict[str, Dict[str, Any]] = {
    "LOW": {
        "summary": "Low cardiac risk. Focus on prevention and routine monitoring.",
        "acc_aha_class": "Class I",
        "recommendations": [
            "Continue routine cardiovascular risk assessment every 3-5 years",
            "Encourage heart-healthy lifestyle (diet, exercise, smoking cessation)",
            "Monitor blood pressure and cholesterol annually",
            "No immediate cardiac testing indicated",
        ],
        "follow_up": "Primary care follow-up in 6-12 months",
        "red_flags": [
            "New onset chest pain",
            "Unexplained shortness of breath",
            "Syncope or near-syncope",
        ],
    },
    "MEDIUM": {
        "summary": "Moderate cardiac risk. Further evaluation recommended.",
        "acc_aha_class": "Class IIa",
        "recommendations": [
            "Consider stress testing or coronary calcium scoring",
            "Optimize blood pressure control (target <130/80 mmHg)",
            "Evaluate for diabetes and metabolic syndrome",
            "Consider statin therapy based on ASCVD risk calculator",
            "Lifestyle modification counseling",
        ],
        "follow_up": "Cardiology consultation within 2-4 weeks",
        "red_flags": [
            "Chest pain with exertion",
            "Worsening exercise tolerance",
            "New ECG abnormalities",
        ],
    },
    "HIGH": {
        "summary": "High cardiac risk. Urgent evaluation required.",
        "acc_aha_class": "Class I",
        "recommendations": [
            "Urgent cardiology referral",
            "Consider coronary angiography if symptoms suggest ACS",
            "Initiate or intensify medical therapy (antiplatelet, statin, beta-blocker)",
            "Evaluate for revascularization if indicated",
            "Close monitoring of symptoms and vital signs",
        ],
        "follow_up": "Immediate cardiology evaluation (within 24-48 hours)",
        "red_flags": [
            "Chest pain at rest or with minimal exertion",
            "Signs of heart failure",
            "Hemodynamic instability",
            "Malignant arrhythmias",
        ],
    },
}


# ─────────────────────────────────────────────
# Treatment Protocols
# ─────────────────────────────────────────────

_TREATMENT_PROTOCOLS: Dict[str, Dict[str, Any]] = {
    "lifestyle": {
        "title": "Lifestyle Modifications",
        "items": [
            {
                "name": "Diet",
                "description": "Mediterranean or DASH diet; limit saturated fats and sodium",
                "priority": "high",
            },
            {
                "name": "Exercise",
                "description": "150 minutes moderate aerobic activity per week",
                "priority": "high",
            },
            {
                "name": "Smoking Cessation",
                "description": "Complete tobacco avoidance; offer cessation support",
                "priority": "critical",
            },
            {
                "name": "Weight Management",
                "description": "Target BMI 18.5-24.9; waist circumference <40\" (M) / <35\" (F)",
                "priority": "medium",
            },
        ],
    },
    "medication": {
        "title": "Pharmacologic Therapy",
        "items": [
            {
                "name": "Antiplatelet",
                "description": "Aspirin 81mg daily if ASCVD risk >10%",
                "priority": "medium",
                "risk_levels": ["MEDIUM", "HIGH"],
            },
            {
                "name": "Statin",
                "description": "Moderate to high-intensity statin based on risk",
                "priority": "high",
                "risk_levels": ["MEDIUM", "HIGH"],
            },
            {
                "name": "ACE Inhibitor/ARB",
                "description": "If hypertension, diabetes, or CKD present",
                "priority": "medium",
                "risk_levels": ["MEDIUM", "HIGH"],
            },
            {
                "name": "Beta-Blocker",
                "description": "If history of MI, heart failure, or angina",
                "priority": "high",
                "risk_levels": ["HIGH"],
            },
        ],
    },
    "diagnostic": {
        "title": "Diagnostic Testing",
        "items": [
            {
                "name": "Resting ECG",
                "description": "Baseline assessment for all patients",
                "priority": "high",
                "risk_levels": ["LOW", "MEDIUM", "HIGH"],
            },
            {
                "name": "Stress Test",
                "description": "Exercise or pharmacologic stress testing",
                "priority": "medium",
                "risk_levels": ["MEDIUM", "HIGH"],
            },
            {
                "name": "Echocardiogram",
                "description": "Assess cardiac structure and function",
                "priority": "medium",
                "risk_levels": ["MEDIUM", "HIGH"],
            },
            {
                "name": "Coronary Angiography",
                "description": "Invasive assessment if high suspicion of CAD",
                "priority": "high",
                "risk_levels": ["HIGH"],
            },
        ],
    },
}


# ─────────────────────────────────────────────
# Risk Factor Reference
# ─────────────────────────────────────────────

_RISK_FACTOR_INFO: Dict[str, Dict[str, Any]] = {
    "age": {
        "label": "Age",
        "description": "Cardiovascular risk increases with age",
        "thresholds": {
            "male": {"elevated": 45, "high": 65},
            "female": {"elevated": 55, "high": 65},
        },
        "guidance": "Age is a non-modifiable risk factor. Focus on controlling modifiable factors.",
    },
    "trestbps": {
        "label": "Resting Blood Pressure",
        "description": "Elevated BP increases cardiac workload and arterial damage",
        "thresholds": {
            "normal": "<120/80",
            "elevated": "120-129/<80",
            "hypertension_stage1": "130-139/80-89",
            "hypertension_stage2": "≥140/90",
        },
        "guidance": "Target BP <130/80 mmHg for most patients with cardiovascular risk.",
    },
    "chol": {
        "label": "Serum Cholesterol",
        "description": "High cholesterol contributes to atherosclerosis",
        "thresholds": {
            "desirable": "<200",
            "borderline": "200-239",
            "high": "≥240",
        },
        "guidance": "Consider statin therapy based on overall ASCVD risk, not just cholesterol level.",
    },
    "thalach": {
        "label": "Maximum Heart Rate",
        "description": "Lower achieved heart rate may indicate reduced exercise capacity",
        "formula": "Predicted max HR = 220 - age",
        "guidance": "Achieving <85% of predicted max HR may suggest chronotropic incompetence.",
    },
    "oldpeak": {
        "label": "ST Depression",
        "description": "ST depression during exercise suggests myocardial ischemia",
        "thresholds": {
            "normal": "0",
            "mild": "0.5-1.0",
            "moderate": "1.0-2.0",
            "severe": ">2.0",
        },
        "guidance": "Significant ST depression warrants further ischemic evaluation.",
    },
    "ca": {
        "label": "Major Vessels",
        "description": "Number of major coronary vessels with significant stenosis",
        "guidance": "Higher values indicate more extensive coronary artery disease.",
    },
    "thal": {
        "label": "Thalassemia",
        "description": "Thallium stress test results",
        "values": {
            "normal": "Normal perfusion",
            "fixed_defect": "Prior myocardial infarction",
            "reversible_defect": "Inducible ischemia",
        },
        "guidance": "Reversible defects suggest viable myocardium at risk.",
    },
}


# ─────────────────────────────────────────────
# Emergency Protocols
# ─────────────────────────────────────────────

_EMERGENCY_PROTOCOLS: Dict[str, Dict[str, Any]] = {
    "acs_suspected": {
        "title": "Suspected Acute Coronary Syndrome",
        "criteria": [
            "Chest pain >20 minutes",
            "Diaphoresis, nausea, dyspnea",
            "ST elevation or depression on ECG",
            "Elevated troponin",
        ],
        "immediate_actions": [
            "Activate emergency response",
            "Administer aspirin 325mg chewed",
            "Obtain 12-lead ECG within 10 minutes",
            "Establish IV access",
            "Monitor vital signs continuously",
        ],
        "consult": "Cardiology / Emergency Department immediately",
    },
    "heart_failure": {
        "title": "Acute Heart Failure",
        "criteria": [
            "Dyspnea at rest or with minimal exertion",
            "Orthopnea, PND",
            "Elevated JVP, pulmonary rales",
            "Peripheral edema",
        ],
        "immediate_actions": [
            "Position upright",
            "Supplemental oxygen if SpO2 <90%",
            "Diuretic therapy",
            "Consider vasodilators if hypertensive",
        ],
        "consult": "Cardiology / Emergency Department",
    },
    "arrhythmia": {
        "title": "Significant Arrhythmia",
        "criteria": [
            "Palpitations with hemodynamic compromise",
            "Syncope or near-syncope",
            "Heart rate >150 or <40 bpm",
            "Irregular rhythm with symptoms",
        ],
        "immediate_actions": [
            "Continuous cardiac monitoring",
            "Assess hemodynamic stability",
            "Prepare for cardioversion if unstable",
        ],
        "consult": "Cardiology / Emergency Department",
    },
}


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def get_guidelines(risk_level: str) -> Dict[str, Any]:
    """
    Get clinical guidelines for a specific risk level.

    Parameters
    ----------
    risk_level : "LOW" | "MEDIUM" | "HIGH"

    Returns
    -------
    dict with summary, recommendations, follow_up, red_flags
    """
    level = risk_level.upper()
    if level not in _GUIDELINES:
        logger.warning("Unknown risk level '%s' — returning MEDIUM guidelines", risk_level)
        level = "MEDIUM"

    guidelines = _GUIDELINES[level].copy()
    guidelines["risk_level"] = level
    guidelines["risk_label"] = RISK_LABELS.get(level, level)
    return guidelines


def get_treatment_recommendations(
    risk_level: str,
    top_factors: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Get personalized treatment recommendations based on risk level and factors.

    Parameters
    ----------
    risk_level  : "LOW" | "MEDIUM" | "HIGH"
    top_factors : list of feature names driving risk

    Returns
    -------
    dict with lifestyle, medication, diagnostic recommendations
    """
    level = risk_level.upper()
    if level not in RISK_LEVELS:
        level = "MEDIUM"

    recommendations = {
        "risk_level": level,
        "risk_label": RISK_LABELS.get(level, level),
        "lifestyle": _TREATMENT_PROTOCOLS["lifestyle"]["items"],
        "medication": [],
        "diagnostic": [],
    }

    # Filter medications by risk level
    for med in _TREATMENT_PROTOCOLS["medication"]["items"]:
        if level in med.get("risk_levels", []):
            recommendations["medication"].append(med)

    # Filter diagnostics by risk level
    for diag in _TREATMENT_PROTOCOLS["diagnostic"]["items"]:
        if level in diag.get("risk_levels", []):
            recommendations["diagnostic"].append(diag)

    # Add factor-specific guidance
    if top_factors:
        factor_guidance = []
        for factor in top_factors:
            if factor in _RISK_FACTOR_INFO:
                info = _RISK_FACTOR_INFO[factor]
                factor_guidance.append({
                    "factor": factor,
                    "label": info["label"],
                    "guidance": info["guidance"],
                })
        recommendations["factor_specific"] = factor_guidance

    return recommendations


def get_reference_data() -> Dict[str, Any]:
    """
    Get all reference data in one call.
    Useful for initializing UI reference panels.
    """
    return {
        "risk_levels": {
            k: {"label": v, "range": RISK_LEVELS[k]}
            for k, v in RISK_LABELS.items()
        },
        "chest_pain_types": CHEST_PAIN_LABELS,
        "thalassemia_types": THAL_LABELS,
        "restecg_types": RESTECG_LABELS,
        "feature_labels": FEATURE_LABELS,
        "emergency_protocols": list(_EMERGENCY_PROTOCOLS.keys()),
    }


def get_risk_factor_info(factor: str) -> Optional[Dict[str, Any]]:
    """
    Get detailed information about a specific risk factor.
    """
    if factor not in _RISK_FACTOR_INFO:
        return None
    info = _RISK_FACTOR_INFO[factor].copy()
    info["factor"] = factor
    return info


def get_emergency_protocols() -> Dict[str, Any]:
    """
    Get all emergency protocols.
    """
    return {
        "protocols": _EMERGENCY_PROTOCOLS,
        "disclaimer": (
            "These protocols are for reference only. Always follow your "
            "institution's emergency procedures and consult appropriate specialists."
        ),
    }


def get_emergency_protocol(protocol_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a specific emergency protocol by ID.
    """
    if protocol_id not in _EMERGENCY_PROTOCOLS:
        return None
    protocol = _EMERGENCY_PROTOCOLS[protocol_id].copy()
    protocol["id"] = protocol_id
    return protocol


# ─────────────────────────────────────────────
# Mock API Endpoints
# For testing UI integration without real API
# ─────────────────────────────────────────────

def mock_api_call(
    endpoint: str,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Simulate an external API call for testing purposes.

    Supported endpoints:
        /guidelines           → get_guidelines()
        /treatments           → get_treatment_recommendations()
        /reference            → get_reference_data()
        /risk-factor/{factor} → get_risk_factor_info()
        /emergency            → get_emergency_protocols()

    Parameters
    ----------
    endpoint : API endpoint path
    params   : Optional query parameters

    Returns
    -------
    dict with status, data, and mock metadata
    """
    params = params or {}
    logger.debug("Mock API call: %s with params %s", endpoint, params)

    try:
        if endpoint == "/guidelines":
            risk_level = params.get("risk_level", "MEDIUM")
            data = get_guidelines(risk_level)

        elif endpoint == "/treatments":
            risk_level = params.get("risk_level", "MEDIUM")
            top_factors = params.get("top_factors", [])
            data = get_treatment_recommendations(risk_level, top_factors)

        elif endpoint == "/reference":
            data = get_reference_data()

        elif endpoint.startswith("/risk-factor/"):
            factor = endpoint.split("/")[-1]
            data = get_risk_factor_info(factor)
            if data is None:
                return {
                    "status": "error",
                    "code": 404,
                    "message": f"Risk factor '{factor}' not found",
                    "data": None,
                }

        elif endpoint == "/emergency":
            data = get_emergency_protocols()

        elif endpoint.startswith("/emergency/"):
            protocol_id = endpoint.split("/")[-1]
            data = get_emergency_protocol(protocol_id)
            if data is None:
                return {
                    "status": "error",
                    "code": 404,
                    "message": f"Protocol '{protocol_id}' not found",
                    "data": None,
                }

        else:
            return {
                "status": "error",
                "code": 404,
                "message": f"Unknown endpoint '{endpoint}'",
                "data": None,
            }

        return {
            "status": "success",
            "code": 200,
            "message": "OK",
            "data": data,
            "meta": {
                "source": "mock_api",
                "endpoint": endpoint,
                "timestamp": "2024-01-01T00:00:00Z",
            },
        }

    except Exception as exc:
        logger.error("Mock API error: %s", exc)
        return {
            "status": "error",
            "code": 500,
            "message": f"Internal error: {exc}",
            "data": None,
        }


# ─────────────────────────────────────────────
# Convenience wrappers for UI
# ─────────────────────────────────────────────

def get_patient_care_plan(
    risk_level: str,
    top_factors: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Generate a comprehensive care plan for a patient.
    Combines guidelines, treatments, and factor-specific guidance.

    Used by ui/diagnosis_view.py for the care plan panel.
    """
    guidelines = get_guidelines(risk_level)
    treatments = get_treatment_recommendations(risk_level, top_factors)

    return {
        "risk_level": risk_level,
        "risk_label": RISK_LABELS.get(risk_level.upper(), risk_level),
        "summary": guidelines["summary"],
        "acc_aha_class": guidelines["acc_aha_class"],
        "immediate_actions": guidelines["recommendations"][:2],
        "follow_up": guidelines["follow_up"],
        "red_flags": guidelines["red_flags"],
        "lifestyle": treatments["lifestyle"],
        "medications": treatments["medication"],
        "diagnostics": treatments["diagnostic"],
        "factor_guidance": treatments.get("factor_specific", []),
        "disclaimer": (
            "This care plan is AI-generated and must be reviewed by a "
            "licensed physician before implementation."
        ),
    }


def get_risk_education(risk_level: str) -> Dict[str, Any]:
    """
    Get patient-friendly education content for a risk level.
    Used for patient education materials.
    """
    level = risk_level.upper()
    guidelines = get_guidelines(level)

    education = {
        "risk_level": level,
        "title": f"Understanding Your {RISK_LABELS.get(level, level)} Assessment",
        "what_this_means": guidelines["summary"],
        "what_you_can_do": [
            rec.replace("Consider", "Talk to your doctor about")
            for rec in guidelines["recommendations"][:3]
        ],
        "warning_signs": guidelines["red_flags"],
        "when_to_seek_help": (
            "Seek immediate medical attention if you experience chest pain, "
            "severe shortness of breath, or fainting."
        ),
    }
    return education