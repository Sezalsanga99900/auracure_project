"""
ai/prompt_builder.py
────────────────────
Structured prompt factory for AuraEcho+ cardiac AI.

Responsibility:
    Transform raw patient data, risk scores, and similar cases into
    carefully structured prompts that guide the LLM to produce
    clinically useful, formatted, and safe responses.

Why a dedicated prompt builder?
    Both offline_ai.py (Ollama/Llama3) and online_ai.py (Groq/OpenAI)
    use IDENTICAL prompts — this file is the single source of truth.
    Changing the prompt format here updates both AI backends at once.

Prompt anatomy:
    ┌──────────────────────────────────────────────────────────┐
    │  SYSTEM PROMPT  — sets the AI's role and constraints     │
    ├──────────────────────────────────────────────────────────┤
    │  USER PROMPT    — structured patient brief:              │
    │    • Demographics + Vitals section                       │
    │    • Risk Assessment section (from risk_model.py)        │
    │    • Similar Cases section  (from similarity.py)         │
    │    • Clinical Question section                           │
    └──────────────────────────────────────────────────────────┘

Public API:
    build_diagnosis_prompt(patient, risk_result, similar_cases)
        → PromptPackage (system_prompt, user_prompt, metadata)

    build_followup_prompt(question, patient, conversation_history)
        → PromptPackage

    build_summary_prompt(patient, risk_result)
        → PromptPackage  (short one-paragraph summary)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from utils.constants import (
    RISK_LOW, RISK_MEDIUM, RISK_HIGH,
    FEATURE_COLUMNS,
    APP_NAME,
)
from utils.helpers import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────
# Human-readable labels for every clinical feature
# ─────────────────────────────────────────────────────────────
_FEATURE_LABELS: Dict[str, str] = {
    "age":      "Age (years)",
    "sex":      "Sex",
    "cp":       "Chest Pain Type",
    "trestbps": "Resting Blood Pressure (mmHg)",
    "chol":     "Serum Cholesterol (mg/dL)",
    "fbs":      "Fasting Blood Sugar > 120 mg/dL",
    "restecg":  "Resting ECG Result",
    "thalach":  "Max Heart Rate Achieved (bpm)",
    "exang":    "Exercise-Induced Angina",
    "oldpeak":  "ST Depression (Exercise vs Rest)",
    "slope":    "Slope of Peak ST Segment",
    "ca":       "Number of Major Vessels (Fluoroscopy)",
    "thal":     "Thalassemia Type",
}

# ─────────────────────────────────────────────────────────────
# Categorical decode maps (numeric code → readable string)
# ─────────────────────────────────────────────────────────────
_DECODE: Dict[str, Dict[Any, str]] = {
    "sex":     {0: "Female", 1: "Male"},
    "cp":      {0: "Typical Angina", 1: "Atypical Angina",
                2: "Non-Anginal Pain", 3: "Asymptomatic"},
    "fbs":     {0: "Normal (≤120 mg/dL)", 1: "High (>120 mg/dL)"},
    "restecg": {0: "Normal", 1: "ST-T Wave Abnormality",
                2: "Left Ventricular Hypertrophy"},
    "exang":   {0: "No", 1: "Yes"},
    "slope":   {0: "Upsloping", 1: "Flat", 2: "Downsloping"},
    "thal":    {0: "Normal", 1: "Fixed Defect",
                2: "Reversible Defect", 3: "Unknown"},
}

# ─────────────────────────────────────────────────────────────
# Risk-level framing used in prompts
# ─────────────────────────────────────────────────────────────
_RISK_CONTEXT: Dict[str, str] = {
    RISK_LOW: (
        "The automated risk model classified this patient as LOW RISK. "
        "Focus your analysis on preventive care, lifestyle recommendations, "
        "and routine monitoring schedules."
    ),
    RISK_MEDIUM: (
        "The automated risk model classified this patient as MEDIUM RISK. "
        "Focus your analysis on identifying which factors are driving risk, "
        "what diagnostic tests should be ordered next, and short-term management."
    ),
    RISK_HIGH: (
        "The automated risk model classified this patient as HIGH RISK. "
        "This is an URGENT assessment. Focus on immediate intervention options, "
        "emergency referral criteria, and critical monitoring parameters."
    ),
}


# ─────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────
@dataclass
class PromptPackage:
    """
    Everything an AI backend needs to make one LLM call.

    Attributes
    ----------
    system_prompt   : str — the AI's role/persona instructions
    user_prompt     : str — the actual patient query
    prompt_type     : str — "diagnosis" | "followup" | "summary"
    token_estimate  : int — rough token count (for model selection)
    metadata        : dict — patient name, risk level, timestamp, etc.
    """
    system_prompt:  str
    user_prompt:    str
    prompt_type:    str                     = "diagnosis"
    token_estimate: int                     = 0
    metadata:       Dict[str, Any]          = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "system_prompt":  self.system_prompt,
            "user_prompt":    self.user_prompt,
            "prompt_type":    self.prompt_type,
            "token_estimate": self.token_estimate,
            "metadata":       self.metadata,
        }

    def full_prompt(self) -> str:
        """Concatenate system + user prompt for models that use a single string."""
        return f"{self.system_prompt}\n\n{self.user_prompt}"


# ─────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────

def _decode_feature(col: str, value: Any) -> str:
    """Convert a numeric feature value to a human-readable string."""
    if col in _DECODE:
        try:
            code = int(float(value))
            return _DECODE[col].get(code, str(value))
        except (ValueError, TypeError):
            return str(value)
    # Continuous feature — round and return as-is
    try:
        return str(round(float(value), 1))
    except (ValueError, TypeError):
        return str(value)


def _format_patient_vitals(patient: Dict[str, Any]) -> str:
    """
    Build the vitals section of the prompt.

    Returns a neatly formatted multi-line string, e.g.:
        • Age (years)                : 55
        • Sex                        : Male
        • Chest Pain Type            : Typical Angina
        ...
    """
    lines = []
    for col in FEATURE_COLUMNS:
        label = _FEATURE_LABELS.get(col, col)
        value = patient.get(col, "N/A")
        readable = _decode_feature(col, value)
        lines.append(f"  • {label:<42}: {readable}")
    return "\n".join(lines)


def _format_risk_section(risk_result: Optional[Dict[str, Any]]) -> str:
    """
    Format the risk model output section.

    risk_result is the .to_dict() of a RiskResult dataclass.
    """
    if not risk_result:
        return "  Risk assessment data not available."

    level      = risk_result.get("risk_level", "Unknown")
    confidence = risk_result.get("confidence_pct", 0)
    prob       = risk_result.get("disease_prob", 0) * 100
    factors    = risk_result.get("top_risk_factors", [])
    explanation= risk_result.get("explanation", "")

    factors_str = ", ".join(factors) if factors else "Not determined"

    return (
        f"  Risk Level    : {level}\n"
        f"  Confidence    : {confidence:.1f}%\n"
        f"  Disease Prob  : {prob:.1f}%\n"
        f"  Key Drivers   : {factors_str}\n"
        f"  Model Summary : {explanation}"
    )


def _format_similar_cases(similar_cases: Optional[List[Dict[str, Any]]]) -> str:
    """
    Format the top-3 similar historical cases section.

    similar_cases is a list of SimilarCase.to_dict() results.
    """
    if not similar_cases:
        return "  No similar cases available in the reference database."

    lines = []
    for case in similar_cases[:3]:
        rank    = case.get("rank", "?")
        sim_pct = case.get("similarity_pct", 0)
        age     = case.get("age", "?")
        sex     = case.get("sex", "?")
        outcome = case.get("outcome", "?")
        risk    = case.get("risk_level", "?")
        lines.append(
            f"  #{rank} ({sim_pct:.1f}% similar): "
            f"{age}yr {sex} — Outcome: {outcome} — Risk: {risk}"
        )

    return "\n".join(lines)


def _estimate_tokens(text: str) -> int:
    """
    Rough token estimate: ~4 characters per token (GPT-style tokenization).
    Used to warn if prompt is near model context limits.
    """
    return len(text) // 4


def _build_system_prompt(risk_level: Optional[str] = None) -> str:
    """
    Build the system prompt that sets the AI's role and constraints.
    """
    risk_context = _RISK_CONTEXT.get(risk_level or RISK_MEDIUM, "")

    return f"""You are a senior cardiologist AI assistant integrated into {APP_NAME}, \
a clinical decision support system used by licensed medical professionals.

YOUR ROLE:
- Analyze patient cardiac data and provide structured clinical insights
- Support (not replace) the treating physician's judgment
- Provide evidence-based recommendations aligned with ACC/AHA guidelines
- Flag urgent findings that require immediate attention

{risk_context}

OUTPUT FORMAT — Always respond with these exact sections:

## 🔍 Clinical Assessment
[2-3 sentences summarizing the overall cardiac picture]

## ⚠️ Key Risk Indicators
[Bullet list of the 3-5 most concerning clinical findings]

## 🔮 Potential Future Symptoms
[Bullet list of 3-4 symptoms to watch for, with timeframes]

## 💊 Treatment Recommendations
[Structured list: Immediate / Short-term / Long-term actions]

## 🏥 Referral & Follow-up
[Who to refer to, when, and what tests to order]

## 📋 Patient Education Points
[2-3 plain-language points the patient should understand]

CONSTRAINTS:
- Never make a definitive diagnosis — use language like "suggests", "indicates", "consistent with"
- Always recommend physician review for High risk cases
- Flag any life-threatening patterns with 🚨 prefix
- Keep medical terminology balanced with plain language
- Base all recommendations on the data provided, not assumptions"""


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def build_diagnosis_prompt(
    patient:       Dict[str, Any],
    risk_result:   Optional[Dict[str, Any]] = None,
    similar_cases: Optional[List[Dict[str, Any]]] = None,
) -> PromptPackage:
    """
    Build the primary diagnosis prompt package.

    Parameters
    ----------
    patient       : raw patient dict (from UI form)
    risk_result   : RiskResult.to_dict() from risk_model.py
    similar_cases : list of SimilarCase.to_dict() from similarity.py

    Returns
    -------
    PromptPackage — pass directly to offline_ai or online_ai
    """
    # Extract convenience fields
    patient_name = patient.get("name", "Unknown Patient")
    age          = patient.get("age", "?")
    sex_code     = patient.get("sex", 1)
    sex_label    = _DECODE["sex"].get(int(float(sex_code)), "Unknown")
    risk_level   = risk_result.get("risk_level") if risk_result else None

    # Build sections
    vitals_section  = _format_patient_vitals(patient)
    risk_section    = _format_risk_section(risk_result)
    similar_section = _format_similar_cases(similar_cases)

    # Compose user prompt
    user_prompt = f"""
═══════════════════════════════════════════════════════════
CARDIAC PATIENT ASSESSMENT REQUEST
═══════════════════════════════════════════════════════════

PATIENT: {patient_name} | Age: {age} | Sex: {sex_label}

─────────────────────────────────────────
SECTION 1: CLINICAL VITALS & FEATURES
─────────────────────────────────────────
{vitals_section}

─────────────────────────────────────────
SECTION 2: AUTOMATED RISK MODEL OUTPUT
─────────────────────────────────────────
{risk_section}

─────────────────────────────────────────
SECTION 3: SIMILAR HISTORICAL CASES (Top 3)
─────────────────────────────────────────
{similar_section}

─────────────────────────────────────────
CLINICAL QUESTION
─────────────────────────────────────────
Based on the above data, please provide:
1. A comprehensive cardiac risk assessment
2. The most likely clinical trajectory for this patient
3. Immediate and long-term management recommendations
4. Any urgent flags that require immediate physician attention

Please follow the structured output format specified in your instructions.
═══════════════════════════════════════════════════════════
""".strip()

    system_prompt = _build_system_prompt(risk_level)
    token_estimate = _estimate_tokens(system_prompt + user_prompt)

    logger.info(
        "Built diagnosis prompt for '%s' | risk=%s | ~%d tokens",
        patient_name, risk_level, token_estimate,
    )

    return PromptPackage(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        prompt_type="diagnosis",
        token_estimate=token_estimate,
        metadata={
            "patient_name": patient_name,
            "risk_level":   risk_level,
            "age":          age,
            "sex":          sex_label,
            "has_similar":  bool(similar_cases),
            "has_risk":     bool(risk_result),
        },
    )


def build_followup_prompt(
    question:             str,
    patient:              Dict[str, Any],
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> PromptPackage:
    """
    Build a follow-up question prompt (conversational mode).

    Parameters
    ----------
    question             : doctor/nurse's follow-up question
    patient              : current patient dict (for context)
    conversation_history : list of {"role": "user"|"assistant", "content": str}

    Returns
    -------
    PromptPackage
    """
    patient_name = patient.get("name", "the patient")
    age          = patient.get("age", "?")

    # Summarise conversation history
    history_text = ""
    if conversation_history:
        history_lines = []
        for turn in conversation_history[-6:]:   # last 3 Q&A pairs
            role    = "Doctor" if turn.get("role") == "user" else "AI"
            content = turn.get("content", "")[:300]   # truncate long turns
            history_lines.append(f"{role}: {content}")
        history_text = "\n".join(history_lines)

    user_prompt = f"""
FOLLOW-UP QUESTION about {patient_name} (Age: {age})

CONVERSATION HISTORY:
{history_text if history_text else "No prior conversation."}

─────────────────────────────
FOLLOW-UP QUESTION:
{question}
─────────────────────────────

Please answer concisely and specifically, referencing the patient's
clinical data where relevant. Use the same structured format as before.
""".strip()

    system_prompt = _build_system_prompt()
    token_estimate = _estimate_tokens(system_prompt + user_prompt)

    logger.info("Built followup prompt | ~%d tokens", token_estimate)

    return PromptPackage(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        prompt_type="followup",
        token_estimate=token_estimate,
        metadata={"patient_name": patient_name, "question": question[:100]},
    )


def build_summary_prompt(
    patient:     Dict[str, Any],
    risk_result: Optional[Dict[str, Any]] = None,
) -> PromptPackage:
    """
    Build a short one-paragraph summary prompt.

    Used for the dashboard summary card — brief, plain-language,
    non-technical summary for nurses or for patient-facing displays.

    Returns
    -------
    PromptPackage  (token_estimate will be ~200–400)
    """
    patient_name = patient.get("name", "the patient")
    risk_level   = risk_result.get("risk_level", "Unknown") if risk_result else "Unknown"
    confidence   = risk_result.get("confidence_pct", 0) if risk_result else 0

    vitals_section = _format_patient_vitals(patient)

    user_prompt = f"""
Write a SHORT (3-4 sentence) plain-language summary of this cardiac patient assessment.
Use simple language suitable for a non-specialist nurse or patient educator.
Do NOT use medical jargon. State the risk level and the 2 most important action items.

PATIENT: {patient_name}
RISK LEVEL: {risk_level} (confidence: {confidence:.1f}%)
VITALS:
{vitals_section}
""".strip()

    summary_system = (
        "You are a clinical communication specialist. Write clear, compassionate, "
        "jargon-free patient summaries for healthcare staff. "
        "Always recommend professional medical review."
    )

    token_estimate = _estimate_tokens(summary_system + user_prompt)

    return PromptPackage(
        system_prompt=summary_system,
        user_prompt=user_prompt,
        prompt_type="summary",
        token_estimate=token_estimate,
        metadata={"patient_name": patient_name, "risk_level": risk_level},
    )