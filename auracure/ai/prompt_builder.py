# =============================================================================
# ai/prompt_builder.py
# AuraEcho+ — Structured Prompt Factory for Cardiac AI
#
# Responsibility:
#     Transform raw patient data, risk scores, and similar cases into
#     carefully structured prompts that guide the LLM to produce
#     clinically useful, formatted, and safe responses.
#
# Why a dedicated prompt builder?
#     Both offline_ai.py (Ollama/Llama3) and online_ai.py (Groq/OpenAI)
#     use IDENTICAL prompts — this file is the single source of truth.
#
# Public API:
#     build_diagnosis_prompt(patient, risk_result, similar_cases) → PromptPackage
#     build_followup_prompt(question, patient, history)           → PromptPackage
#     build_summary_prompt(patient, risk_result)                  → PromptPackage
#     build_alert_prompt(patient, risk_result, alert_reason)      → PromptPackage
#     validate_prompt_package(pkg)                                → Tuple[bool, str]
# =============================================================================

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from utils.constants import (
    APP_NAME,
    FEATURE_COLUMNS,
    FEATURE_LABELS,
    RISK_LEVELS,
    RISK_LABELS,
    RISK_COLORS,
    RISK_ICONS,
    CHEST_PAIN_LABELS,
    THAL_LABELS,
    SLOPE_LABELS,
    RESTECG_LABELS,
    LLM_MAX_TOKENS,
)
from utils.helpers import get_logger, normalize_risk_level, truncate

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# Categorical decode map
# FIXED: Built from constants — no duplication of decode maps.
# ─────────────────────────────────────────────

_DECODE: Dict[str, Dict[Any, str]] = {
    "sex":     {0: "Female",          1: "Male"},
    "cp":      CHEST_PAIN_LABELS,
    "fbs":     {0: "Normal (≤120 mg/dL)", 1: "High (>120 mg/dL)"},
    "restecg": RESTECG_LABELS,
    "exang":   {0: "No",              1: "Yes"},
    "slope":   SLOPE_LABELS,
    "thal":    THAL_LABELS,
}

# ─────────────────────────────────────────────
# Risk-level framing used in prompts
# Keys are "LOW" | "MEDIUM" | "HIGH" (consistent with RISK_LEVELS)
# ─────────────────────────────────────────────

_RISK_CONTEXT: Dict[str, str] = {
    "LOW": (
        "The automated risk model classified this patient as LOW RISK. "
        "Focus your analysis on preventive care, lifestyle recommendations, "
        "and routine monitoring schedules."
    ),
    "MEDIUM": (
        "The automated risk model classified this patient as MEDIUM RISK. "
        "Focus your analysis on identifying which factors are driving risk, "
        "what diagnostic tests should be ordered next, and short-term management."
    ),
    "HIGH": (
        "The automated risk model classified this patient as HIGH RISK. "
        "This is an URGENT assessment. Focus on immediate intervention options, "
        "emergency referral criteria, and critical monitoring parameters."
    ),
}


# ─────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────

@dataclass
class PromptPackage:
    """
    Everything an AI backend needs to make one LLM call.

    Attributes
    ----------
    system_prompt  : str  — AI role/persona instructions
    user_prompt    : str  — actual patient query
    prompt_type    : str  — "diagnosis" | "followup" | "summary" | "alert"
    token_estimate : int  — rough token count
    metadata       : dict — patient name, risk level, timestamp, etc.
    """
    system_prompt:  str
    user_prompt:    str
    prompt_type:    str                = "diagnosis"
    token_estimate: int                = 0
    metadata:       Dict[str, Any]     = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "system_prompt":  self.system_prompt,
            "user_prompt":    self.user_prompt,
            "prompt_type":    self.prompt_type,
            "token_estimate": self.token_estimate,
            "metadata":       self.metadata,
        }

    def full_prompt(self) -> str:
        """Concatenate system + user prompt for single-string models."""
        return f"{self.system_prompt}\n\n{self.user_prompt}"


# ─────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────

def _decode_feature(col: str, value: Any) -> str:
    """Convert a numeric feature value to a human-readable string."""
    if col in _DECODE:
        try:
            code = int(float(value))
            return _DECODE[col].get(code, str(value))
        except (ValueError, TypeError):
            return str(value)
    try:
        return str(round(float(value), 1))
    except (ValueError, TypeError):
        return str(value)


def _format_patient_vitals(patient: Dict[str, Any]) -> str:
    """
    Build the vitals section of the prompt.

    FIXED: Simplified format — no column alignment padding.
           LLMs don't need monospace alignment.
    """
    lines = []
    for col in FEATURE_COLUMNS:
        label    = FEATURE_LABELS.get(col, col)
        value    = patient.get(col, "N/A")
        readable = _decode_feature(col, value)
        lines.append(f"  • {label}: {readable}")
    return "\n".join(lines)


def _normalize_risk_for_prompt(risk_result: Optional[Dict[str, Any]]) -> Tuple[str, str, str]:
    """
    ADDED: Central helper to extract and normalize risk level from risk_result.

    FIXED: Handles both KEY ("HIGH") and LABEL ("High Risk") formats.
           Uses normalize_risk_level() from helpers.py.

    Returns
    -------
    (level_key, risk_label, risk_icon)
        level_key  : "LOW" | "MEDIUM" | "HIGH"
        risk_label : "Low Risk" | "Medium Risk" | "High Risk"
        risk_icon  : "✅" | "⚠️" | "🚨"
    """
    if not risk_result or not isinstance(risk_result, dict):
        return "MEDIUM", RISK_LABELS["MEDIUM"], RISK_ICONS["MEDIUM"]

    raw_level = risk_result.get("risk_level", "MEDIUM")
    level_key = normalize_risk_level(raw_level) or "MEDIUM"

    # Validate key is in RISK_LEVELS
    if level_key not in RISK_LEVELS:
        level_key = "MEDIUM"

    risk_label = RISK_LABELS.get(level_key, "Medium Risk")
    risk_icon  = RISK_ICONS.get(level_key, "⚠️")

    return level_key, risk_label, risk_icon


def _format_risk_section(risk_result: Optional[Dict[str, Any]]) -> str:
    """
    Format the risk model output section.

    FIXED:
    - Uses _normalize_risk_for_prompt() to handle key/label confusion.
    - badge_icon and badge_color now derived from level_key directly
      (no longer relies on missing keys in risk_result dict).
    """
    if not risk_result:
        return "  Risk assessment data not available."

    level_key, risk_label, risk_icon = _normalize_risk_for_prompt(risk_result)

    # FIXED: derive badge_color from level_key, not from risk_result dict
    badge_color = RISK_COLORS.get(level_key, "#95a5a6")

    confidence  = risk_result.get("confidence_pct", 0)
    prob        = risk_result.get("disease_prob", 0) * 100
    factors     = risk_result.get("top_risk_factors", [])
    explanation = risk_result.get("explanation", "")

    factors_str = ", ".join(factors) if factors else "Not determined"

    return (
        f"  Risk Level    : {risk_icon} {risk_label} ({level_key})\n"
        f"  Confidence    : {confidence:.1f}%\n"
        f"  Disease Prob  : {prob:.1f}%\n"
        f"  Key Drivers   : {factors_str}\n"
        f"  Model Summary : {explanation}"
    )


def _format_similar_cases(
    similar_cases: Optional[List[Dict[str, Any]]],
) -> str:
    """
    Format the top-3 similar historical cases section.
    similar_cases is a list of SimilarCase.to_dict() results.
    """
    if not similar_cases:
        return "  No similar cases available in the reference database."

    lines = []
    for case in similar_cases[:3]:
        rank       = case.get("rank", "?")
        sim_pct    = case.get("similarity_pct", 0)
        age        = case.get("age", "?")
        sex        = case.get("sex", "?")
        outcome    = case.get("outcome", "?")
        risk_key   = case.get("risk_level", "MEDIUM")
        risk_label = RISK_LABELS.get(risk_key, risk_key)
        risk_icon  = RISK_ICONS.get(risk_key, "❔")
        lines.append(
            f"  #{rank} ({sim_pct:.1f}% similar): "
            f"{age}yr {sex} — Outcome: {outcome} — {risk_icon} {risk_label}"
        )

    return "\n".join(lines)


def _estimate_tokens(text: str) -> int:
    """
    Rough token estimate: ~4 characters per token (GPT-style).
    Used to warn if prompt is near model context limits.
    """
    return len(text) // 4


def _build_system_prompt(risk_level_key: Optional[str] = None) -> str:
    """
    Build the system prompt that sets the AI role and constraints.

    FIXED: Parameter renamed to risk_level_key for clarity.
           Validates key against RISK_LEVELS before lookup.
    """
    key = (risk_level_key or "MEDIUM").upper()
    if key not in _RISK_CONTEXT:
        key = "MEDIUM"

    risk_context = _RISK_CONTEXT[key]

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


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def build_diagnosis_prompt(
    patient:       Dict[str, Any],
    risk_result:   Optional[Dict[str, Any]] = None,
    similar_cases: Optional[List[Dict[str, Any]]] = None,
) -> PromptPackage:
    """
    Build the primary diagnosis prompt package.

    FIXED:
    - Uses _normalize_risk_for_prompt() for safe key/label extraction.
    - sex_code None safety added.
    - _FEATURE_LABELS_DISPLAY removed — uses FEATURE_LABELS from constants.

    Parameters
    ----------
    patient       : raw patient dict (from UI form)
    risk_result   : RiskResult.to_dict() from core.risk_model
    similar_cases : list of SimilarCase.to_dict() from core.similarity

    Returns
    -------
    PromptPackage
    """
    patient_name = patient.get("name", "Unknown Patient")
    age          = patient.get("age", "?")

    # FIXED: Safe sex_code extraction with None guard
    sex_code = patient.get("sex", 1)
    try:
        sex_label = _DECODE["sex"].get(int(float(sex_code)), "Unknown")
    except (TypeError, ValueError):
        sex_label = "Unknown"

    # FIXED: normalize risk level using central helper
    level_key, risk_label, risk_icon = _normalize_risk_for_prompt(risk_result)

    vitals_section  = _format_patient_vitals(patient)
    risk_section    = _format_risk_section(risk_result)
    similar_section = _format_similar_cases(similar_cases)

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

    system_prompt  = _build_system_prompt(level_key)
    token_estimate = _estimate_tokens(system_prompt + user_prompt)

    logger.info(
        "Built diagnosis prompt for '%s' | risk=%s | ~%d tokens",
        patient_name, level_key, token_estimate,
    )

    return PromptPackage(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        prompt_type="diagnosis",
        token_estimate=token_estimate,
        metadata={
            "patient_name": patient_name,
            "risk_level":   level_key,
            "risk_label":   risk_label,
            "age":          age,
            "sex":          sex_label,
            "has_similar":  bool(similar_cases),
            "has_risk":     bool(risk_result),
            "app_name":     APP_NAME,
        },
    )


def build_followup_prompt(
    question:             str,
    patient:              Dict[str, Any],
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> PromptPackage:
    """
    Build a follow-up question prompt (conversational mode).

    FIXED: Uses helpers.truncate() for clean word-boundary truncation
           instead of raw [:300] slice.

    Parameters
    ----------
    question             : doctor/nurse's follow-up question
    patient              : current patient dict
    conversation_history : list of {"role": "user"|"assistant", "content": str}

    Returns
    -------
    PromptPackage
    """
    patient_name = patient.get("name", "the patient")
    age          = patient.get("age", "?")

    history_text = ""
    if conversation_history:
        history_lines = []
        # Keep last 6 turns (3 Q&A pairs)
        recent = conversation_history[-6:]
        for turn in recent:
            role    = "Doctor" if turn.get("role") == "user" else "AI"
            # FIXED: uses truncate() for clean word-boundary truncation
            content = truncate(turn.get("content", ""), max_len=300)
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

    system_prompt  = _build_system_prompt()
    token_estimate = _estimate_tokens(system_prompt + user_prompt)

    logger.info(
        "Built followup prompt for '%s' | ~%d tokens",
        patient_name, token_estimate,
    )

    return PromptPackage(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        prompt_type="followup",
        token_estimate=token_estimate,
        metadata={
            "patient_name": patient_name,
            "question":     truncate(question, max_len=100),
            "app_name":     APP_NAME,
        },
    )


def build_summary_prompt(
    patient:     Dict[str, Any],
    risk_result: Optional[Dict[str, Any]] = None,
) -> PromptPackage:
    """
    Build a short one-paragraph summary prompt.

    FIXED: Uses _normalize_risk_for_prompt() for consistent risk extraction.

    Used for dashboard summary card — brief, plain-language,
    non-technical summary for nurses or patient-facing displays.

    Returns
    -------
    PromptPackage  (token_estimate ~200–400)
    """
    patient_name = patient.get("name", "the patient")

    # FIXED: central risk normalization
    level_key, risk_label, risk_icon = _normalize_risk_for_prompt(risk_result)
    confidence = (
        risk_result.get("confidence_pct", 0.0)
        if risk_result and isinstance(risk_result, dict)
        else 0.0
    )

    vitals_section = _format_patient_vitals(patient)

    user_prompt = f"""
Write a SHORT (3-4 sentence) plain-language summary of this cardiac patient assessment.
Use simple language suitable for a non-specialist nurse or patient educator.
Do NOT use medical jargon. State the risk level and the 2 most important action items.

PATIENT: {patient_name}
RISK LEVEL: {risk_icon} {risk_label} ({level_key}) — confidence: {confidence:.1f}%
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
        metadata={
            "patient_name": patient_name,
            "risk_level":   level_key,
            "risk_label":   risk_label,
            "app_name":     APP_NAME,
        },
    )


def build_alert_prompt(
    patient:      Dict[str, Any],
    risk_result:  Optional[Dict[str, Any]] = None,
    alert_reason: str = "",
) -> PromptPackage:
    """
    ADDED: Build an urgent alert prompt for HIGH risk patients.
    Produces a triage-focused, concise output for emergency use.
    Used by ui/results_panel.py when risk_level == HIGH.

    Parameters
    ----------
    patient      : raw patient dict
    risk_result  : RiskResult.to_dict()
    alert_reason : reason for the alert (optional)

    Returns
    -------
    PromptPackage
    """
    patient_name = patient.get("name", "Unknown Patient")
    vitals       = _format_patient_vitals(patient)
    risk_section = _format_risk_section(risk_result)

    system_prompt = (
        "You are an emergency cardiology triage AI. "
        "Provide URGENT, concise clinical guidance. "
        "Focus only on immediate life-saving actions. "
        "Flag any STEMI/ACS criteria immediately with 🚨."
    )

    user_prompt = f"""
🚨 URGENT HIGH-RISK CARDIAC ALERT 🚨

PATIENT: {patient_name}
ALERT REASON: {alert_reason or "High cardiac risk score detected"}

VITALS:
{vitals}

RISK ASSESSMENT:
{risk_section}

Provide:
1. IMMEDIATE actions (next 30 minutes)
2. Emergency referral criteria met (yes/no + reason)
3. Critical monitoring parameters
4. Red flag symptoms to watch for NOW

Keep response under 200 words. Be direct and actionable.
""".strip()

    token_estimate = _estimate_tokens(system_prompt + user_prompt)

    logger.info(
        "Built alert prompt for '%s' | reason=%s | ~%d tokens",
        patient_name,
        truncate(alert_reason, max_len=50) if alert_reason else "N/A",
        token_estimate,
    )

    return PromptPackage(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        prompt_type="alert",
        token_estimate=token_estimate,
        metadata={
            "patient_name": patient_name,
            "alert_reason": alert_reason,
            "app_name":     APP_NAME,
        },
    )


def validate_prompt_package(pkg: PromptPackage) -> Tuple[bool, str]:
    """
    ADDED: Validate a PromptPackage before sending to LLM.

    Checks:
    - Prompts are non-empty
    - Token count is within model limits
    - prompt_type is valid

    Returns
    -------
    (is_valid: bool, warning_message: str)
    """
    valid_types = {"diagnosis", "followup", "summary", "alert"}

    if not pkg.system_prompt.strip():
        return False, "System prompt is empty."
    if not pkg.user_prompt.strip():
        return False, "User prompt is empty."
    if pkg.prompt_type not in valid_types:
        return False, f"Invalid prompt_type '{pkg.prompt_type}'. Must be one of {valid_types}."
    if pkg.token_estimate > LLM_MAX_TOKENS:
        return False, (
            f"Prompt too long: ~{pkg.token_estimate} tokens "
            f"(max: {LLM_MAX_TOKENS}). "
            f"Consider reducing similar cases or conversation history."
        )
    return True, ""