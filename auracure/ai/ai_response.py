# ai/ai_response.py
# Shared AI response components for AuraEcho+

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from utils.constants import (
    APP_NAME,
    RISK_LEVELS,
    RISK_LABELS,
    RISK_ICONS,
)
from utils.helpers import normalize_risk_level

# ─────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────

@dataclass
class AIResponse:
    """
    Standardised AI response — shared by offline and online backends.
    """
    content:       str
    source:        str
    model:         str
    prompt_tokens: int   = 0
    output_tokens: int   = 0
    latency_ms:    float = 0.0
    success:       bool  = True
    error:         str   = ""
    sections:      Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content":       self.content,
            "source":        self.source,
            "model":         self.model,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "latency_ms":    round(self.latency_ms, 1),
            "success":       self.success,
            "error":         self.error,
            "sections":      self.sections,
        }

    @property
    def is_empty(self) -> bool:
        """True if response failed OR has no meaningful content."""
        return not self.success or not bool(self.content.strip())


# ─────────────────────────────────────────────
# Response parser
# ─────────────────────────────────────────────

_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U00002600-\U000027BF"
    "\U0001F900-\U0001F9FF"
    "]+",
    flags=re.UNICODE,
)

def parse_sections(text: str) -> Dict[str, str]:
    """
    Parse ## headers into labelled sections dict.
    Uses regex to remove only emoji, preserving punctuation/digits.
    """
    sections: Dict[str, str] = {}
    current_header = "Introduction"
    current_lines: List[str] = []

    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("## "):
            if current_lines:
                sections[current_header] = "\n".join(current_lines).strip()
            raw_header = stripped[3:].strip()
            clean = _EMOJI_RE.sub("", raw_header).strip()
            current_header = clean if clean else raw_header
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections[current_header] = "\n".join(current_lines).strip()

    return sections


# ─────────────────────────────────────────────
# Fallback response
# ─────────────────────────────────────────────

def build_fallback_response(
    error:        str,
    patient_name: Optional[str] = None,
    risk_level:   Optional[str] = None,
) -> AIResponse:
    """
    Rule-based fallback when all AI backends fail.
    Uses normalize_risk_level() for key/label safety.
    """
    level_key = normalize_risk_level(risk_level) or "MEDIUM"
    risk_label_display = RISK_LABELS.get(level_key, "Unknown")
    risk_icon = RISK_ICONS.get(level_key, "❔")

    p_name = patient_name or "Unknown Patient"

    fallback_content = (
        f"## 🔍 Clinical Assessment\n"
        f"{APP_NAME} AI analysis is temporarily unavailable. "
        f"The risk score from the automated model is still valid.\n\n"
        f"## ⚠️ Key Risk Indicators\n"
        f"• Patient: {p_name}\n"
        f"• Risk Level: {risk_icon} {risk_label_display} ({level_key})\n"
        f"• Refer to Risk Score panel for automated assessment\n"
        f"• Review Similar Cases panel for historical context\n"
        f"• Consult ACC/AHA cardiac risk guidelines\n\n"
        f"## 💊 Treatment Recommendations\n"
        f"• **Immediate**: Review risk level and act accordingly\n"
        f"• **Short-term**: Cardiology consult if Medium/High risk\n"
        f"• **Long-term**: Lifestyle + medication review\n\n"
        f"## 🏥 Referral & Follow-up\n"
        f"• High Risk: Immediate cardiology referral\n"
        f"• Medium Risk: Cardiology within 1-2 weeks\n"
        f"• Low Risk: Primary care in 3-6 months\n\n"
        f"## ℹ️ System Note\n"
        f"AI unavailable: {error}"
    )

    return AIResponse(
        content=fallback_content,
        source="rule_based_fallback",
        model="rule_based",
        success=False,
        error=error,
        sections=parse_sections(fallback_content),
    )


# ─────────────────────────────────────────────
# Error sanitizer
# ─────────────────────────────────────────────

def safe_error_msg(exc: Exception, provider: str = "AI") -> str:
    """
    Convert technical exception to user-safe message.
    Prevents internal API details from being shown to clinical users.
    """
    exc_str = str(exc).lower()
    if "rate limit" in exc_str or "429" in exc_str:
        return f"\n⚠️ {provider} rate limit reached. Please try again later.\n"
    elif "timeout" in exc_str:
        return f"\n⚠️ {provider} timed out. Please check connectivity.\n"
    elif "connection" in exc_str:
        return f"\n⚠️ {provider} connection lost. Please check network.\n"
    elif "api key" in exc_str or "unauthorized" in exc_str or "401" in exc_str:
        return f"\n⚠️ {provider} API key invalid. Please check configuration.\n"
    else:
        return f"\n⚠️ {provider} temporarily unavailable. Using fallback guidance.\n"