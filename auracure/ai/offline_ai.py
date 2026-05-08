# =============================================================================
# ai/offline_ai.py
# AuraEcho+ — Offline AI Backend (Ollama + Llama 3)
#
# Responsibility:
#     Provide AI-powered cardiac diagnosis when the system is OFFLINE
#     or when privacy mode is enabled. All inference runs 100% locally.
#
# Public API:
#     analyze_patient(patient, risk_result, similar_cases) → AIResponse
#     is_ollama_available()                                → bool
#     get_ollama_status()                                  → dict
#     stream_analysis(prompt_package)                      → Generator[str]
#     analyze_with_question(...)                           → AIResponse
#     get_model_info()                                     → dict
#     warmup_model()                                       → bool
# =============================================================================

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from ai.prompt_builder import (
    PromptPackage,
    build_diagnosis_prompt,
    build_summary_prompt,
    build_followup_prompt,
)
from utils.constants import (
    APP_NAME,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT,
    OLLAMA_MAX_TOKENS,
    OLLAMA_TEMPERATURE,
    CONNECTIVITY_RETRIES,
    RISK_LEVELS,
    RISK_LABELS,
    RISK_COLORS,
    RISK_ICONS,
)
from utils.helpers import get_logger, normalize_risk_level

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────

@dataclass
class AIResponse:
    """
    Standardised AI response — same structure from both offline and online backends.

    Attributes
    ----------
    content       : str   — full AI-generated text
    source        : str   — "offline_llama3" | "online_groq" | "online_openai"
    model         : str   — exact model identifier used
    prompt_tokens : int   — input tokens consumed
    output_tokens : int   — output tokens generated
    latency_ms    : float — wall-clock time for the LLM call
    success       : bool  — False if an error occurred
    error         : str   — error message if success=False
    sections      : dict  — parsed response sections
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
        """
        FIXED: Now checks success flag as well as content.
        Returns True if response failed OR has no meaningful content.
        """
        return not self.success or not bool(self.content.strip())


# ─────────────────────────────────────────────
# Response parser
# FIXED: Made public (parse_sections) for potential shared use.
#        Uses regex to strip emoji correctly (preserves punctuation).
# ─────────────────────────────────────────────

_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"   # emoticons
    "\U0001F300-\U0001F5FF"   # symbols & pictographs
    "\U0001F680-\U0001F6FF"   # transport & map
    "\U00002600-\U000027BF"   # misc symbols
    "\U0001F900-\U0001F9FF"   # supplemental symbols
    "]+",
    flags=re.UNICODE,
)

def parse_sections(text: str) -> Dict[str, str]:
    """
    Parse the structured AI response into labelled sections.

    FIXED: Uses regex to remove only emoji, preserving punctuation
           and digits in headers (e.g., "Referral & Follow-up" stays intact).

    Returns
    -------
    dict: { "Clinical Assessment": "...", "Key Risk Indicators": "..." }
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
            # FIXED: Remove only emoji, preserve punctuation/digits
            clean = _EMOJI_RE.sub("", raw_header).strip()
            current_header = clean if clean else raw_header
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections[current_header] = "\n".join(current_lines).strip()

    return sections


# ─────────────────────────────────────────────
# Fallback response builder
# FIXED: Made public, uses normalize_risk_level for key/label safety.
# ─────────────────────────────────────────────

def build_fallback_response(
    error:        str,
    patient_name: Optional[str] = None,
    risk_level:   Optional[str] = None,
) -> AIResponse:
    """
    Return a safe fallback response when Ollama is unavailable.
    The fallback gives basic clinical guidance based on rule-based logic.

    FIXED:
    - Uses normalize_risk_level() to handle key/label confusion.
    - Returns AIResponse with success=False.
    """
    # FIXED: Normalize risk level
    level_key = normalize_risk_level(risk_level) or "MEDIUM"
    risk_label_display = RISK_LABELS.get(level_key, "Unknown")
    risk_icon = RISK_ICONS.get(level_key, "❔")

    p_name = patient_name or "Unknown Patient"

    fallback_content = (
        f"## 🔍 Clinical Assessment\n"
        f"{APP_NAME} AI analysis is temporarily unavailable (offline mode — Ollama not running). "
        f"The risk score from the automated model is still valid and should guide clinical decisions.\n\n"
        f"## ⚠️ Key Risk Indicators\n"
        f"• Patient: {p_name}\n"
        f"• Risk Level: {risk_icon} {risk_label_display} ({level_key})\n"
        f"• Please refer to the Risk Score panel for automated risk assessment\n"
        f"• Review the Similar Cases panel for historical context\n"
        f"• Consult standard ACC/AHA cardiac risk guidelines\n\n"
        f"## 💊 Treatment Recommendations\n"
        f"• **Immediate**: Review automated risk level and act accordingly\n"
        f"• **Short-term**: Schedule cardiology consultation if Medium or High risk\n"
        f"• **Long-term**: Lifestyle modifications + medication review with physician\n\n"
        f"## 🏥 Referral & Follow-up\n"
        f"• High Risk: Immediate cardiology referral\n"
        f"• Medium Risk: Cardiology within 1-2 weeks\n"
        f"• Low Risk: Primary care follow-up in 3-6 months\n\n"
        f"## ℹ️ System Note\n"
        f"Offline AI unavailable: {error}\n"
        f"To enable AI analysis: ensure Ollama is running (`ollama serve`) "
        f"and the model is pulled (`ollama pull {OLLAMA_MODEL}`)."
    )

    return AIResponse(
        content=fallback_content,
        source="offline_fallback",
        model="rule_based",
        success=False,
        error=error,
        sections=parse_sections(fallback_content),
    )


# ─────────────────────────────────────────────
# Ollama connectivity helpers
# ─────────────────────────────────────────────

def is_ollama_available() -> bool:
    """
    Check if Ollama service is running and the target model is loaded.
    Returns True if Ollama is reachable, False otherwise.
    """
    if not REQUESTS_AVAILABLE:
        logger.warning("requests library not installed — cannot check Ollama")
        return False

    try:
        resp = requests.get(
            f"{OLLAMA_BASE_URL}/api/tags",
            timeout=3.0,
        )
        if resp.status_code == 200:
            tags = resp.json()
            models = [m["name"] for m in tags.get("models", [])]
            available = any(OLLAMA_MODEL in m for m in models)
            if not available:
                logger.warning(
                    "Ollama is running but model '%s' not found. "
                    "Run: ollama pull %s",
                    OLLAMA_MODEL, OLLAMA_MODEL,
                )
            return available
        return False
    except Exception as exc:
        logger.debug("Ollama availability check failed: %s", exc)
        return False


def get_ollama_status() -> Dict[str, Any]:
    """
    Return a detailed status dict for the system status panel.
    """
    if not REQUESTS_AVAILABLE:
        return {
            "available": False,
            "base_url": OLLAMA_BASE_URL,
            "model": OLLAMA_MODEL,
            "model_loaded": False,
            "loaded_models": [],
            "error": "requests library not installed",
        }

    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3.0)
        if resp.status_code != 200:
            return {
                "available": False,
                "base_url": OLLAMA_BASE_URL,
                "model": OLLAMA_MODEL,
                "model_loaded": False,
                "loaded_models": [],
                "error": f"HTTP {resp.status_code}",
            }

        tags = resp.json()
        loaded_models = [m["name"] for m in tags.get("models", [])]
        model_loaded  = any(OLLAMA_MODEL in m for m in loaded_models)

        return {
            "available":    True,
            "base_url":     OLLAMA_BASE_URL,
            "model":        OLLAMA_MODEL,
            "model_loaded": model_loaded,
            "loaded_models": loaded_models,
            "error":        "" if model_loaded else f"Model '{OLLAMA_MODEL}' not pulled",
        }
    except Exception as exc:
        return {
            "available": False,
            "base_url": OLLAMA_BASE_URL,
            "model": OLLAMA_MODEL,
            "model_loaded": False,
            "loaded_models": [],
            "error": str(exc),
        }


# ─────────────────────────────────────────────
# Core Ollama call
# FIXED: Added retry logic with exponential backoff.
# ─────────────────────────────────────────────

def _call_ollama(
    system_prompt: str,
    user_prompt:   str,
    stream:        bool = False,
    retries:       int  = CONNECTIVITY_RETRIES,
) -> Any:
    """
    Make the HTTP POST to Ollama's /api/chat endpoint with retry logic.

    FIXED: Implements CONNECTIVITY_RETRIES with exponential backoff.
           Retries on transient errors (timeout, connection refused, 5xx).
    """
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "stream": stream,
        "options": {
            "temperature":   OLLAMA_TEMPERATURE,
            "num_predict":   OLLAMA_MAX_TOKENS,
            "top_p":         0.9,
            "repeat_penalty": 1.1,
        },
    }

    last_exc = None

    for attempt in range(1, retries + 1):
        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json=payload,
                timeout=OLLAMA_TIMEOUT,
                stream=stream,
            )
            response.raise_for_status()
            return response if stream else response.json()

        except requests.RequestException as exc:
            last_exc = exc
            err_str = str(exc).lower()
            is_transient = any(
                x in err_str for x in
                ["timeout", "connection", "503", "502", "500"]
            )

            if is_transient and attempt < retries:
                wait = 2 ** (attempt - 1)  # 1s, 2s, 4s...
                logger.warning(
                    "Ollama attempt %d/%d failed — retrying in %ds: %s",
                    attempt, retries, wait, exc,
                )
                time.sleep(wait)
            else:
                logger.error(
                    "Ollama call failed after %d attempts: %s", retries, exc
                )
                raise

    raise last_exc


# ─────────────────────────────────────────────
# Error message sanitizer
# ADDED: Sanitizes technical errors for clinical users.
# ─────────────────────────────────────────────

def _safe_error_msg(exc: Exception, provider: str = "Ollama") -> str:
    """
    Convert technical exception to user-safe message.
    Prevents internal API details from being shown to clinical users.
    """
    exc_str = str(exc).lower()
    if "rate limit" in exc_str or "429" in exc_str:
        return f"\n⚠️ {provider} rate limit reached. Please try again later.\n"
    elif "timeout" in exc_str:
        return f"\n⚠️ {provider} timed out. Please check system load.\n"
    elif "connection" in exc_str:
        return f"\n⚠️ {provider} connection lost. Please ensure service is running.\n"
    else:
        return f"\n⚠️ {provider} temporarily unavailable. Using fallback guidance.\n"


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def analyze_patient(
    patient:       Dict[str, Any],
    risk_result:   Optional[Dict[str, Any]] = None,
    similar_cases: Optional[List[Dict[str, Any]]] = None,
) -> AIResponse:
    """
    Run a full offline AI cardiac analysis for one patient.

    Returns AIResponse — never raises, falls back gracefully.
    """
    if not REQUESTS_AVAILABLE:
        return build_fallback_response(
            "requests library not installed",
            patient_name=patient.get("name"),
            risk_level=risk_result.get("risk_level") if risk_result else None,
        )

    if not is_ollama_available():
        return build_fallback_response(
            f"Ollama not running or model '{OLLAMA_MODEL}' not loaded",
            patient_name=patient.get("name"),
            risk_level=risk_result.get("risk_level") if risk_result else None,
        )

    pkg = build_diagnosis_prompt(patient, risk_result, similar_cases)

    patient_name   = patient.get("name", "Unknown Patient")
    risk_level_val = risk_result.get("risk_level") if risk_result else None

    logger.info(
        "Calling Ollama/%s | prompt_tokens~%d | patient=%s | risk=%s",
        OLLAMA_MODEL, pkg.token_estimate, patient_name, risk_level_val,
    )

    t0 = time.monotonic()
    try:
        raw = _call_ollama(pkg.system_prompt, pkg.user_prompt, stream=False)
        latency_ms = (time.monotonic() - t0) * 1000

        content = (
            raw.get("message", {}).get("content", "")
            or raw.get("response", "")
        )

        prompt_tokens = raw.get("prompt_eval_count", pkg.token_estimate)
        output_tokens = raw.get("eval_count", 0)

        sections = parse_sections(content)

        logger.info(
            "Ollama response: %d chars | %d output tokens | %.0f ms | patient=%s",
            len(content), output_tokens, latency_ms, patient_name,
        )

        return AIResponse(
            content=content,
            source="offline_llama3",
            model=OLLAMA_MODEL,
            prompt_tokens=int(prompt_tokens),
            output_tokens=int(output_tokens),
            latency_ms=latency_ms,
            success=True,
            sections=sections,
        )

    except requests.RequestException as exc:
        latency_ms = (time.monotonic() - t0) * 1000
        logger.error(
            "Ollama HTTP call failed after %.0f ms (patient=%s): %s",
            latency_ms, patient_name, exc,
        )
        return build_fallback_response(
            f"Ollama HTTP error: {exc}", patient_name, risk_level_val
        )

    except json.JSONDecodeError as exc:
        latency_ms = (time.monotonic() - t0) * 1000
        logger.error(
            "Ollama response JSON decode failed after %.0f ms (patient=%s): %s",
            latency_ms, patient_name, exc,
        )
        return build_fallback_response(
            f"Ollama invalid JSON: {exc}", patient_name, risk_level_val
        )

    except Exception as exc:
        latency_ms = (time.monotonic() - t0) * 1000
        logger.error(
            "Ollama call failed after %.0f ms (patient=%s): %s",
            latency_ms, patient_name, exc,
        )
        return build_fallback_response(str(exc), patient_name, risk_level_val)


def stream_analysis(
    patient:       Dict[str, Any],
    risk_result:   Optional[Dict[str, Any]] = None,
    similar_cases: Optional[List[Dict[str, Any]]] = None,
) -> Generator[str, None, None]:
    """
    Stream the AI response token-by-token.

    FIXED:
    - Removed redundant is_ollama_available() check (TOCTOU fix).
    - Let _call_ollama raise, catch exception.
    - Sanitize error messages for clinical users.

    Usage in Streamlit:
        with st.chat_message("assistant"):
            st.write_stream(stream_analysis(patient, risk_result, similar_cases))
    """
    if not REQUESTS_AVAILABLE:
        yield "⚠️ requests library not installed."
        return

    pkg = build_diagnosis_prompt(patient, risk_result, similar_cases)
    patient_name = patient.get("name", "Unknown Patient")

    try:
        # FIXED: No pre-check — call directly, handle exceptions
        raw_response = _call_ollama(pkg.system_prompt, pkg.user_prompt, stream=True)

        for line in raw_response.iter_lines():
            if line:
                try:
                    chunk = json.loads(line.decode("utf-8"))
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if chunk.get("done", False):
                        break
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue

    except requests.ConnectionError as exc:
        logger.error("Ollama connection lost during stream: %s", exc)
        yield _safe_error_msg(exc, "Ollama")

    except requests.Timeout as exc:
        logger.error("Ollama stream timed out: %s", exc)
        yield _safe_error_msg(exc, "Ollama")

    except Exception as exc:
        logger.error("Streaming failed for patient '%s': %s", patient_name, exc)
        yield _safe_error_msg(exc, "Ollama")


def analyze_with_question(
    question:             str,
    patient:              Dict[str, Any],
    conversation_history: Optional[List[Dict[str, str]]] = None,
    risk_result:          Optional[Dict[str, Any]] = None,   # ADDED
) -> AIResponse:
    """
    Answer a follow-up question about a patient (conversational mode).

    ADDED: risk_result parameter — passed to fallback for context.
    """
    if not REQUESTS_AVAILABLE or not is_ollama_available():
        return build_fallback_response(
            "Ollama not available for follow-up questions",
            patient_name=patient.get("name"),
            risk_level=risk_result.get("risk_level") if risk_result else None,
        )

    pkg = build_followup_prompt(question, patient, conversation_history)
    patient_name = patient.get("name", "Unknown Patient")
    risk_level_val = risk_result.get("risk_level") if risk_result else None

    t0 = time.monotonic()
    try:
        raw = _call_ollama(pkg.system_prompt, pkg.user_prompt, stream=False)
        latency_ms = (time.monotonic() - t0) * 1000
        content = raw.get("message", {}).get("content", "") or raw.get("response", "")

        return AIResponse(
            content=content,
            source="offline_llama3",
            model=OLLAMA_MODEL,
            prompt_tokens=raw.get("prompt_eval_count", 0),
            output_tokens=raw.get("eval_count", 0),
            latency_ms=latency_ms,
            success=True,
            sections=parse_sections(content),
        )
    except Exception as exc:
        return build_fallback_response(str(exc), patient_name, risk_level_val)


def get_model_info() -> Dict[str, Any]:
    """
    ADDED: Return model configuration info for ui/system_status.py.
    Mirrors online_ai.get_model_info() structure.
    """
    return {
        "provider":      "Ollama (Local)",
        "model":         OLLAMA_MODEL,
        "base_url":      OLLAMA_BASE_URL,
        "timeout_s":     OLLAMA_TIMEOUT,
        "max_tokens":    OLLAMA_MAX_TOKENS,
        "temperature":   OLLAMA_TEMPERATURE,
        "privacy":       "100% local — no data leaves device",
        "available":     is_ollama_available(),
    }


def warmup_model() -> bool:
    """
    ADDED: Send a minimal prompt to warm up the Ollama model cache.
    Call this at app startup to prevent first-patient delay.

    Returns True if warmup succeeded, False otherwise.
    """
    if not is_ollama_available():
        return False

    try:
        logger.info("Warming up Ollama model '%s'...", OLLAMA_MODEL)
        t0 = time.monotonic()
        _call_ollama(
            system_prompt="You are a helpful assistant.",
            user_prompt="Reply with exactly one word: Ready",
            stream=False,
        )
        elapsed = (time.monotonic() - t0) * 1000
        logger.info("Ollama warmup complete in %.0f ms", elapsed)
        return True
    except Exception as exc:
        logger.warning("Ollama warmup failed: %s", exc)
        return False