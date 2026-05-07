"""
ai/offline_ai.py
────────────────
Offline AI backend for AuraEcho+ using Ollama + Llama 3.

Responsibility:
    Provide AI-powered cardiac diagnosis when the system is OFFLINE
    or when privacy mode is enabled. All inference runs 100% locally —
    no patient data ever leaves the device.

How it works:
    1. AuraEcho+ detects offline mode (core/mode_detector.py)
    2. offline_ai.py is called with a PromptPackage
    3. It sends the prompt to Ollama's local HTTP API
       (default: http://localhost:11434)
    4. Ollama runs Llama 3 locally and streams the response back
    5. The response is parsed and returned as an AIResponse object

Requirements:
    - Ollama installed: https://ollama.ai
    - Model pulled: `ollama pull llama3`
    - Ollama service running: `ollama serve`

Public API:
    analyze_patient(patient, risk_result, similar_cases) → AIResponse
    is_ollama_available()                                → bool
    get_ollama_status()                                  → dict
    stream_analysis(prompt_package)                      → Generator[str]
"""

import json
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
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT,
    OLLAMA_MAX_TOKENS,
    OLLAMA_TEMPERATURE,
)
from utils.helpers import get_logger

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
    content       : str   — the full AI-generated text
    source        : str   — "offline_llama3" | "online_groq" | "online_openai"
    model         : str   — exact model identifier used
    prompt_tokens : int   — input tokens consumed
    output_tokens : int   — output tokens generated
    latency_ms    : float — wall-clock time for the LLM call
    success       : bool  — False if an error occurred
    error         : str   — error message if success=False
    sections      : dict  — parsed response sections (Assessment, Risk, etc.)
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
        return not bool(self.content.strip())


# ─────────────────────────────────────────────
# Response parser
# ─────────────────────────────────────────────

def _parse_sections(text: str) -> Dict[str, str]:
    """
    Parse the structured AI response into labelled sections.

    Looks for markdown headers like:
        ## 🔍 Clinical Assessment
        ## ⚠️ Key Risk Indicators
        etc.

    Returns a dict: { "Clinical Assessment": "...", "Key Risk Indicators": "..." }
    """
    sections: Dict[str, str] = {}
    current_header = "Introduction"
    current_lines: List[str] = []

    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("## "):
            # Save previous section
            if current_lines:
                sections[current_header] = "\n".join(current_lines).strip()
            # Start new section — strip emoji and ##
            current_header = stripped[3:].strip()
            # Remove leading emoji characters (Unicode ranges)
            clean = "".join(c for c in current_header if c.isalpha() or c in " _-").strip()
            current_header = clean if clean else current_header
            current_lines = []
        else:
            current_lines.append(line)

    # Save last section
    if current_lines:
        sections[current_header] = "\n".join(current_lines).strip()

    return sections


# ─────────────────────────────────────────────
# Ollama connectivity helpers
# ─────────────────────────────────────────────

def is_ollama_available() -> bool:
    """
    Check if Ollama service is running and the target model is loaded.

    Returns
    -------
    True if Ollama is reachable, False otherwise.
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

    Returns
    -------
    dict:
        available     : bool
        base_url      : str
        model         : str
        model_loaded  : bool
        loaded_models : list[str]
        error         : str
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
                "available": False, "base_url": OLLAMA_BASE_URL,
                "model": OLLAMA_MODEL, "model_loaded": False,
                "loaded_models": [], "error": f"HTTP {resp.status_code}",
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
            "available": False, "base_url": OLLAMA_BASE_URL,
            "model": OLLAMA_MODEL, "model_loaded": False,
            "loaded_models": [], "error": str(exc),
        }


# ─────────────────────────────────────────────
# Core Ollama call
# ─────────────────────────────────────────────

def _call_ollama(
    system_prompt: str,
    user_prompt:   str,
    stream:        bool = False,
) -> Dict[str, Any]:
    """
    Make the HTTP POST to Ollama's /api/chat endpoint.

    Parameters
    ----------
    system_prompt : str
    user_prompt   : str
    stream        : bool  — if True, returns streaming response

    Returns
    -------
    dict from Ollama's JSON response
    """
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system",  "content": system_prompt},
            {"role": "user",    "content": user_prompt},
        ],
        "stream": stream,
        "options": {
            "temperature":   OLLAMA_TEMPERATURE,
            "num_predict":   OLLAMA_MAX_TOKENS,
            "top_p":         0.9,
            "repeat_penalty": 1.1,
        },
    }

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json=payload,
        timeout=OLLAMA_TIMEOUT,
        stream=stream,
    )
    response.raise_for_status()

    if stream:
        return response     # return raw response object for streaming

    return response.json()


def _fallback_response(error: str) -> AIResponse:
    """
    Return a safe fallback response when Ollama is unavailable.
    The fallback gives basic clinical guidance based on rule-based logic.
    """
    fallback_content = (
        "## 🔍 Clinical Assessment\n"
        "AI analysis is temporarily unavailable (offline mode — Ollama not running). "
        "The risk score from the automated model is still valid and should guide clinical decisions.\n\n"
        "## ⚠️ Key Risk Indicators\n"
        "• Please refer to the Risk Score panel for automated risk assessment\n"
        "• Review the Similar Cases panel for historical context\n"
        "• Consult standard ACC/AHA cardiac risk guidelines\n\n"
        "## 💊 Treatment Recommendations\n"
        "• **Immediate**: Review automated risk level and act accordingly\n"
        "• **Short-term**: Schedule cardiology consultation if Medium or High risk\n"
        "• **Long-term**: Lifestyle modifications + medication review with physician\n\n"
        "## 🏥 Referral & Follow-up\n"
        "• High Risk: Immediate cardiology referral\n"
        "• Medium Risk: Cardiology within 1-2 weeks\n"
        "• Low Risk: Primary care follow-up in 3-6 months\n\n"
        "## ℹ️ System Note\n"
        f"Offline AI unavailable: {error}\n"
        "To enable AI analysis: ensure Ollama is running (`ollama serve`) "
        f"and the model is pulled (`ollama pull {OLLAMA_MODEL}`)."
    )

    return AIResponse(
        content=fallback_content,
        source="offline_fallback",
        model="rule_based",
        success=False,
        error=error,
        sections=_parse_sections(fallback_content),
    )


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

    Parameters
    ----------
    patient       : raw patient dict (from UI form)
    risk_result   : RiskResult.to_dict() — from risk_model.py
    similar_cases : list of SimilarCase.to_dict() — from similarity.py

    Returns
    -------
    AIResponse — always returns (never raises), falls back gracefully
    """
    if not REQUESTS_AVAILABLE:
        return _fallback_response("requests library not installed")

    if not is_ollama_available():
        return _fallback_response(
            f"Ollama not running or model '{OLLAMA_MODEL}' not loaded"
        )

    # Build prompt
    pkg = build_diagnosis_prompt(patient, risk_result, similar_cases)

    logger.info(
        "Calling Ollama/%s | prompt_tokens~%d | patient=%s",
        OLLAMA_MODEL, pkg.token_estimate, patient.get("name", "?"),
    )

    t0 = time.monotonic()
    try:
        raw = _call_ollama(pkg.system_prompt, pkg.user_prompt, stream=False)
        latency_ms = (time.monotonic() - t0) * 1000

        # Extract content
        content = (
            raw.get("message", {}).get("content", "")
            or raw.get("response", "")
        )

        # Token counts from Ollama
        prompt_tokens = raw.get("prompt_eval_count", pkg.token_estimate)
        output_tokens = raw.get("eval_count", 0)

        sections = _parse_sections(content)

        logger.info(
            "Ollama response: %d chars | %d output tokens | %.0f ms",
            len(content), output_tokens, latency_ms,
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

    except Exception as exc:
        latency_ms = (time.monotonic() - t0) * 1000
        logger.error("Ollama call failed after %.0f ms: %s", latency_ms, exc)
        return _fallback_response(str(exc))


def stream_analysis(
    patient:       Dict[str, Any],
    risk_result:   Optional[Dict[str, Any]] = None,
    similar_cases: Optional[List[Dict[str, Any]]] = None,
) -> Generator[str, None, None]:
    """
    Stream the AI response token-by-token (for Streamlit st.write_stream).

    Usage in Streamlit:
        with st.chat_message("assistant"):
            st.write_stream(stream_analysis(patient, risk_result, similar_cases))

    Yields
    ------
    str — one token/chunk at a time
    """
    if not REQUESTS_AVAILABLE or not is_ollama_available():
        yield "⚠️ Offline AI unavailable. Please ensure Ollama is running."
        return

    pkg = build_diagnosis_prompt(patient, risk_result, similar_cases)

    try:
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

    except Exception as exc:
        logger.error("Streaming failed: %s", exc)
        yield f"\n\n⚠️ Stream interrupted: {exc}"


def analyze_with_question(
    question:             str,
    patient:              Dict[str, Any],
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> AIResponse:
    """
    Answer a follow-up question about a patient (conversational mode).

    Parameters
    ----------
    question             : the doctor/nurse's follow-up question
    patient              : current patient context dict
    conversation_history : prior turns [{"role": "user"|"assistant", "content": str}]

    Returns
    -------
    AIResponse
    """
    if not REQUESTS_AVAILABLE or not is_ollama_available():
        return _fallback_response("Ollama not available for follow-up questions")

    pkg = build_followup_prompt(question, patient, conversation_history)

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
            sections=_parse_sections(content),
        )
    except Exception as exc:
        return _fallback_response(str(exc))