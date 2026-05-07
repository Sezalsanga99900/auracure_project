"""
ai/online_ai.py
───────────────
Online AI backend for AuraEcho+ using Groq (primary) + OpenAI (fallback).

Responsibility:
    Provide fast, high-quality AI-powered cardiac diagnosis when the
    system has internet access. Uses Groq's ultra-fast inference API
    as the primary backend, with OpenAI as a fallback.

Why Groq?
    Groq hardware achieves 300–500 tokens/second on Llama3 — roughly
    10× faster than standard OpenAI API. For a clinical tool where
    doctors are waiting, speed matters.

Fallback chain:
    Groq (Llama3-70b) → OpenAI (GPT-4o-mini) → offline_ai → rule-based

Public API:
    analyze_patient(patient, risk_result, similar_cases) → AIResponse
    is_groq_available()    → bool
    is_openai_available()  → bool
    get_api_status()       → dict
    stream_analysis(...)   → Generator[str]
"""

import os
import time
from typing import Any, Dict, Generator, List, Optional

from ai.prompt_builder import (
    PromptPackage,
    build_diagnosis_prompt,
    build_followup_prompt,
    build_summary_prompt,
)
from ai.offline_ai import AIResponse, _parse_sections, _fallback_response
from utils.constants import (
    GROQ_MODEL,
    GROQ_MAX_TOKENS,
    GROQ_TEMPERATURE,
    OPENAI_MODEL,
    OPENAI_MAX_TOKENS,
    OPENAI_TEMPERATURE,
    API_TIMEOUT,
)
from utils.helpers import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────
# Optional imports — gracefully handle missing libs
# ─────────────────────────────────────────────
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logger.warning("groq package not installed — pip install groq")

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("openai package not installed — pip install openai")

# ─────────────────────────────────────────────
# Client singletons (created once, reused)
# ─────────────────────────────────────────────
_groq_client:   Optional[Any] = None
_openai_client: Optional[Any] = None


def _get_groq_client():
    """Return a cached Groq client, creating it if needed."""
    global _groq_client
    if _groq_client is None and GROQ_AVAILABLE:
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            logger.error("GROQ_API_KEY not set in environment")
            return None
        try:
            _groq_client = Groq(api_key=api_key)
            logger.info("Groq client initialised (model=%s)", GROQ_MODEL)
        except Exception as exc:
            logger.error("Failed to init Groq client: %s", exc)
            return None
    return _groq_client


def _get_openai_client():
    """Return a cached OpenAI client, creating it if needed."""
    global _openai_client
    if _openai_client is None and OPENAI_AVAILABLE:
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set — OpenAI fallback unavailable")
            return None
        try:
            _openai_client = OpenAI(api_key=api_key)
            logger.info("OpenAI client initialised (model=%s)", OPENAI_MODEL)
        except Exception as exc:
            logger.error("Failed to init OpenAI client: %s", exc)
            return None
    return _openai_client


# ─────────────────────────────────────────────
# Availability checks
# ─────────────────────────────────────────────

def is_groq_available() -> bool:
    """
    Return True if Groq is installed and GROQ_API_KEY is set.
    Does NOT make a live API call — just checks configuration.
    """
    return GROQ_AVAILABLE and bool(os.getenv("GROQ_API_KEY", ""))


def is_openai_available() -> bool:
    """
    Return True if OpenAI is installed and OPENAI_API_KEY is set.
    """
    return OPENAI_AVAILABLE and bool(os.getenv("OPENAI_API_KEY", ""))


def get_api_status() -> Dict[str, Any]:
    """
    Return a full status dict for the system status panel.

    Returns
    -------
    dict:
        groq_ready       : bool
        openai_ready     : bool
        active_backend   : str   "groq" | "openai" | "none"
        groq_model       : str
        openai_model     : str
        groq_key_set     : bool
        openai_key_set   : bool
    """
    groq_ready   = is_groq_available()
    openai_ready = is_openai_available()

    if groq_ready:
        active = "groq"
    elif openai_ready:
        active = "openai"
    else:
        active = "none"

    return {
        "groq_ready":     groq_ready,
        "openai_ready":   openai_ready,
        "active_backend": active,
        "groq_model":     GROQ_MODEL,
        "openai_model":   OPENAI_MODEL,
        "groq_key_set":   bool(os.getenv("GROQ_API_KEY")),
        "openai_key_set": bool(os.getenv("OPENAI_API_KEY")),
    }


# ─────────────────────────────────────────────
# Groq call
# ─────────────────────────────────────────────

def _call_groq(
    system_prompt: str,
    user_prompt:   str,
    stream:        bool = False,
) -> Any:
    """
    Make a Groq API chat completion call.

    Returns
    -------
    Groq ChatCompletion object (or streaming object if stream=True)
    """
    client = _get_groq_client()
    if client is None:
        raise RuntimeError("Groq client not available")

    return client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system",  "content": system_prompt},
            {"role": "user",    "content": user_prompt},
        ],
        max_tokens=GROQ_MAX_TOKENS,
        temperature=GROQ_TEMPERATURE,
        stream=stream,
    )


def _call_openai(
    system_prompt: str,
    user_prompt:   str,
    stream:        bool = False,
) -> Any:
    """
    Make an OpenAI API chat completion call.

    Returns
    -------
    OpenAI ChatCompletion object (or streaming object if stream=True)
    """
    client = _get_openai_client()
    if client is None:
        raise RuntimeError("OpenAI client not available")

    return client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system",  "content": system_prompt},
            {"role": "user",    "content": user_prompt},
        ],
        max_tokens=OPENAI_MAX_TOKENS,
        temperature=OPENAI_TEMPERATURE,
        stream=stream,
    )


# ─────────────────────────────────────────────
# Response builder
# ─────────────────────────────────────────────

def _build_response_from_completion(
    completion: Any,
    source:     str,
    model:      str,
    latency_ms: float,
) -> AIResponse:
    """
    Convert a Groq/OpenAI completion object → AIResponse.
    Both SDKs use identical response structure.
    """
    choice  = completion.choices[0]
    content = choice.message.content or ""
    usage   = completion.usage

    sections = _parse_sections(content)

    return AIResponse(
        content=content,
        source=source,
        model=model,
        prompt_tokens=usage.prompt_tokens if usage else 0,
        output_tokens=usage.completion_tokens if usage else 0,
        latency_ms=latency_ms,
        success=True,
        sections=sections,
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
    Run an online AI cardiac analysis using Groq → OpenAI → fallback.

    Parameters
    ----------
    patient       : raw patient dict (from UI form)
    risk_result   : RiskResult.to_dict()
    similar_cases : list of SimilarCase.to_dict()

    Returns
    -------
    AIResponse — never raises, always returns a result
    """
    pkg = build_diagnosis_prompt(patient, risk_result, similar_cases)

    # ── Try Groq first ─────────────────────────────────────────────
    if is_groq_available():
        logger.info(
            "Calling Groq/%s | ~%d tokens | patient=%s",
            GROQ_MODEL, pkg.token_estimate, patient.get("name", "?"),
        )
        t0 = time.monotonic()
        try:
            completion = _call_groq(pkg.system_prompt, pkg.user_prompt)
            latency_ms = (time.monotonic() - t0) * 1000
            logger.info("Groq responded in %.0f ms", latency_ms)
            return _build_response_from_completion(
                completion, "online_groq", GROQ_MODEL, latency_ms
            )
        except Exception as exc:
            logger.warning("Groq call failed: %s — trying OpenAI", exc)

    # ── Try OpenAI fallback ─────────────────────────────────────────
    if is_openai_available():
        logger.info("Calling OpenAI/%s | patient=%s", OPENAI_MODEL, patient.get("name", "?"))
        t0 = time.monotonic()
        try:
            completion = _call_openai(pkg.system_prompt, pkg.user_prompt)
            latency_ms = (time.monotonic() - t0) * 1000
            logger.info("OpenAI responded in %.0f ms", latency_ms)
            return _build_response_from_completion(
                completion, "online_openai", OPENAI_MODEL, latency_ms
            )
        except Exception as exc:
            logger.warning("OpenAI call failed: %s — using fallback", exc)

    # ── Both online backends failed ─────────────────────────────────
    logger.error("All online AI backends failed — returning rule-based fallback")
    return _fallback_response("All online AI backends unavailable")


def stream_analysis(
    patient:       Dict[str, Any],
    risk_result:   Optional[Dict[str, Any]] = None,
    similar_cases: Optional[List[Dict[str, Any]]] = None,
) -> Generator[str, None, None]:
    """
    Stream the AI response token-by-token (for Streamlit st.write_stream).

    Tries Groq streaming first, then OpenAI streaming.

    Usage in Streamlit:
        with st.chat_message("assistant"):
            st.write_stream(stream_analysis(patient, risk_result, similar_cases))

    Yields
    ------
    str — one token/chunk at a time
    """
    pkg = build_diagnosis_prompt(patient, risk_result, similar_cases)

    # Try Groq streaming
    if is_groq_available():
        try:
            stream = _call_groq(pkg.system_prompt, pkg.user_prompt, stream=True)
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content
            return
        except Exception as exc:
            logger.warning("Groq stream failed: %s", exc)
            yield f"\n⚠️ Groq stream interrupted ({exc}). Trying OpenAI...\n"

    # Try OpenAI streaming
    if is_openai_available():
        try:
            stream = _call_openai(pkg.system_prompt, pkg.user_prompt, stream=True)
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content
            return
        except Exception as exc:
            logger.warning("OpenAI stream failed: %s", exc)

    yield "\n⚠️ All online AI backends unavailable. Please check API keys."


def analyze_with_question(
    question:             str,
    patient:              Dict[str, Any],
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> AIResponse:
    """
    Answer a follow-up question using the online AI backend.

    Parameters
    ----------
    question             : doctor/nurse's follow-up question
    patient              : current patient context
    conversation_history : prior conversation turns

    Returns
    -------
    AIResponse
    """
    pkg = build_followup_prompt(question, patient, conversation_history)

    # Groq first
    if is_groq_available():
        t0 = time.monotonic()
        try:
            completion = _call_groq(pkg.system_prompt, pkg.user_prompt)
            latency_ms = (time.monotonic() - t0) * 1000
            return _build_response_from_completion(
                completion, "online_groq", GROQ_MODEL, latency_ms
            )
        except Exception as exc:
            logger.warning("Groq followup failed: %s", exc)

    # OpenAI fallback
    if is_openai_available():
        t0 = time.monotonic()
        try:
            completion = _call_openai(pkg.system_prompt, pkg.user_prompt)
            latency_ms = (time.monotonic() - t0) * 1000
            return _build_response_from_completion(
                completion, "online_openai", OPENAI_MODEL, latency_ms
            )
        except Exception as exc:
            logger.warning("OpenAI followup failed: %s", exc)

    return _fallback_response("Online AI unavailable for follow-up")


def get_short_summary(
    patient:     Dict[str, Any],
    risk_result: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Get a 3-4 sentence plain-language patient summary.
    Returns just the content string (not a full AIResponse).
    """
    pkg = build_summary_prompt(patient, risk_result)

    if is_groq_available():
        try:
            t0 = time.monotonic()
            completion = _call_groq(pkg.system_prompt, pkg.user_prompt)
            latency_ms = (time.monotonic() - t0) * 1000
            content = completion.choices[0].message.content or ""
            logger.info("Summary generated in %.0f ms via Groq", latency_ms)
            return content
        except Exception as exc:
            logger.warning("Groq summary failed: %s", exc)

    if is_openai_available():
        try:
            completion = _call_openai(pkg.system_prompt, pkg.user_prompt)
            return completion.choices[0].message.content or ""
        except Exception as exc:
            logger.warning("OpenAI summary failed: %s", exc)

    return (
        f"Patient risk level: {risk_result.get('risk_level', 'Unknown') if risk_result else 'Unknown'}. "
        "Full AI summary unavailable — please check API configuration."
    )