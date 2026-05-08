# =============================================================================
# ai/online_ai.py
# AuraEcho+ — Online AI Backend (Groq primary + OpenAI fallback)
#
# Responsibility:
#     Provide fast, high-quality AI-powered cardiac diagnosis when the
#     system has internet access. Uses Groq's ultra-fast inference API
#     as the primary backend, with OpenAI as a fallback.
#
# Fallback chain:
#     Groq (Llama3-70b) → OpenAI (GPT-4o-mini) → build_fallback_response
#
# Public API:
#     analyze_patient(patient, risk_result, similar_cases) → AIResponse
#     is_groq_available()    → bool
#     is_openai_available()  → bool
#     get_api_status()       → dict
#     stream_analysis(...)   → Generator[str]
#     analyze_with_question(...) → AIResponse
#     get_short_summary(...) → str
#     test_api_connection(provider) → dict
#     get_model_info()       → dict
# =============================================================================

import os
import time
import json
import functools
from typing import Any, Dict, Generator, List, Optional

# ─────────────────────────────────────────────
# Optional imports — gracefully handle missing libs
# ─────────────────────────────────────────────
try:
    from groq import Groq, APIError as GroqAPIError
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    GroqAPIError = Exception

try:
    from openai import OpenAI, APIError as OpenAIAPIError
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAIAPIError = Exception

# ─────────────────────────────────────────────
# Shared components
# FIXED: Imports from ai.ai_response instead of private functions in offline_ai.
#        This removes tight coupling and encapsulation violations.
# ─────────────────────────────────────────────
from ai.ai_response import (
    AIResponse,
    parse_sections,
    build_fallback_response,
    safe_error_msg,
)
from ai.prompt_builder import (
    PromptPackage,
    build_diagnosis_prompt,
    build_followup_prompt,
    build_summary_prompt,
)
from utils.constants import (
    APP_NAME,
    GROQ_MODEL,
    GROQ_MAX_TOKENS,
    GROQ_TEMPERATURE,
    OPENAI_MODEL,
    OPENAI_MAX_TOKENS,
    OPENAI_TEMPERATURE,
    API_TIMEOUT,
    RISK_LEVELS,
    RISK_LABELS,
    RISK_COLORS,
    RISK_ICONS,
)
from utils.helpers import get_logger, normalize_risk_level

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# Client singletons (created once, reused)
# ─────────────────────────────────────────────
_groq_client:        Optional[Any] = None
_openai_client:      Optional[Any] = None
_groq_init_failed:   bool = False
_openai_init_failed: bool = False


def _get_groq_client() -> Optional[Any]:
    """Return a cached Groq client, creating it if needed."""
    global _groq_client, _groq_init_failed

    if _groq_init_failed:
        return None

    if _groq_client is None and GROQ_AVAILABLE:
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            logger.error("GROQ_API_KEY not set in environment")
            _groq_init_failed = True
            return None
        try:
            _groq_client = Groq(api_key=api_key)
            logger.info("Groq client initialised (model=%s)", GROQ_MODEL)
        except Exception as exc:
            logger.error("Failed to init Groq client: %s", exc)
            _groq_init_failed = True
            return None
    return _groq_client


def _get_openai_client() -> Optional[Any]:
    """Return a cached OpenAI client, creating it if needed."""
    global _openai_client, _openai_init_failed

    if _openai_init_failed:
        return None

    if _openai_client is None and OPENAI_AVAILABLE:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            logger.warning("OPENAI_API_KEY not set — OpenAI fallback unavailable")
            _openai_init_failed = True
            return None
        try:
            _openai_client = OpenAI(api_key=api_key)
            logger.info("OpenAI client initialised (model=%s)", OPENAI_MODEL)
        except Exception as exc:
            logger.error("Failed to init OpenAI client: %s", exc)
            _openai_init_failed = True
            return None
    return _openai_client


# ─────────────────────────────────────────────
# Availability checks
# FIXED: Added API key format validation.
# ─────────────────────────────────────────────

def is_groq_available() -> bool:
    """
    Return True if Groq is installed and GROQ_API_KEY is valid.
    FIXED: Checks key format (starts with 'gsk_', length > 20).
    """
    if not GROQ_AVAILABLE:
        return False
    key = os.getenv("GROQ_API_KEY", "").strip()
    return bool(key) and len(key) > 20 and key.startswith("gsk_")


def is_openai_available() -> bool:
    """
    Return True if OpenAI is installed and OPENAI_API_KEY is valid.
    FIXED: Checks key format (starts with 'sk-', length > 20).
    """
    if not OPENAI_AVAILABLE:
        return False
    key = os.getenv("OPENAI_API_KEY", "").strip()
    return key.startswith("sk-") and len(key) > 20


def get_api_status() -> Dict[str, Any]:
    """
    Return a full status dict for the system status panel.
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
# Retry decorator
# ADDED: Retries transient errors with exponential backoff.
# ─────────────────────────────────────────────

def _with_retry(max_retries: int = 2, backoff: float = 1.0):
    """
    Decorator: retry on transient API errors with exponential backoff.
    Retries on: timeout, rate limit, 500/502/503.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    err_str = str(exc).lower()
                    is_transient = any(
                        x in err_str for x in
                        ["timeout", "rate limit", "503", "502", "500"]
                    )
                    if is_transient and attempt < max_retries:
                        wait = backoff * (2 ** (attempt - 1))
                        logger.warning(
                            "Transient error attempt %d/%d — retry in %.1fs: %s",
                            attempt, max_retries, wait, exc,
                        )
                        time.sleep(wait)
                    else:
                        raise
            raise last_exc
        return wrapper
    return decorator


# ─────────────────────────────────────────────
# API calls
# FIXED: Added @_with_retry decorator.
# ─────────────────────────────────────────────

@_with_retry(max_retries=2, backoff=1.0)
def _call_groq(
    system_prompt: str,
    user_prompt:   str,
    stream:        bool = False,
) -> Any:
    """Make a Groq API chat completion call with retry logic."""
    client = _get_groq_client()
    if client is None:
        raise RuntimeError("Groq client not available")

    return client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=GROQ_MAX_TOKENS,
        temperature=GROQ_TEMPERATURE,
        stream=stream,
        timeout=API_TIMEOUT,
    )


@_with_retry(max_retries=2, backoff=1.0)
def _call_openai(
    system_prompt: str,
    user_prompt:   str,
    stream:        bool = False,
) -> Any:
    """Make an OpenAI API chat completion call with retry logic."""
    client = _get_openai_client()
    if client is None:
        raise RuntimeError("OpenAI client not available")

    return client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=OPENAI_MAX_TOKENS,
        temperature=OPENAI_TEMPERATURE,
        stream=stream,
        timeout=API_TIMEOUT,
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
    """Convert a Groq/OpenAI completion object → AIResponse."""
    choice  = completion.choices[0]
    content = choice.message.content or ""
    usage   = completion.usage

    sections = parse_sections(content)

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
    Returns AIResponse — never raises, always returns a result.
    """
    pkg = build_diagnosis_prompt(patient, risk_result, similar_cases)

    patient_name   = patient.get("name", "Unknown Patient")
    risk_level_val = risk_result.get("risk_level") if risk_result else None

    # ── Try Groq first ─────────────────────────────────────────────
    if is_groq_available():
        logger.info(
            "Calling Groq/%s | ~%d tokens | patient=%s | risk=%s",
            GROQ_MODEL, pkg.token_estimate, patient_name, risk_level_val,
        )
        t0 = time.monotonic()
        try:
            completion = _call_groq(pkg.system_prompt, pkg.user_prompt)
            latency_ms = (time.monotonic() - t0) * 1000
            logger.info(
                "Groq responded in %.0f ms (patient=%s)", latency_ms, patient_name
            )
            return _build_response_from_completion(
                completion, "online_groq", GROQ_MODEL, latency_ms
            )
        except GroqAPIError as exc:
            logger.warning(
                "Groq APIError (patient=%s): %s — trying OpenAI", patient_name, exc
            )
        except Exception as exc:
            logger.warning(
                "Groq call failed (patient=%s): %s — trying OpenAI", patient_name, exc
            )

    # ── Try OpenAI fallback ─────────────────────────────────────────
    if is_openai_available():
        logger.info(
            "Calling OpenAI/%s | patient=%s | risk=%s",
            OPENAI_MODEL, patient_name, risk_level_val,
        )
        t0 = time.monotonic()
        try:
            completion = _call_openai(pkg.system_prompt, pkg.user_prompt)
            latency_ms = (time.monotonic() - t0) * 1000
            logger.info(
                "OpenAI responded in %.0f ms (patient=%s)", latency_ms, patient_name
            )
            return _build_response_from_completion(
                completion, "online_openai", OPENAI_MODEL, latency_ms
            )
        except OpenAIAPIError as exc:
            logger.warning(
                "OpenAI APIError (patient=%s): %s — using fallback", patient_name, exc
            )
        except Exception as exc:
            logger.warning(
                "OpenAI call failed (patient=%s): %s — using fallback", patient_name, exc
            )

    # ── Both online backends failed ─────────────────────────────────
    logger.error(
        "All online AI backends failed (patient=%s) — returning fallback", patient_name
    )
    return build_fallback_response(
        "All online AI backends unavailable",
        patient_name=patient_name,
        risk_level=risk_level_val,
    )


def stream_analysis(
    patient:       Dict[str, Any],
    risk_result:   Optional[Dict[str, Any]] = None,
    similar_cases: Optional[List[Dict[str, Any]]] = None,
) -> Generator[str, None, None]:
    """
    Stream the AI response token-by-token.
    Tries Groq streaming first, then OpenAI streaming.
    FIXED: Sanitizes error messages for clinical users using safe_error_msg().
    """
    pkg = build_diagnosis_prompt(patient, risk_result, similar_cases)
    patient_name   = patient.get("name", "Unknown Patient")
    risk_level_val = risk_result.get("risk_level") if risk_result else None

    # Try Groq streaming
    if is_groq_available():
        try:
            stream = _call_groq(pkg.system_prompt, pkg.user_prompt, stream=True)
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content
            return
        except GroqAPIError as exc:
            logger.warning("Groq stream APIError: %s — trying OpenAI", exc)
            yield safe_error_msg(exc, "Groq")
        except Exception as exc:
            logger.warning("Groq stream failed: %s — trying OpenAI", exc)
            yield safe_error_msg(exc, "Groq")

    # Try OpenAI streaming
    if is_openai_available():
        try:
            stream = _call_openai(pkg.system_prompt, pkg.user_prompt, stream=True)
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content
            return
        except OpenAIAPIError as exc:
            logger.warning("OpenAI stream APIError: %s", exc)
            yield safe_error_msg(exc, "OpenAI")
        except Exception as exc:
            logger.warning("OpenAI stream failed: %s", exc)
            yield safe_error_msg(exc, "OpenAI")

    yield "\n⚠️ All online AI backends unavailable. Please check API keys and network."


def analyze_with_question(
    question:             str,
    patient:              Dict[str, Any],
    conversation_history: Optional[List[Dict[str, str]]] = None,
    risk_result:          Optional[Dict[str, Any]] = None,   # ADDED
) -> AIResponse:
    """
    Answer a follow-up question using the online AI backend.
    ADDED: risk_result parameter — passed to fallback for context.
    """
    pkg = build_followup_prompt(question, patient, conversation_history)
    patient_name   = patient.get("name", "Unknown Patient")
    risk_level_val = risk_result.get("risk_level") if risk_result else None

    # Groq first
    if is_groq_available():
        t0 = time.monotonic()
        try:
            completion = _call_groq(pkg.system_prompt, pkg.user_prompt)
            latency_ms = (time.monotonic() - t0) * 1000
            return _build_response_from_completion(
                completion, "online_groq", GROQ_MODEL, latency_ms
            )
        except GroqAPIError as exc:
            logger.warning("Groq followup APIError: %s", exc)
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
        except OpenAIAPIError as exc:
            logger.warning("OpenAI followup APIError: %s", exc)
        except Exception as exc:
            logger.warning("OpenAI followup failed: %s", exc)

    return build_fallback_response(
        "Online AI unavailable for follow-up",
        patient_name=patient_name,
        risk_level=risk_level_val,
    )


def get_short_summary(
    patient:     Dict[str, Any],
    risk_result: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Get a 3-4 sentence plain-language patient summary.
    FIXED: Uses normalize_risk_level() to handle key/label confusion.
    """
    pkg = build_summary_prompt(patient, risk_result)
    patient_name = patient.get("name", "Unknown Patient")

    # FIXED: Normalize risk level
    raw_level    = risk_result.get("risk_level") if risk_result else None
    level_key    = normalize_risk_level(raw_level) or "MEDIUM"
    risk_label   = RISK_LABELS.get(level_key, "Unknown")
    risk_icon    = RISK_ICONS.get(level_key, "❔")

    # Try Groq
    if is_groq_available():
        try:
            t0 = time.monotonic()
            completion = _call_groq(pkg.system_prompt, pkg.user_prompt)
            latency_ms = (time.monotonic() - t0) * 1000
            content = completion.choices[0].message.content or ""
            logger.info(
                "Summary generated in %.0f ms via Groq (patient=%s)",
                latency_ms, patient_name,
            )
            return content
        except GroqAPIError as exc:
            logger.warning("Groq summary APIError: %s", exc)
        except Exception as exc:
            logger.warning("Groq summary failed: %s", exc)

    # Try OpenAI
    if is_openai_available():
        try:
            completion = _call_openai(pkg.system_prompt, pkg.user_prompt)
            content = completion.choices[0].message.content or ""
            logger.info("Summary generated via OpenAI (patient=%s)", patient_name)
            return content
        except OpenAIAPIError as exc:
            logger.warning("OpenAI summary APIError: %s", exc)
        except Exception as exc:
            logger.warning("OpenAI summary failed: %s", exc)

    # Final fallback
    return (
        f"{APP_NAME}: {patient_name} is classified as {risk_icon} {risk_label} "
        f"({level_key}). Full AI summary unavailable — please check API configuration "
        "(GROQ_API_KEY / OPENAI_API_KEY) and network connectivity."
    )


def test_api_connection(provider: str = "groq") -> Dict[str, Any]:
    """
    ADDED: Make a minimal live API call to verify credentials work.
    Used by ui/system_status.py for the 'Test Connection' button.

    Parameters
    ----------
    provider : "groq" | "openai"

    Returns
    -------
    dict: success, latency_ms, error, model
    """
    test_system = "You are a helpful assistant."
    test_user   = "Reply with exactly one word: OK"

    t0 = time.monotonic()
    try:
        if provider == "groq" and is_groq_available():
            completion = _call_groq(test_system, test_user)
            latency_ms = (time.monotonic() - t0) * 1000
            return {
                "success":    True,
                "provider":   "groq",
                "model":      GROQ_MODEL,
                "latency_ms": round(latency_ms, 1),
                "error":      "",
            }
        elif provider == "openai" and is_openai_available():
            completion = _call_openai(test_system, test_user)
            latency_ms = (time.monotonic() - t0) * 1000
            return {
                "success":    True,
                "provider":   "openai",
                "model":      OPENAI_MODEL,
                "latency_ms": round(latency_ms, 1),
                "error":      "",
            }
        else:
            return {
                "success":  False,
                "provider": provider,
                "model":    "",
                "latency_ms": 0.0,
                "error":    f"{provider} not available or API key not set",
            }
    except Exception as exc:
        latency_ms = (time.monotonic() - t0) * 1000
        return {
            "success":    False,
            "provider":   provider,
            "model":      GROQ_MODEL if provider == "groq" else OPENAI_MODEL,
            "latency_ms": round(latency_ms, 1),
            "error":      str(exc),
        }


def get_model_info() -> Dict[str, Any]:
    """
    ADDED: Return model configuration for ui/system_status.py.
    Mirrors offline_ai.get_model_info() structure.
    """
    groq_ready   = is_groq_available()
    openai_ready = is_openai_available()

    return {
        "primary_provider":   "Groq",
        "primary_model":      GROQ_MODEL,
        "fallback_provider":  "OpenAI",
        "fallback_model":     OPENAI_MODEL,
        "active_backend":     (
            "groq" if groq_ready else ("openai" if openai_ready else "none")
        ),
        "groq_available":     groq_ready,
        "openai_available":   openai_ready,
        "max_tokens":         GROQ_MAX_TOKENS,
        "temperature":        GROQ_TEMPERATURE,
        "timeout_s":          API_TIMEOUT,
        "privacy":            "Cloud API — data sent to provider",
    }