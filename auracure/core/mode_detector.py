"""
core/mode_detector.py
─────────────────────
Internet connectivity detector for AuraEcho+.

Responsibility:
    Determine whether the system is online or offline so the rest of
    the application can route to cloud AI / local Ollama and to
    Firebase / SQLite accordingly.

How it works:
    1. Try to open a TCP socket to multiple reliable hosts
       (Google DNS 8.8.8.8, Cloudflare 1.1.1.1, OpenDNS 208.67.222.222).
    2. If ANY connection succeeds within the timeout → ONLINE.
    3. If ALL fail → OFFLINE.
    4. Result is cached for CACHE_TTL seconds so we don't hammer the
       network on every Streamlit re-render.

Public API:
    is_online()           → bool
    get_connection_info() → dict  (status, latency_ms, host, timestamp)
    require_online()      → None  (raises ConnectionError if offline)
"""

import socket
import time
import threading
from datetime import datetime
from typing import Dict, Optional, Tuple

from utils.constants import (
    CONNECTIVITY_CHECK_HOSTS,
    CONNECTIVITY_TIMEOUT,
    CONNECTIVITY_CACHE_TTL,
    MODE_ONLINE,
    MODE_OFFLINE,
)
from utils.helpers import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────
# Module-level cache  (thread-safe via Lock)
# ─────────────────────────────────────────────
_cache_lock = threading.Lock()
_cached_result: Optional[bool] = None
_cached_at: float = 0.0          # epoch seconds
_last_info: Dict = {}


# ─────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────

def _probe_host(host: str, port: int = 53, timeout: float = CONNECTIVITY_TIMEOUT) -> Tuple[bool, float]:
    """
    Open a TCP socket to *host:port*.

    Returns
    -------
    (success: bool, latency_ms: float)
        latency_ms is -1.0 on failure.
    """
    t0 = time.monotonic()
    try:
        with socket.create_connection((host, port), timeout=timeout):
            latency = (time.monotonic() - t0) * 1000  # convert to ms
            return True, round(latency, 2)
    except OSError:
        return False, -1.0


def _check_connectivity() -> Tuple[bool, float, str]:
    """
    Try every host in CONNECTIVITY_CHECK_HOSTS.

    Returns
    -------
    (online: bool, best_latency_ms: float, successful_host: str)
    """
    for host, port in CONNECTIVITY_CHECK_HOSTS:
        success, latency = _probe_host(host, port)
        if success:
            logger.debug("Connectivity OK via %s:%s (%.1f ms)", host, port, latency)
            return True, latency, f"{host}:{port}"

    logger.warning("All connectivity probes failed — switching to OFFLINE mode")
    return False, -1.0, "none"


def _build_info(online: bool, latency: float, host: str) -> Dict:
    """Assemble the connection-info dict returned by get_connection_info()."""
    return {
        "mode":        MODE_ONLINE if online else MODE_OFFLINE,
        "online":      online,
        "latency_ms":  latency,
        "host":        host,
        "checked_at":  datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "cached":      False,           # caller flips this to True when served from cache
    }


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def is_online(force_refresh: bool = False) -> bool:
    """
    Return True if the system has internet access, False otherwise.

    Parameters
    ----------
    force_refresh : bool
        Bypass the cache and re-probe immediately.

    Notes
    -----
    Result is cached for CONNECTIVITY_CACHE_TTL seconds (default 30 s)
    to avoid flooding the network on every Streamlit widget interaction.
    """
    global _cached_result, _cached_at, _last_info

    now = time.monotonic()
    cache_age = now - _cached_at

    with _cache_lock:
        if not force_refresh and _cached_result is not None and cache_age < CONNECTIVITY_CACHE_TTL:
            logger.debug("Connectivity cache hit (age=%.1fs)", cache_age)
            cached_info = dict(_last_info)
            cached_info["cached"] = True
            _last_info = cached_info
            return _cached_result

        # Cache miss or forced refresh — actually probe
        online, latency, host = _check_connectivity()
        _cached_result = online
        _cached_at = now
        _last_info = _build_info(online, latency, host)

        logger.info(
            "Connectivity check: %s | latency=%.1f ms | via=%s",
            "ONLINE" if online else "OFFLINE",
            latency,
            host,
        )
        return online


def get_connection_info(force_refresh: bool = False) -> Dict:
    """
    Return detailed connectivity information.

    Returns
    -------
    dict with keys:
        mode        : "online" | "offline"
        online      : bool
        latency_ms  : float   (-1.0 if offline)
        host        : str     ("none" if offline)
        checked_at  : str     (ISO-8601 UTC timestamp)
        cached      : bool    (True if served from cache)
    """
    # Trigger a check (respects cache internally)
    is_online(force_refresh=force_refresh)

    with _cache_lock:
        return dict(_last_info)


def get_mode_label() -> str:
    """
    Return a human-readable mode string.

    Returns
    -------
    "🟢 Online"  or  "🔴 Offline"
    """
    return "🟢 Online" if is_online() else "🔴 Offline"


def require_online(message: str = "This feature requires an internet connection.") -> None:
    """
    Raise ConnectionError if the system is offline.

    Use this as a guard at the top of any function that MUST have
    internet access (e.g., cloud AI calls, Firebase sync).

    Parameters
    ----------
    message : str
        Human-readable error text surfaced to the user.

    Raises
    ------
    ConnectionError
    """
    if not is_online():
        logger.error("require_online() failed — system is offline")
        raise ConnectionError(message)


def simulate_offline(duration_seconds: float = 10.0) -> None:
    """
    **Development / testing only.**

    Force the cached result to False for *duration_seconds*.
    Useful for testing offline UI paths without pulling a network cable.
    """
    global _cached_result, _cached_at, _last_info

    logger.warning("SIMULATING OFFLINE MODE for %.1f seconds", duration_seconds)

    with _cache_lock:
        _cached_result = False
        # Set cache timestamp so it won't expire for duration_seconds
        _cached_at = time.monotonic() + duration_seconds - CONNECTIVITY_CACHE_TTL
        _last_info = _build_info(False, -1.0, "simulated-offline")


# ─────────────────────────────────────────────
# Quick self-test (python -m core.mode_detector)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Running connectivity check …")
    info = get_connection_info(force_refresh=True)
    for k, v in info.items():
        print(f"  {k:15s}: {v}")
    print(f"\nMode label : {get_mode_label()}")