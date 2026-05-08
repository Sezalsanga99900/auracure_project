# =============================================================================
# core/mode_detector.py
# AuraEcho+ — Internet Connectivity Detector
#
# Responsibility:
#     Determine whether the system is online or offline so the rest of
#     the application can route to cloud AI / local Ollama and to
#     Firebase / SQLite accordingly.
#
# How it works:
#     1. Try to open a TCP socket to multiple reliable hosts.
#     2. If ANY connection succeeds within the timeout → ONLINE.
#     3. If ALL fail → OFFLINE.
#     4. Result is cached for CONNECTIVITY_CACHE_TTL seconds.
#
# Public API:
#     is_online(force_refresh)      → bool
#     get_mode()                    → str  ("online" | "offline")
#     get_connection_info()         → dict
#     get_mode_label()              → str  ("🟢 Online" / "🔴 Offline")
#     require_online()              → None (raises ConnectionError if offline)
#     invalidate_cache()            → None
#     simulate_offline(duration)    → None (dev/testing only)
#     watch_connectivity(callback)  → threading.Thread
# =============================================================================

import socket
import time
import threading
from typing import Any, Callable, Dict, Optional, Tuple

from utils.constants import (
    CONNECTIVITY_CHECK_HOSTS,
    CONNECTIVITY_TIMEOUT,
    CONNECTIVITY_RETRIES,
    CONNECTIVITY_CACHE_TTL,
    MODE_ONLINE,
    MODE_OFFLINE,
    MODE_ONLINE_LABEL,
    MODE_OFFLINE_LABEL,
)
from utils.helpers import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# Module-level cache  (thread-safe via Lock)
# ─────────────────────────────────────────────

_cache_lock:            threading.Lock      = threading.Lock()
_cached_result:         Optional[bool]      = None
_cached_at:             float               = 0.0
_last_info:             Dict[str, Any]      = {}
_simulate_offline_until: float              = 0.0   # ADDED: simulation flag


# ─────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────

def _probe_host(
    host:    str,
    port:    int   = 53,
    timeout: float = CONNECTIVITY_TIMEOUT,
    retries: int   = CONNECTIVITY_RETRIES,
) -> Tuple[bool, float]:
    """
    Open a TCP socket to *host:port* with retry logic.

    FIXED: Now implements CONNECTIVITY_RETRIES (was making only 1 attempt).

    Returns
    -------
    (success: bool, latency_ms: float)
        latency_ms is -1.0 on failure.
    """
    for attempt in range(1, retries + 1):
        t0 = time.monotonic()
        try:
            with socket.create_connection((host, port), timeout=timeout):
                latency = (time.monotonic() - t0) * 1000
                logger.debug(
                    "Probe success: %s:%s on attempt %d (%.1f ms)",
                    host, port, attempt, latency,
                )
                return True, round(latency, 2)
        except OSError as exc:
            logger.debug(
                "Probe attempt %d/%d failed: %s:%s — %s",
                attempt, retries, host, port, exc,
            )

    return False, -1.0


def _check_connectivity() -> Tuple[bool, float, str]:
    """
    Try every host in CONNECTIVITY_CHECK_HOSTS.

    FIXED: Checks simulation flag first before probing network.

    Returns
    -------
    (online: bool, best_latency_ms: float, successful_host: str)
    """
    # FIXED: Respect active simulation
    if time.monotonic() < _simulate_offline_until:
        logger.debug("Simulation active — returning OFFLINE")
        return False, -1.0, "simulated-offline"

    for host, port in CONNECTIVITY_CHECK_HOSTS:
        success, latency = _probe_host(host, port)
        if success:
            logger.debug(
                "Connectivity OK via %s:%s (%.1f ms)", host, port, latency
            )
            return True, latency, f"{host}:{port}"

    logger.warning(
        "All connectivity probes failed — switching to OFFLINE mode. "
        "Tried: %s",
        [f"{h}:{p}" for h, p in CONNECTIVITY_CHECK_HOSTS],
    )
    return False, -1.0, "none"


def _build_info(online: bool, latency: float, host: str) -> Dict[str, Any]:
    """
    Assemble the connection-info dict returned by get_connection_info().

    FIXED: datetime.utcnow() deprecated → datetime.now(timezone.utc).
    """
    import datetime
    checked_at = (
        datetime.datetime.now(datetime.timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )
    return {
        "mode":       MODE_ONLINE if online else MODE_OFFLINE,
        "online":     online,
        "latency_ms": latency,
        "host":       host,
        "checked_at": checked_at,
        "cached":     False,
    }


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def is_online(force_refresh: bool = False) -> bool:
    """
    Return True if the system has internet access.

    FIXED: All cache reads moved inside the lock (race condition removed).
           Network I/O performed OUTSIDE the lock to avoid blocking threads.

    Parameters
    ----------
    force_refresh : bypass cache and re-probe immediately.
    """
    global _cached_result, _cached_at, _last_info

    # ── Phase 1: Check cache (inside lock) ───────────────────────────
    with _cache_lock:
        now       = time.monotonic()
        cache_age = now - _cached_at

        if (
            not force_refresh
            and _cached_result is not None
            and cache_age < CONNECTIVITY_CACHE_TTL
        ):
            logger.debug("Connectivity cache hit (age=%.1fs)", cache_age)
            cached_info          = dict(_last_info)
            cached_info["cached"] = True
            _last_info           = cached_info
            return _cached_result

    # ── Phase 2: Probe network (outside lock — no blocking) ──────────
    online, latency, host = _check_connectivity()

    # ── Phase 3: Write result back (inside lock) ─────────────────────
    with _cache_lock:
        _cached_result = online
        _cached_at     = time.monotonic()
        _last_info     = _build_info(online, latency, host)

    logger.info(
        "Connectivity check: %s | latency=%.1f ms | via=%s",
        "ONLINE" if online else "OFFLINE",
        latency,
        host,
    )
    return online


def get_mode() -> str:
    """
    ADDED: Return the current mode string constant.

    Returns
    -------
    "online" | "offline"  (matches MODE_ONLINE / MODE_OFFLINE constants)

    Use this for routing logic — use get_mode_label() for UI display.
    """
    return MODE_ONLINE if is_online() else MODE_OFFLINE


def get_connection_info(force_refresh: bool = False) -> Dict[str, Any]:
    """
    Return detailed connectivity information.

    Returns
    -------
    dict:
        mode        : "online" | "offline"
        online      : bool
        latency_ms  : float
        host        : str
        checked_at  : str  (ISO-8601 UTC)
        cached      : bool
    """
    is_online(force_refresh=force_refresh)

    with _cache_lock:
        return dict(_last_info)


def get_mode_label() -> str:
    """
    Return a human-readable mode string.

    FIXED: Uses MODE_ONLINE_LABEL / MODE_OFFLINE_LABEL from constants
           instead of hardcoded emoji strings.

    Returns
    -------
    "🟢 Online"  or  "🔴 Offline"
    """
    return MODE_ONLINE_LABEL if is_online() else MODE_OFFLINE_LABEL


def require_online(
    message: str = "This feature requires an internet connection.",
) -> None:
    """
    Raise ConnectionError if the system is offline.

    Use as a guard at the top of any function that MUST have
    internet access (Groq, OpenAI, Firebase sync).

    Raises
    ------
    ConnectionError
    """
    if not is_online():
        logger.error("require_online() failed — system is offline")
        raise ConnectionError(message)


def invalidate_cache() -> None:
    """
    Force the next call to is_online() to perform a fresh probe.

    Useful for:
    - UI "Refresh connection" button
    - Tests that simulate network toggling
    - After user changes VPN / firewall settings
    """
    global _cached_result, _cached_at, _last_info

    with _cache_lock:
        _cached_result = None
        _cached_at     = 0.0
        _last_info     = {}

    logger.info(
        "Connectivity cache invalidated — next call will probe network"
    )


def simulate_offline(duration_seconds: float = 10.0) -> None:
    """
    **Development / testing only.**

    FIXED: Uses dedicated _simulate_offline_until flag instead of
           flawed TTL math. _check_connectivity() checks this flag first.

    Parameters
    ----------
    duration_seconds : how long to simulate offline mode.
    """
    global _cached_result, _cached_at, _last_info, _simulate_offline_until

    logger.warning(
        "SIMULATING OFFLINE MODE for %.1f seconds", duration_seconds
    )

    with _cache_lock:
        # FIXED: dedicated simulation flag — clean expiry time
        _simulate_offline_until = time.monotonic() + duration_seconds
        _cached_result          = False
        _cached_at              = time.monotonic()
        _last_info              = _build_info(False, -1.0, "simulated-offline")


def watch_connectivity(
    on_change:     Callable[[bool], None],
    poll_interval: float = 30.0,
) -> threading.Thread:
    """
    ADDED: Start a background thread that calls on_change(is_online: bool)
    whenever connectivity status changes.

    Useful for ui/system_status.py to auto-update the mode badge.

    Parameters
    ----------
    on_change     : callable(bool) — called when status flips
    poll_interval : seconds between checks

    Returns
    -------
    threading.Thread (daemon=True, already started)
    """
    def _watcher() -> None:
        last_status = is_online()
        while True:
            time.sleep(poll_interval)
            current = is_online(force_refresh=True)
            if current != last_status:
                logger.info(
                    "Connectivity changed: %s → %s",
                    "ONLINE"  if last_status else "OFFLINE",
                    "ONLINE"  if current     else "OFFLINE",
                )
                on_change(current)
                last_status = current

    t = threading.Thread(target=_watcher, daemon=True, name="connectivity-watcher")
    t.start()
    logger.info(
        "Connectivity watcher started (poll=%.0fs)", poll_interval
    )
    return t


# ─────────────────────────────────────────────
# Quick self-test  (python -m core.mode_detector)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Running connectivity check …")
    info = get_connection_info(force_refresh=True)
    for k, v in info.items():
        print(f"  {k:15s}: {v}")
    print(f"\nMode label : {get_mode_label()}")
    print(f"Mode string: {get_mode()}")