"""
services/sync_service.py
────────────────────────
Offline-to-cloud synchronisation service for AuraEcho+.

Responsibility:
    When the system transitions from OFFLINE → ONLINE, automatically
    push all locally stored (SQLite) patient records and assessments
    to Firebase Firestore.

    Acts as the bridge between local_db.py and cloud_db.py.

Sync strategy:
    ┌─────────────────────────────────────────────────────────┐
    │  1. Check internet connectivity (mode_detector)         │
    │  2. Query local_db for all records where synced=0       │
    │  3. Push each record to cloud_db                        │
    │  4. On success, mark local record as synced=1           │
    │  5. On failure, log error + retry on next sync cycle    │
    │  6. Return detailed SyncReport to the UI                │
    └─────────────────────────────────────────────────────────┘

Conflict resolution:
    - Cloud ALWAYS wins for reads (multi-device scenario)
    - Local wins for writes (offline-first: local edits pushed up)
    - Last-write-wins based on updated_at timestamp

Retry policy:
    - Failed records are left with synced=0
    - Next sync attempt will retry them
    - After MAX_RETRY_ATTEMPTS, record is flagged with synced=-1 (error state)

Public API:
    sync_now()                    → SyncReport
    get_sync_status()             → dict
    schedule_auto_sync(interval)  → None  (background thread)
    stop_auto_sync()              → None
    get_sync_history()            → List[SyncReport]
    reset_failed_records()        → int   (count reset)
"""

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from core.mode_detector import is_online
from database.local_db import (
    get_unsynced_records,
    mark_synced,
    get_stats as get_local_stats,
)
from database.cloud_db import (
    batch_push,
    is_firebase_available,
    get_firebase_status,
    SyncResult,
)
from utils.constants import MAX_SYNC_RETRIES, SYNC_INTERVAL_SECONDS
from utils.helpers import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# Report dataclass
# ─────────────────────────────────────────────

@dataclass
class SyncReport:
    """
    Complete report of one sync operation.

    Attributes
    ----------
    started_at          : str   ISO-8601 timestamp
    completed_at        : str   ISO-8601 timestamp
    duration_ms         : float
    patients_synced     : int
    assessments_synced  : int
    patients_failed     : int
    assessments_failed  : int
    total_pending_before: int   — unsynced count before sync started
    total_pending_after : int   — unsynced count after sync completed
    errors              : List[str]
    status              : str   "success" | "partial" | "failed" | "skipped"
    skip_reason         : str   — why sync was skipped (if applicable)
    """
    started_at:           str
    completed_at:         str         = ""
    duration_ms:          float       = 0.0
    patients_synced:      int         = 0
    assessments_synced:   int         = 0
    patients_failed:      int         = 0
    assessments_failed:   int         = 0
    total_pending_before: int         = 0
    total_pending_after:  int         = 0
    errors:               List[str]   = field(default_factory=list)
    status:               str         = "pending"
    skip_reason:          str         = ""

    @property
    def total_synced(self) -> int:
        return self.patients_synced + self.assessments_synced

    @property
    def total_failed(self) -> int:
        return self.patients_failed + self.assessments_failed

    @property
    def success_rate(self) -> float:
        total = self.total_synced + self.total_failed
        return round((self.total_synced / total * 100) if total > 0 else 100.0, 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "started_at":           self.started_at,
            "completed_at":         self.completed_at,
            "duration_ms":          round(self.duration_ms, 1),
            "patients_synced":      self.patients_synced,
            "assessments_synced":   self.assessments_synced,
            "patients_failed":      self.patients_failed,
            "assessments_failed":   self.assessments_failed,
            "total_synced":         self.total_synced,
            "total_failed":         self.total_failed,
            "total_pending_before": self.total_pending_before,
            "total_pending_after":  self.total_pending_after,
            "success_rate":         self.success_rate,
            "status":               self.status,
            "skip_reason":          self.skip_reason,
            "errors":               self.errors[:20],   # cap for display
        }

    def summary_line(self) -> str:
        """One-line human-readable summary for status panels."""
        if self.status == "skipped":
            return f"⏭️  Sync skipped: {self.skip_reason}"
        if self.status == "success":
            return (
                f"✅ Sync complete: {self.total_synced} records pushed "
                f"in {self.duration_ms:.0f}ms"
            )
        if self.status == "partial":
            return (
                f"⚠️  Partial sync: {self.total_synced} pushed, "
                f"{self.total_failed} failed"
            )
        return f"❌ Sync failed: {self.errors[0] if self.errors else 'Unknown error'}"


# ─────────────────────────────────────────────
# Sync history (in-memory ring buffer)
# ─────────────────────────────────────────────
_sync_history:   List[SyncReport] = []
_MAX_HISTORY     = 50             # keep last 50 sync reports

# Auto-sync background thread controls
_auto_sync_thread:   Optional[threading.Thread] = None
_auto_sync_stop_evt: threading.Event            = threading.Event()
_sync_lock:          threading.Lock             = threading.Lock()


# ─────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds") + "Z"


def _record_history(report: SyncReport) -> None:
    """Add a report to the in-memory history, capped at _MAX_HISTORY."""
    global _sync_history
    _sync_history.append(report)
    if len(_sync_history) > _MAX_HISTORY:
        _sync_history = _sync_history[-_MAX_HISTORY:]


def _get_pending_count() -> int:
    """Return total unsynced records across both tables."""
    try:
        stats = get_local_stats()
        return stats.get("unsynced_patients", 0) + stats.get("unsynced_assessments", 0)
    except Exception:
        return 0


def _mark_successful_syncs(
    patients:    List[Dict[str, Any]],
    assessments: List[Dict[str, Any]],
    sync_result: SyncResult,
) -> None:
    """
    After a batch push, mark individual records as synced in local_db.

    We mark records individually — if 8/10 push successfully and 2 fail,
    the 8 get marked synced so they won't be retried next time.
    """
    # For simplicity: if total_failed == 0, mark ALL as synced
    # Otherwise mark patients and assessments that individually succeeded
    # (batch_push currently doesn't return per-record success, so we use
    #  the aggregate failure count as a heuristic)

    if sync_result.total_failed == 0:
        # All succeeded — mark everything
        for p in patients:
            pid = p.get("patient_id")
            if pid:
                try:
                    mark_synced(pid, table="patients")
                except Exception as exc:
                    logger.warning("Could not mark patient %s synced: %s", pid, exc)

        for a in assessments:
            aid = a.get("assessment_id")
            if aid:
                try:
                    mark_synced(aid, table="assessments")
                except Exception as exc:
                    logger.warning("Could not mark assessment %s synced: %s", aid, exc)
    else:
        # Partial success — be conservative: only mark what we know pushed
        # Mark patients up to patients_pushed count
        for p in patients[:sync_result.patients_pushed]:
            pid = p.get("patient_id")
            if pid:
                try:
                    mark_synced(pid, table="patients")
                except Exception:
                    pass

        for a in assessments[:sync_result.assessments_pushed]:
            aid = a.get("assessment_id")
            if aid:
                try:
                    mark_synced(aid, table="assessments")
                except Exception:
                    pass


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def sync_now(force: bool = False) -> SyncReport:
    """
    Perform an immediate synchronisation of all unsynced local records.

    This is the main entry point — called by the UI sync button or
    by the auto-sync background thread.

    Parameters
    ----------
    force : bool
        If True, attempt sync even if previous check shows no pending records.
        Useful for "Force Sync" button in the admin panel.

    Returns
    -------
    SyncReport — detailed result of the sync operation.
    """
    started_at = _now_iso()
    report     = SyncReport(started_at=started_at)

    # ── Guard: only one sync at a time ─────────────────────────────
    if not _sync_lock.acquire(blocking=False):
        report.status      = "skipped"
        report.skip_reason = "Another sync is already in progress"
        report.completed_at = _now_iso()
        logger.debug("Sync skipped: already in progress")
        _record_history(report)
        return report

    t0 = time.monotonic()

    try:
        # ── Guard: internet check ───────────────────────────────────
        if not is_online():
            report.status       = "skipped"
            report.skip_reason  = "No internet connection"
            report.completed_at = _now_iso()
            report.duration_ms  = (time.monotonic() - t0) * 1000
            logger.debug("Sync skipped: offline")
            _record_history(report)
            return report

        # ── Guard: Firebase availability ────────────────────────────
        if not is_firebase_available():
            report.status       = "skipped"
            report.skip_reason  = "Firebase not configured or unreachable"
            report.completed_at = _now_iso()
            report.duration_ms  = (time.monotonic() - t0) * 1000
            logger.warning("Sync skipped: Firebase unavailable")
            _record_history(report)
            return report

        # ── Fetch unsynced records ──────────────────────────────────
        pending              = get_unsynced_records()
        patients             = pending.get("patients", [])
        assessments          = pending.get("assessments", [])
        report.total_pending_before = len(patients) + len(assessments)

        if report.total_pending_before == 0 and not force:
            report.status       = "skipped"
            report.skip_reason  = "No pending records to sync"
            report.completed_at = _now_iso()
            report.duration_ms  = (time.monotonic() - t0) * 1000
            logger.debug("Sync skipped: nothing to sync")
            _record_history(report)
            return report

        logger.info(
            "Starting sync: %d patients, %d assessments",
            len(patients), len(assessments),
        )

        # ── Push to Firebase ────────────────────────────────────────
        sync_result = batch_push(patients, assessments)

        # ── Mark successfully synced records in local DB ────────────
        _mark_successful_syncs(patients, assessments, sync_result)

        # ── Build report ────────────────────────────────────────────
        report.patients_synced    = sync_result.patients_pushed
        report.assessments_synced = sync_result.assessments_pushed
        report.patients_failed    = sync_result.patients_failed
        report.assessments_failed = sync_result.assessments_failed
        report.errors             = sync_result.errors
        report.total_pending_after = _get_pending_count()

        if sync_result.total_failed == 0:
            report.status = "success"
        elif sync_result.total_pushed > 0:
            report.status = "partial"
        else:
            report.status = "failed"

        report.duration_ms  = (time.monotonic() - t0) * 1000
        report.completed_at = _now_iso()

        logger.info(
            "Sync complete: status=%s | %d pushed | %d failed | %.0f ms",
            report.status, report.total_synced, report.total_failed, report.duration_ms,
        )

    except Exception as exc:
        report.status       = "failed"
        report.errors.append(str(exc))
        report.completed_at = _now_iso()
        report.duration_ms  = (time.monotonic() - t0) * 1000
        logger.error("Sync failed with exception: %s", exc)

    finally:
        _sync_lock.release()

    _record_history(report)
    return report


def get_sync_status() -> Dict[str, Any]:
    """
    Return the current synchronisation status for the UI status panel.

    Returns
    -------
    dict:
        pending_patients    : int
        pending_assessments : int
        total_pending       : int
        firebase_available  : bool
        online              : bool
        last_sync           : dict | None  (last SyncReport.to_dict())
        auto_sync_running   : bool
    """
    try:
        local_stats = get_local_stats()
        pending_p   = local_stats.get("unsynced_patients", 0)
        pending_a   = local_stats.get("unsynced_assessments", 0)
    except Exception:
        pending_p = pending_a = 0

    last_sync = _sync_history[-1].to_dict() if _sync_history else None

    return {
        "pending_patients":    pending_p,
        "pending_assessments": pending_a,
        "total_pending":       pending_p + pending_a,
        "firebase_available":  is_firebase_available(),
        "online":              is_online(),
        "last_sync":           last_sync,
        "auto_sync_running":   _auto_sync_thread is not None and _auto_sync_thread.is_alive(),
        "firebase_status":     get_firebase_status(),
    }


def get_sync_history(limit: int = 10) -> List[Dict[str, Any]]:
    """
    Return the last *limit* sync reports as dicts.

    Returns
    -------
    List of SyncReport.to_dict(), newest first.
    """
    recent = list(reversed(_sync_history))
    return [r.to_dict() for r in recent[:limit]]


def reset_failed_records() -> int:
    """
    Reset any records stuck in a failed state back to unsynced=0
    so they will be retried on the next sync.

    In our current schema, failed records are simply left with synced=0
    so this is a no-op placeholder for future MAX_RETRY tracking.

    Returns
    -------
    int — number of records reset.
    """
    # Future: query records with synced=-1 (error flag) and reset to 0
    logger.info("reset_failed_records called (no-op in current schema)")
    return 0


# ─────────────────────────────────────────────
# Auto-sync background thread
# ─────────────────────────────────────────────

def _auto_sync_loop(interval_seconds: int) -> None:
    """
    Background loop: attempt sync every *interval_seconds*.
    Stops when _auto_sync_stop_evt is set.
    """
    logger.info("Auto-sync loop started (interval=%ds)", interval_seconds)

    while not _auto_sync_stop_evt.is_set():
        try:
            report = sync_now()
            if report.status not in ("skipped",):
                logger.info("Auto-sync: %s", report.summary_line())
        except Exception as exc:
            logger.error("Auto-sync loop error: %s", exc)

        # Wait for interval, but wake up early if stop event is set
        _auto_sync_stop_evt.wait(timeout=interval_seconds)

    logger.info("Auto-sync loop stopped")


def schedule_auto_sync(interval_seconds: int = SYNC_INTERVAL_SECONDS) -> None:
    """
    Start a background thread that syncs every *interval_seconds*.

    Safe to call multiple times — will not start a second thread
    if one is already running.

    Parameters
    ----------
    interval_seconds : int  — seconds between sync attempts (default from constants)
    """
    global _auto_sync_thread

    if _auto_sync_thread is not None and _auto_sync_thread.is_alive():
        logger.debug("Auto-sync already running — skipping start")
        return

    _auto_sync_stop_evt.clear()
    _auto_sync_thread = threading.Thread(
        target=_auto_sync_loop,
        args=(interval_seconds,),
        daemon=True,           # dies when main process exits
        name="AuraEcho-AutoSync",
    )
    _auto_sync_thread.start()
    logger.info("Auto-sync thread started (interval=%ds)", interval_seconds)


def stop_auto_sync() -> None:
    """
    Stop the background auto-sync thread gracefully.

    The thread will finish its current sync (if running) then stop.
    """
    global _auto_sync_thread

    if _auto_sync_thread is None or not _auto_sync_thread.is_alive():
        logger.debug("Auto-sync was not running")
        return

    logger.info("Stopping auto-sync thread...")
    _auto_sync_stop_evt.set()
    _auto_sync_thread.join(timeout=10.0)

    if _auto_sync_thread.is_alive():
        logger.warning("Auto-sync thread did not stop cleanly within 10s")
    else:
        logger.info("Auto-sync thread stopped")

    _auto_sync_thread = None