# =============================================================================
# services/sync_service.py
# AuraEcho+ — Offline-First Cloud Synchronization Service
#
# Responsibility:
#     Manage bidirectional sync between local SQLite and cloud Firestore.
#     Automatically pushes pending records when online, handles retries,
#     and provides sync status for the UI.
#
# Sync Flow:
#     1. Local operations add items to sync_queue table
#     2. Sync service periodically checks connectivity
#     3. If online, fetches pending queue items
#     4. Pushes records to cloud in batches
#     5. Updates queue status (SYNCED / FAILED)
#     6. Retries failed items up to MAX_SYNC_RETRIES
#
# Public API:
#     init_sync_service()         → None
#     run_sync_cycle()            → dict (sync results)
#     get_sync_status()           → dict
#     start_auto_sync()           → None
#     stop_auto_sync()            → None
#     force_sync()                → dict
#     is_sync_active()            → bool
#     get_sync_stats()            → dict
# =============================================================================

import os
import time
import threading
from typing import Any, Dict, List, Optional
from datetime import datetime

from database.local_db import (
    get_pending_sync,
    mark_synced,
    mark_sync_failed,
    get_sync_stats as get_local_sync_stats,
    get_patient,
    get_patient_history,
    DB_TABLE_PATIENTS,
    DB_TABLE_PREDICTIONS,
)
from database.cloud_db import (
    init_cloud_db,
    is_cloud_available,
    push_patient,
    push_prediction,
    batch_push,
    get_cloud_status,
)
from core.mode_detector import is_online, get_connection_info
from utils.constants import (
    SYNC_INTERVAL_SECONDS,
    SYNC_BATCH_SIZE,
    MAX_SYNC_RETRIES,
)
from utils.helpers import get_logger, now_str

logger = get_logger(__name__)

# ─────────────────────────────────────────────
# Module-level state
# ─────────────────────────────────────────────

_sync_thread: Optional[threading.Thread] = None
_sync_active: bool = False
_sync_lock: threading.Lock = threading.Lock()
_last_sync_time: Optional[str] = None
_last_sync_result: Dict[str, Any] = {}
_sync_stats: Dict[str, int] = {
    "total_synced": 0,
    "total_failed": 0,
    "cycles_run": 0,
}


# ─────────────────────────────────────────────
# Initialization
# ─────────────────────────────────────────────

def init_sync_service() -> None:
    """
    Initialize the sync service.
    Call this once at app startup.
    """
    global _sync_stats

    logger.info("Initializing sync service...")

    # Initialize cloud DB (lazy — won't fail if not configured)
    init_cloud_db()

    # Load initial stats
    _sync_stats = get_local_sync_stats()
    _sync_stats["total_synced"] = _sync_stats.get("SYNCED", 0)
    _sync_stats["total_failed"] = _sync_stats.get("FAILED", 0)

    logger.info(
        "Sync service initialized — pending: %d, synced: %d, failed: %d",
        _sync_stats.get("PENDING", 0),
        _sync_stats["total_synced"],
        _sync_stats["total_failed"],
    )


# ─────────────────────────────────────────────
# Sync operations
# ─────────────────────────────────────────────

def _fetch_record(table: str, record_id: int) -> Optional[Dict[str, Any]]:
    """
    Fetch a record from local DB by table and ID.
    """
    try:
        if table == DB_TABLE_PATIENTS:
            return get_patient(record_id)
        elif table == DB_TABLE_PREDICTIONS:
            # Get prediction from patient history
            # Note: This assumes prediction ID is unique across all patients
            # In practice, you may need a get_prediction(prediction_id) function
            # For now, we'll return a placeholder structure
            logger.warning(
                "Direct prediction fetch not implemented — "
                "consider adding get_prediction(prediction_id)"
            )
            return None
        else:
            logger.error("Unknown table for sync: %s", table)
            return None
    except Exception as exc:
        logger.error("Failed to fetch record %s/%d: %s", table, record_id, exc)
        return None


def _push_record(table: str, record: Dict[str, Any]) -> bool:
    """
    Push a single record to cloud.
    Returns True on success, False on failure.
    """
    try:
        if table == DB_TABLE_PATIENTS:
            doc_id = push_patient(record)
            return doc_id is not None
        elif table == DB_TABLE_PREDICTIONS:
            doc_id = push_prediction(record)
            return doc_id is not None
        else:
            logger.error("Unknown table for push: %s", table)
            return False
    except Exception as exc:
        logger.error("Push failed for %s/%d: %s", table, record.get("id"), exc)
        return False


def run_sync_cycle() -> Dict[str, Any]:
    """
    Execute one sync cycle.
    Checks connectivity, fetches pending items, pushes to cloud.

    Returns
    -------
    dict with sync results:
        {
            "success": bool,
            "synced_count": int,
            "failed_count": int,
            "pending_count": int,
            "message": str,
        }
    """
    global _last_sync_time, _last_sync_result, _sync_stats

    result = {
        "success": False,
        "synced_count": 0,
        "failed_count": 0,
        "pending_count": 0,
        "message": "",
        "timestamp": now_str(),
    }

    # Check connectivity
    if not is_online():
        result["message"] = "Offline — sync skipped"
        logger.debug("Sync cycle skipped — system is offline")
        _last_sync_result = result
        return result

    # Check cloud availability
    if not is_cloud_available():
        result["message"] = "Cloud unavailable — sync skipped"
        logger.warning("Sync cycle skipped — cloud database not available")
        _last_sync_result = result
        return result

    # Fetch pending items
    pending = get_pending_sync(limit=SYNC_BATCH_SIZE)
    if not pending:
        result["success"] = True
        result["message"] = "No pending items to sync"
        logger.debug("Sync cycle complete — no pending items")
        _last_sync_result = result
        return result

    logger.info("Sync cycle started — %d pending items", len(pending))

    synced_count = 0
    failed_count = 0

    for item in pending:
        queue_id = item["id"]
        table = item["table_name"]
        record_id = item["record_id"]
        retry_count = item.get("retry_count", 0)

        # Fetch record
        record = _fetch_record(table, record_id)
        if record is None:
            logger.error(
                "Record not found for sync queue %d — marking failed", queue_id
            )
            mark_sync_failed(queue_id, "Record not found in local DB")
            failed_count += 1
            continue

        # Push to cloud
        success = _push_record(table, record)

        if success:
            mark_synced(queue_id)
            synced_count += 1
            logger.debug("Synced %s/%d (queue_id=%d)", table, record_id, queue_id)
        else:
            # Check retry limit
            if retry_count >= MAX_SYNC_RETRIES:
                mark_sync_failed(queue_id, f"Max retries ({MAX_SYNC_RETRIES}) exceeded")
                logger.warning(
                    "Sync failed for %s/%d — max retries exceeded",
                    table, record_id,
                )
            else:
                mark_sync_failed(queue_id, f"Push failed (attempt {retry_count + 1})")
                logger.debug(
                    "Sync failed for %s/%d — will retry (attempt %d/%d)",
                    table, record_id, retry_count + 1, MAX_SYNC_RETRIES,
                )
            failed_count += 1

    # Update stats
    _sync_stats["total_synced"] += synced_count
    _sync_stats["total_failed"] += failed_count
    _sync_stats["cycles_run"] += 1

    # Get updated pending count
    local_stats = get_local_sync_stats()
    pending_count = local_stats.get("PENDING", 0)

    result["success"] = True
    result["synced_count"] = synced_count
    result["failed_count"] = failed_count
    result["pending_count"] = pending_count
    result["message"] = f"Synced {synced_count}, failed {failed_count}, pending {pending_count}"

    _last_sync_time = now_str()
    _last_sync_result = result

    logger.info(
        "Sync cycle complete — synced=%d, failed=%d, pending=%d",
        synced_count, failed_count, pending_count,
    )

    return result


def force_sync() -> Dict[str, Any]:
    """
    Force an immediate sync cycle.
    Useful for manual sync trigger from UI.
    """
    logger.info("Force sync requested")
    return run_sync_cycle()


# ─────────────────────────────────────────────
# Background sync thread
# ─────────────────────────────────────────────

def _sync_worker() -> None:
    """
    Background thread that runs sync cycles periodically.
    """
    global _sync_active

    logger.info(
        "Auto-sync worker started — interval=%ds", SYNC_INTERVAL_SECONDS
    )

    while _sync_active:
        try:
            run_sync_cycle()
        except Exception as exc:
            logger.error("Sync cycle error: %s", exc)

        # Sleep with interrupt check
        for _ in range(SYNC_INTERVAL_SECONDS):
            if not _sync_active:
                break
            time.sleep(1)

    logger.info("Auto-sync worker stopped")


def start_auto_sync() -> None:
    """
    Start the background auto-sync thread.
    """
    global _sync_thread, _sync_active

    with _sync_lock:
        if _sync_active:
            logger.warning("Auto-sync already running")
            return

        _sync_active = True
        _sync_thread = threading.Thread(
            target=_sync_worker,
            daemon=True,
            name="sync-worker",
        )
        _sync_thread.start()
        logger.info("Auto-sync started")


def stop_auto_sync() -> None:
    """
    Stop the background auto-sync thread.
    """
    global _sync_active, _sync_thread

    with _sync_lock:
        if not _sync_active:
            logger.warning("Auto-sync not running")
            return

        _sync_active = False
        if _sync_thread and _sync_thread.is_alive():
            _sync_thread.join(timeout=5.0)
        _sync_thread = None
        logger.info("Auto-sync stopped")


def is_sync_active() -> bool:
    """
    Check if auto-sync is currently running.
    """
    return _sync_active


# ─────────────────────────────────────────────
# Status and stats
# ─────────────────────────────────────────────

def get_sync_status() -> Dict[str, Any]:
    """
    Get comprehensive sync status for UI display.
    """
    local_stats = get_local_sync_stats()
    cloud_status = get_cloud_status()
    conn_info = get_connection_info()

    return {
        "auto_sync_active": _sync_active,
        "online": conn_info.get("online", False),
        "cloud_available": cloud_status.get("available", False),
        "cloud_connected": cloud_status.get("connected", False),
        "pending_count": local_stats.get("PENDING", 0),
        "synced_count": local_stats.get("SYNCED", 0),
        "failed_count": local_stats.get("FAILED", 0),
        "total_synced": _sync_stats.get("total_synced", 0),
        "total_failed": _sync_stats.get("total_failed", 0),
        "cycles_run": _sync_stats.get("cycles_run", 0),
        "last_sync_time": _last_sync_time,
        "last_sync_result": _last_sync_result,
        "sync_interval": SYNC_INTERVAL_SECONDS,
        "max_retries": MAX_SYNC_RETRIES,
        "batch_size": SYNC_BATCH_SIZE,
        "cloud_info": cloud_status,
    }


def get_sync_stats() -> Dict[str, int]:
    """
    Get simple sync statistics.
    """
    local_stats = get_local_sync_stats()
    return {
        "pending": local_stats.get("PENDING", 0),
        "synced": local_stats.get("SYNCED", 0),
        "failed": local_stats.get("FAILED", 0),
        "total_synced": _sync_stats.get("total_synced", 0),
        "total_failed": _sync_stats.get("total_failed", 0),
    }


def get_failed_sync_items(limit: int = 50) -> List[Dict[str, Any]]:
    """
    Get failed sync queue items for debugging.
    """
    from database.local_db import _get_connection

    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM sync_queue
        WHERE status = 'FAILED'
        ORDER BY created_at DESC
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def retry_failed_sync() -> Dict[str, Any]:
    """
    Retry all failed sync items by resetting their status to PENDING.
    """
    from database.local_db import _get_connection

    with _sync_lock:
        conn = _get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE sync_queue
            SET status = 'PENDING', retry_count = 0, error_msg = NULL
            WHERE status = 'FAILED'
        """)
        count = cursor.rowcount
        conn.commit()
        conn.close()

    logger.info("Reset %d failed sync items to PENDING", count)

    return {
        "success": True,
        "reset_count": count,
        "message": f"Reset {count} failed items for retry",
    }


def clear_synced_items() -> Dict[str, Any]:
    """
    Clear old synced items from the queue to reduce database size.
    Keeps items from the last 7 days.
    """
    from database.local_db import _get_connection

    with _sync_lock:
        conn = _get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            DELETE FROM sync_queue
            WHERE status = 'SYNCED'
            AND synced_at < datetime('now', '-7 days')
        """)
        count = cursor.rowcount
        conn.commit()
        conn.close()

    logger.info("Cleared %d old synced items", count)

    return {
        "success": True,
        "cleared_count": count,
        "message": f"Cleared {count} old synced items",
    }


# ─────────────────────────────────────────────
# Module init
# ─────────────────────────────────────────────

try:
    init_sync_service()
except Exception as _exc:
    logger.error("Sync service initialization failed: %s", _exc)