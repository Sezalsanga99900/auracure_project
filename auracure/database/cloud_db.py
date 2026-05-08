# =============================================================================
# database/cloud_db.py
# AuraEcho+ — Cloud Database Engine (Firebase Firestore)
#
# Responsibility:
#     Manage cloud persistence and synchronization with Firebase Firestore.
#     Provides batch push operations for the sync service.
#     Supports graceful degradation if Firebase is not configured.
#
# Public API:
#     init_cloud_db()               → bool
#     is_cloud_available()          → bool
#     push_patient(patient)         → str (doc_id)
#     push_prediction(prediction)   → str (doc_id)
#     batch_push(records)           → int (success_count)
#     get_cloud_status()            → dict
#     verify_connection()           → bool
# =============================================================================

import os
import json
from typing import Any, Dict, List, Optional
from datetime import datetime

try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False
    firestore = None
    firebase_admin = None

from utils.constants import (
    FIREBASE_COLLECTION,
    DB_TABLE_PATIENTS,
    DB_TABLE_PREDICTIONS,
    SYNC_BATCH_SIZE,
)
from utils.helpers import get_logger, now_str, mask_key

logger = get_logger(__name__)

# Module-level state
_db_client: Optional[Any] = None
_initialized: bool = False
_init_error: str = ""


# ─────────────────────────────────────────────
# Initialization
# ─────────────────────────────────────────────

def init_cloud_db() -> bool:
    """
    Initialize Firebase connection.
    Looks for GOOGLE_APPLICATION_CREDENTIALS env var or service account JSON.
    Returns True if initialized successfully, False otherwise.
    """
    global _db_client, _initialized, _init_error

    if _initialized:
        return _db_client is not None

    if not FIREBASE_AVAILABLE:
        _init_error = "firebase_admin not installed — pip install firebase-admin"
        logger.warning(_init_error)
        _initialized = True
        return False

    try:
        # Check for credentials
        cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
        cred_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON", "")

        if cred_path and os.path.exists(cred_path):
            cred = credentials.Certificate(cred_path)
            logger.info(
                "Firebase credentials loaded from file: %s",
                mask_key(cred_path, visible=10),
            )
        elif cred_json:
            cred_dict = json.loads(cred_json)
            cred = credentials.Certificate(cred_dict)
            logger.info("Firebase credentials loaded from JSON env var")
        else:
            _init_error = (
                "No Firebase credentials found. Set GOOGLE_APPLICATION_CREDENTIALS "
                "or FIREBASE_SERVICE_ACCOUNT_JSON environment variable."
            )
            logger.warning(_init_error)
            _initialized = True
            return False

        # Initialize app
        firebase_admin.initialize_app(cred)
        _db_client = firestore.client()
        _initialized = True
        logger.info("Firebase Firestore client initialized successfully")
        return True

    except Exception as exc:
        _init_error = f"Firebase init failed: {exc}"
        logger.error(_init_error)
        _initialized = True
        return False


def is_cloud_available() -> bool:
    """
    Check if cloud database is available and initialized.
    """
    return init_cloud_db() and _db_client is not None


# ─────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────

def _get_collection() -> Optional[Any]:
    """Get the Firestore collection reference."""
    if not is_cloud_available():
        return None
    return _db_client.collection(FIREBASE_COLLECTION)


def _prepare_record(
    table: str,
    record: Dict[str, Any],
    local_id: int,
) -> Dict[str, Any]:
    """
    Prepare a local record for cloud storage.
    Adds metadata fields for sync tracking.
    """
    now = now_str()
    cloud_record = {
        **record,
        "_meta": {
            "local_id": local_id,
            "local_table": table,
            "synced_at": now,
            "source": "auraecho_local",
        },
    }
    # Ensure timestamps are strings
    for key in ["created_at", "updated_at", "synced_at"]:
        if key in cloud_record and isinstance(cloud_record[key], datetime):
            cloud_record[key] = cloud_record[key].isoformat()
    return cloud_record


# ─────────────────────────────────────────────
# Push operations
# ─────────────────────────────────────────────

def push_patient(patient: Dict[str, Any]) -> Optional[str]:
    """
    Push a patient record to Firestore.

    Parameters
    ----------
    patient : dict with 'id' key (local patient ID)

    Returns
    -------
    doc_id : str or None if failed
    """
    coll = _get_collection()
    if coll is None:
        logger.warning("Cannot push patient — cloud unavailable")
        return None

    try:
        local_id = patient.get("id")
        if not local_id:
            logger.error("Patient missing 'id' field")
            return None

        cloud_record = _prepare_record(DB_TABLE_PATIENTS, patient, local_id)
        doc_ref = coll.document(f"patient_{local_id}")
        doc_ref.set(cloud_record)

        logger.info("Pushed patient local_id=%d → doc_id=%s", local_id, doc_ref.id)
        return doc_ref.id

    except Exception as exc:
        logger.error("Failed to push patient local_id=%d: %s", local_id, exc)
        return None


def push_prediction(prediction: Dict[str, Any]) -> Optional[str]:
    """
    Push a prediction record to Firestore.

    Parameters
    ----------
    prediction : dict with 'id' key (local prediction ID)

    Returns
    -------
    doc_id : str or None if failed
    """
    coll = _get_collection()
    if coll is None:
        logger.warning("Cannot push prediction — cloud unavailable")
        return None

    try:
        local_id = prediction.get("id")
        if not local_id:
            logger.error("Prediction missing 'id' field")
            return None

        cloud_record = _prepare_record(DB_TABLE_PREDICTIONS, prediction, local_id)
        doc_ref = coll.document(f"prediction_{local_id}")
        doc_ref.set(cloud_record)

        logger.info(
            "Pushed prediction local_id=%d → doc_id=%s",
            local_id, doc_ref.id,
        )
        return doc_ref.id

    except Exception as exc:
        logger.error("Failed to push prediction local_id=%d: %s", local_id, exc)
        return None


def batch_push(records: List[Dict[str, Any]]) -> int:
    """
    Push multiple records to Firestore in a batch.
    Each record must have: 'table', 'id', and 'data' keys.

    Parameters
    ----------
    records : list of {
        'table': 'patients' | 'predictions',
        'id': int,
        'data': dict
    }

    Returns
    -------
    success_count : int
    """
    coll = _get_collection()
    if coll is None:
        logger.warning("Cannot batch push — cloud unavailable")
        return 0

    if not records:
        return 0

    # Firestore batch limit is 500 operations
    batch_size = min(SYNC_BATCH_SIZE, 500)
    success_count = 0

    for i in range(0, len(records), batch_size):
        batch_records = records[i : i + batch_size]
        batch = _db_client.batch()
        batch_success = 0

        for rec in batch_records:
            try:
                table = rec.get("table")
                local_id = rec.get("id")
                data = rec.get("data", {})

                if not table or not local_id:
                    logger.warning("Invalid record format: %s", rec)
                    continue

                cloud_record = _prepare_record(table, data, local_id)
                doc_id = f"{table.rstrip('s')}_{local_id}"
                doc_ref = coll.document(doc_id)
                batch.set(doc_ref, cloud_record)
                batch_success += 1

            except Exception as exc:
                logger.error("Error preparing record for batch: %s", exc)
                continue

        try:
            batch.commit()
            success_count += batch_success
            logger.info(
                "Batch push committed: %d/%d records",
                batch_success, len(batch_records),
            )
        except Exception as exc:
            logger.error("Batch commit failed: %s", exc)
            # Batch is atomic — if commit fails, none are written

    logger.info("Total batch push: %d/%d records succeeded", success_count, len(records))
    return success_count


# ─────────────────────────────────────────────
# Status / Health checks
# ─────────────────────────────────────────────

def verify_connection() -> bool:
    """
    Verify cloud connection by performing a lightweight read.
    Returns True if connection is working.
    """
    if not is_cloud_available():
        return False

    try:
        coll = _get_collection()
        # List collections is a lightweight operation
        _db_client.collections()
        return True
    except Exception as exc:
        logger.error("Cloud connection verification failed: %s", exc)
        return False


def get_cloud_status() -> Dict[str, Any]:
    """
    Return detailed cloud database status for system status panel.
    """
    cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
    cred_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON", "")

    has_creds = bool(cred_path or cred_json)
    available = is_cloud_available()
    connected = verify_connection() if available else False

    return {
        "provider":       "Firebase Firestore",
        "available":      available,
        "connected":      connected,
        "collection":     FIREBASE_COLLECTION,
        "credentials_set": has_creds,
        "init_error":     _init_error if not available else "",
        "last_check":     now_str(),
    }


# ─────────────────────────────────────────────
# Optional: Pull operations (for multi-device sync)
# ─────────────────────────────────────────────

def pull_patients(since: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Pull patient records from Firestore.
    If 'since' is provided, only pull records synced after that timestamp.
    """
    coll = _get_collection()
    if coll is None:
        return []

    try:
        query = coll.where("_meta.local_table", "==", DB_TABLE_PATIENTS)
        if since:
            query = query.where("_meta.synced_at", ">", since)

        docs = query.stream()
        patients = []
        for doc in docs:
            data = doc.to_dict()
            data["_doc_id"] = doc.id
            patients.append(data)

        logger.info("Pulled %d patients from cloud", len(patients))
        return patients

    except Exception as exc:
        logger.error("Failed to pull patients: %s", exc)
        return []


def pull_predictions(since: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Pull prediction records from Firestore.
    """
    coll = _get_collection()
    if coll is None:
        return []

    try:
        query = coll.where("_meta.local_table", "==", DB_TABLE_PREDICTIONS)
        if since:
            query = query.where("_meta.synced_at", ">", since)

        docs = query.stream()
        predictions = []
        for doc in docs:
            data = doc.to_dict()
            data["_doc_id"] = doc.id
            predictions.append(data)

        logger.info("Pulled %d predictions from cloud", len(predictions))
        return predictions

    except Exception as exc:
        logger.error("Failed to pull predictions: %s", exc)
        return []


# ─────────────────────────────────────────────
# Module init
# ─────────────────────────────────────────────

try:
    # Lazy init — don't fail app startup if cloud is not configured
    pass
except Exception as _exc:
    logger.error("Cloud DB module error: %s", _exc)