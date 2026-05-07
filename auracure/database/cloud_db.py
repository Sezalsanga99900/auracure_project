"""
database/cloud_db.py
────────────────────
Firebase Firestore cloud database for AuraEcho+.

Responsibility:
    Push patient records and assessments to Firebase Firestore when
    the system is online.  Acts as the cloud mirror of local_db.py —
    same data structure, different storage backend.

Why Firebase?
    - Real-time sync across devices (doctor's tablet + nurse's laptop)
    - Offline SDK support (Firebase caches writes when offline)
    - Simple NoSQL document model that matches our patient dict structure
    - Free tier sufficient for clinic-scale usage

Architecture:
    local_db.py  →  sync_service.py  →  cloud_db.py  →  Firestore
                    (on reconnect)

Collections:
    /patients/{patient_id}          — patient demographic + clinical data
    /assessments/{assessment_id}    — risk scores + AI diagnosis results

Public API:
    push_patient(patient_dict)              → bool
    push_assessment(assessment_dict)        → bool
    fetch_patient(patient_id)               → dict | None
    fetch_all_patients(limit)               → List[dict]
    fetch_assessments(patient_id)           → List[dict]
    is_firebase_available()                 → bool
    get_firebase_status()                   → dict
    batch_push(patients, assessments)       → SyncResult
"""

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from utils.constants import (
    FIREBASE_PATIENTS_COLLECTION,
    FIREBASE_ASSESSMENTS_COLLECTION,
    FIREBASE_PROJECT_ID,
)
from utils.helpers import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────
# Optional Firebase import
# ─────────────────────────────────────────────
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    FIREBASE_SDK_AVAILABLE = True
except ImportError:
    FIREBASE_SDK_AVAILABLE = False
    logger.warning(
        "firebase-admin not installed — cloud sync disabled. "
        "Install with: pip install firebase-admin"
    )

# ─────────────────────────────────────────────
# Firebase app singleton
# ─────────────────────────────────────────────
_firebase_app  = None
_firestore_db  = None


def _init_firebase() -> bool:
    """
    Initialise the Firebase Admin SDK.

    Looks for credentials in this priority order:
    1. FIREBASE_CREDENTIALS_JSON env var (JSON string — for cloud deployment)
    2. FIREBASE_CREDENTIALS_PATH env var (path to service-account JSON file)
    3. data/firebase_credentials.json (local development fallback)

    Returns
    -------
    True if Firebase was successfully initialised, False otherwise.
    """
    global _firebase_app, _firestore_db

    if _firestore_db is not None:
        return True   # already initialised

    if not FIREBASE_SDK_AVAILABLE:
        logger.warning("Firebase SDK not available")
        return False

    # Already initialised by another call
    try:
        existing = firebase_admin.get_app()
        _firebase_app = existing
        _firestore_db = firestore.client()
        return True
    except ValueError:
        pass   # No app initialised yet — proceed below

    # ── Locate credentials ──────────────────────────────────────────
    cred_obj = None

    # Option 1: JSON string in environment (for deployment)
    cred_json_str = os.getenv("FIREBASE_CREDENTIALS_JSON", "")
    if cred_json_str:
        try:
            cred_dict = json.loads(cred_json_str)
            cred_obj  = credentials.Certificate(cred_dict)
            logger.info("Firebase: using credentials from FIREBASE_CREDENTIALS_JSON env var")
        except (json.JSONDecodeError, Exception) as exc:
            logger.error("Failed to parse FIREBASE_CREDENTIALS_JSON: %s", exc)

    # Option 2: Path to JSON file
    if cred_obj is None:
        cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH", "data/firebase_credentials.json")
        if os.path.exists(cred_path):
            try:
                cred_obj = credentials.Certificate(cred_path)
                logger.info("Firebase: using credentials from %s", cred_path)
            except Exception as exc:
                logger.error("Failed to load Firebase credentials from %s: %s", cred_path, exc)

    if cred_obj is None:
        logger.warning(
            "No Firebase credentials found. "
            "Set FIREBASE_CREDENTIALS_JSON or FIREBASE_CREDENTIALS_PATH, "
            "or place data/firebase_credentials.json in the project root."
        )
        return False

    # ── Initialise app ──────────────────────────────────────────────
    try:
        project_id = os.getenv("FIREBASE_PROJECT_ID", FIREBASE_PROJECT_ID)
        _firebase_app = firebase_admin.initialize_app(
            cred_obj,
            options={"projectId": project_id} if project_id else {},
        )
        _firestore_db = firestore.client()
        logger.info("Firebase initialised successfully (project=%s)", project_id)
        return True
    except Exception as exc:
        logger.error("Firebase initialisation failed: %s", exc)
        return False


def _get_db():
    """
    Return the Firestore client, initialising Firebase if needed.
    Returns None if Firebase is unavailable.
    """
    if _firestore_db is not None:
        return _firestore_db
    success = _init_firebase()
    return _firestore_db if success else None


# ─────────────────────────────────────────────
# Status checks
# ─────────────────────────────────────────────

def is_firebase_available() -> bool:
    """
    Return True if Firebase is configured and reachable.
    """
    db = _get_db()
    if db is None:
        return False

    # Quick connectivity check — try listing 1 document
    try:
        next(iter(
            db.collection(FIREBASE_PATIENTS_COLLECTION).limit(1).stream()
        ), None)
        return True
    except Exception as exc:
        logger.debug("Firebase connectivity check failed: %s", exc)
        return False


def get_firebase_status() -> Dict[str, Any]:
    """
    Return a detailed Firebase status dict for the system panel.

    Returns
    -------
    dict:
        available         : bool
        sdk_installed     : bool
        credentials_found : bool
        project_id        : str
        patients_collection   : str
        assessments_collection: str
        error             : str
    """
    sdk_ok  = FIREBASE_SDK_AVAILABLE
    cred_ok = (
        bool(os.getenv("FIREBASE_CREDENTIALS_JSON"))
        or bool(os.getenv("FIREBASE_CREDENTIALS_PATH"))
        or os.path.exists("data/firebase_credentials.json")
    )
    db      = _get_db()
    avail   = db is not None

    error = ""
    if not sdk_ok:
        error = "firebase-admin not installed"
    elif not cred_ok:
        error = "No credentials found"
    elif not avail:
        error = "Firebase init failed"

    return {
        "available":              avail,
        "sdk_installed":          sdk_ok,
        "credentials_found":      cred_ok,
        "project_id":             os.getenv("FIREBASE_PROJECT_ID", FIREBASE_PROJECT_ID),
        "patients_collection":    FIREBASE_PATIENTS_COLLECTION,
        "assessments_collection": FIREBASE_ASSESSMENTS_COLLECTION,
        "error":                  error,
    }


# ─────────────────────────────────────────────
# Sync result
# ─────────────────────────────────────────────

@dataclass
class SyncResult:
    """Result of a batch sync operation."""
    patients_pushed:    int = 0
    assessments_pushed: int = 0
    patients_failed:    int = 0
    assessments_failed: int = 0
    errors:             List[str] = field(default_factory=list)
    duration_ms:        float = 0.0

    @property
    def total_pushed(self) -> int:
        return self.patients_pushed + self.assessments_pushed

    @property
    def total_failed(self) -> int:
        return self.patients_failed + self.assessments_failed

    @property
    def success(self) -> bool:
        return self.total_failed == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "patients_pushed":    self.patients_pushed,
            "assessments_pushed": self.assessments_pushed,
            "patients_failed":    self.patients_failed,
            "assessments_failed": self.assessments_failed,
            "total_pushed":       self.total_pushed,
            "total_failed":       self.total_failed,
            "success":            self.success,
            "duration_ms":        round(self.duration_ms, 1),
            "errors":             self.errors[:10],   # cap error list
        }


# ─────────────────────────────────────────────
# Write operations
# ─────────────────────────────────────────────

def push_patient(patient: Dict[str, Any]) -> bool:
    """
    Push a single patient record to Firestore.

    Uses the patient_id as the Firestore document ID so that
    re-pushing the same patient is idempotent (merge=True).

    Parameters
    ----------
    patient : dict  — full patient record (from local_db or UI)

    Returns
    -------
    True on success, False on failure.
    """
    db = _get_db()
    if db is None:
        logger.warning("push_patient: Firebase not available")
        return False

    patient_id = patient.get("patient_id")
    if not patient_id:
        logger.error("push_patient: patient_id missing — cannot push")
        return False

    # Add cloud metadata
    doc_data = dict(patient)
    doc_data["cloud_updated_at"] = datetime.now(timezone.utc).isoformat()
    doc_data["source"]           = "auraecho_plus"

    # Remove non-serialisable items
    doc_data.pop("_sa_instance_state", None)   # SQLAlchemy artifact if any

    try:
        db.collection(FIREBASE_PATIENTS_COLLECTION).document(patient_id).set(
            doc_data, merge=True
        )
        logger.info("Patient pushed to Firebase: %s", patient_id)
        return True
    except Exception as exc:
        logger.error("Failed to push patient %s: %s", patient_id, exc)
        return False


def push_assessment(assessment: Dict[str, Any]) -> bool:
    """
    Push a single assessment record to Firestore.

    Parameters
    ----------
    assessment : dict  — assessment record (from local_db.get_assessments)

    Returns
    -------
    True on success, False on failure.
    """
    db = _get_db()
    if db is None:
        return False

    assessment_id = assessment.get("assessment_id")
    if not assessment_id:
        logger.error("push_assessment: assessment_id missing")
        return False

    doc_data = dict(assessment)
    doc_data["cloud_updated_at"] = datetime.now(timezone.utc).isoformat()

    # Serialise nested dicts/lists to JSON strings for Firestore
    for field_name in ("ai_diagnosis", "similar_cases"):
        val = doc_data.get(field_name)
        if isinstance(val, (dict, list)):
            doc_data[field_name] = json.dumps(val)

    try:
        db.collection(FIREBASE_ASSESSMENTS_COLLECTION).document(assessment_id).set(
            doc_data, merge=True
        )
        logger.info("Assessment pushed to Firebase: %s", assessment_id)
        return True
    except Exception as exc:
        logger.error("Failed to push assessment %s: %s", assessment_id, exc)
        return False


def batch_push(
    patients:    List[Dict[str, Any]],
    assessments: List[Dict[str, Any]],
) -> SyncResult:
    """
    Push multiple patients and assessments in one call.

    Uses Firestore batch writes (up to 500 documents per batch).

    Parameters
    ----------
    patients    : list of patient dicts
    assessments : list of assessment dicts

    Returns
    -------
    SyncResult — detailed breakdown of what succeeded/failed
    """
    db = _get_db()
    result = SyncResult()
    t0 = time.monotonic()

    if db is None:
        result.patients_failed    = len(patients)
        result.assessments_failed = len(assessments)
        result.errors.append("Firebase not available")
        result.duration_ms = (time.monotonic() - t0) * 1000
        return result

    # ── Push patients ───────────────────────────────────────────────
    for p in patients:
        success = push_patient(p)
        if success:
            result.patients_pushed += 1
        else:
            result.patients_failed += 1
            result.errors.append(f"Patient {p.get('patient_id','?')} push failed")

    # ── Push assessments ────────────────────────────────────────────
    for a in assessments:
        success = push_assessment(a)
        if success:
            result.assessments_pushed += 1
        else:
            result.assessments_failed += 1
            result.errors.append(f"Assessment {a.get('assessment_id','?')} push failed")

    result.duration_ms = (time.monotonic() - t0) * 1000

    logger.info(
        "Batch push complete: %d patients, %d assessments | %.0f ms | %d errors",
        result.patients_pushed, result.assessments_pushed,
        result.duration_ms, result.total_failed,
    )
    return result


# ─────────────────────────────────────────────
# Read operations
# ─────────────────────────────────────────────

def fetch_patient(patient_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch a single patient from Firestore by patient_id.

    Returns
    -------
    Patient dict, or None if not found.
    """
    db = _get_db()
    if db is None:
        return None

    try:
        doc = db.collection(FIREBASE_PATIENTS_COLLECTION).document(patient_id).get()
        if doc.exists:
            data = doc.to_dict()
            data["patient_id"] = patient_id
            return data
        return None
    except Exception as exc:
        logger.error("fetch_patient failed for %s: %s", patient_id, exc)
        return None


def fetch_all_patients(limit: int = 100) -> List[Dict[str, Any]]:
    """
    Fetch the most recent *limit* patients from Firestore.

    Returns
    -------
    List of patient dicts, ordered by cloud_updated_at descending.
    """
    db = _get_db()
    if db is None:
        return []

    try:
        docs = (
            db.collection(FIREBASE_PATIENTS_COLLECTION)
            .order_by("cloud_updated_at", direction="DESCENDING")
            .limit(limit)
            .stream()
        )
        patients = []
        for doc in docs:
            data = doc.to_dict()
            data["patient_id"] = doc.id
            patients.append(data)
        logger.info("Fetched %d patients from Firebase", len(patients))
        return patients
    except Exception as exc:
        logger.error("fetch_all_patients failed: %s", exc)
        return []


def fetch_assessments(patient_id: str) -> List[Dict[str, Any]]:
    """
    Fetch all assessments for a patient from Firestore.

    Returns
    -------
    List of assessment dicts, newest first.
    """
    db = _get_db()
    if db is None:
        return []

    try:
        docs = (
            db.collection(FIREBASE_ASSESSMENTS_COLLECTION)
            .where("patient_id", "==", patient_id)
            .order_by("created_at", direction="DESCENDING")
            .stream()
        )
        assessments = []
        for doc in docs:
            data = doc.to_dict()
            data["assessment_id"] = doc.id

            # Deserialise JSON strings
            for json_field in ("ai_diagnosis", "similar_cases"):
                val = data.get(json_field)
                if isinstance(val, str):
                    try:
                        data[json_field] = json.loads(val)
                    except json.JSONDecodeError:
                        pass

            assessments.append(data)

        return assessments
    except Exception as exc:
        logger.error("fetch_assessments failed for %s: %s", patient_id, exc)
        return []


def delete_patient_cloud(patient_id: str) -> bool:
    """
    Delete a patient and all their assessments from Firestore.

    Returns
    -------
    True if the patient document was found and deleted.
    """
    db = _get_db()
    if db is None:
        return False

    try:
        # Delete patient document
        db.collection(FIREBASE_PATIENTS_COLLECTION).document(patient_id).delete()

        # Delete all linked assessments
        assessment_docs = (
            db.collection(FIREBASE_ASSESSMENTS_COLLECTION)
            .where("patient_id", "==", patient_id)
            .stream()
        )
        for doc in assessment_docs:
            doc.reference.delete()

        logger.info("Patient deleted from Firebase: %s", patient_id)
        return True
    except Exception as exc:
        logger.error("Cloud delete failed for %s: %s", patient_id, exc)
        return False