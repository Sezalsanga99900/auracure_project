"""
database/local_db.py
────────────────────
Local SQLite database for AuraEcho+ patient records.

Responsibility:
    Persist patient records, AI diagnoses, and risk assessments locally
    — completely offline, zero cloud dependency. Acts as the primary
    data store in offline mode and as the write-ahead buffer in online
    mode (records are later synced to Firebase by sync_service.py).

Database schema:
    ┌─────────────────────────────────────────────────────────┐
    │  TABLE: patients                                        │
    │    id           INTEGER PRIMARY KEY AUTOINCREMENT       │
    │    patient_id   TEXT UNIQUE  (UUID)                     │
    │    name         TEXT                                    │
    │    age          INTEGER                                 │
    │    sex          TEXT                                    │
    │    created_at   TEXT  (ISO-8601)                        │
    │    updated_at   TEXT  (ISO-8601)                        │
    │    synced       INTEGER  0=not synced, 1=synced         │
    │    raw_data     TEXT  (full JSON blob)                  │
    ├─────────────────────────────────────────────────────────┤
    │  TABLE: assessments                                     │
    │    id              INTEGER PRIMARY KEY AUTOINCREMENT    │
    │    assessment_id   TEXT UNIQUE  (UUID)                  │
    │    patient_id      TEXT  (FK → patients.patient_id)     │
    │    risk_level      TEXT                                 │
    │    confidence_pct  REAL                                 │
    │    disease_prob    REAL                                 │
    │    ai_diagnosis    TEXT  (AI response JSON)             │
    │    similar_cases   TEXT  (JSON array)                   │
    │    created_at      TEXT                                 │
    │    synced          INTEGER                              │
    └─────────────────────────────────────────────────────────┘

Public API:
    save_patient(patient_dict)              → patient_id: str
    get_patient(patient_id)                 → dict | None
    get_all_patients(limit, offset)         → List[dict]
    save_assessment(patient_id, assessment) → assessment_id: str
    get_assessments(patient_id)             → List[dict]
    get_unsynced_records()                  → List[dict]
    mark_synced(record_id, table)           → None
    search_patients(query)                  → List[dict]
    delete_patient(patient_id)             → bool
    export_to_csv(filepath)                → str
    get_stats()                            → dict
"""

import csv
import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from utils.constants import (
    LOCAL_DB_PATH,
    DB_PATIENTS_TABLE,
    DB_ASSESSMENTS_TABLE,
)
from utils.helpers import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# Schema DDL
# ─────────────────────────────────────────────

_CREATE_PATIENTS_TABLE = """
CREATE TABLE IF NOT EXISTS patients (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id   TEXT    NOT NULL UNIQUE,
    name         TEXT    NOT NULL DEFAULT 'Unknown',
    age          INTEGER,
    sex          TEXT,
    created_at   TEXT    NOT NULL,
    updated_at   TEXT    NOT NULL,
    synced       INTEGER NOT NULL DEFAULT 0,
    raw_data     TEXT    NOT NULL
);
"""

_CREATE_ASSESSMENTS_TABLE = """
CREATE TABLE IF NOT EXISTS assessments (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    assessment_id   TEXT    NOT NULL UNIQUE,
    patient_id      TEXT    NOT NULL,
    risk_level      TEXT,
    confidence_pct  REAL,
    disease_prob    REAL,
    ai_source       TEXT,
    ai_diagnosis    TEXT,
    similar_cases   TEXT,
    created_at      TEXT    NOT NULL,
    synced          INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_patients_patient_id ON patients(patient_id);",
    "CREATE INDEX IF NOT EXISTS idx_patients_name       ON patients(name);",
    "CREATE INDEX IF NOT EXISTS idx_assessments_patient ON assessments(patient_id);",
    "CREATE INDEX IF NOT EXISTS idx_patients_synced     ON patients(synced);",
    "CREATE INDEX IF NOT EXISTS idx_assessments_synced  ON assessments(synced);",
]


# ─────────────────────────────────────────────
# Database initialisation
# ─────────────────────────────────────────────

def _ensure_db_dir() -> None:
    """Create the directory for LOCAL_DB_PATH if it doesn't exist."""
    db_dir = Path(LOCAL_DB_PATH).parent
    db_dir.mkdir(parents=True, exist_ok=True)


def init_db() -> None:
    """
    Create tables and indexes if they don't exist.
    Safe to call multiple times (all statements use IF NOT EXISTS).
    """
    _ensure_db_dir()
    with _get_connection() as conn:
        conn.execute(_CREATE_PATIENTS_TABLE)
        conn.execute(_CREATE_ASSESSMENTS_TABLE)
        for idx_sql in _CREATE_INDEXES:
            conn.execute(idx_sql)
        conn.commit()
    logger.info("Local database initialised at %s", LOCAL_DB_PATH)


@contextmanager
def _get_connection() -> Generator[sqlite3.Connection, None, None]:
    """
    Context manager that yields a SQLite connection with sensible defaults.

    - WAL mode for concurrent reads
    - Row factory for dict-style row access
    - Foreign key enforcement
    """
    _ensure_db_dir()
    conn = sqlite3.connect(LOCAL_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ─────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────

def _now_iso() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _generate_id(prefix: str = "") -> str:
    """Generate a unique ID like 'pat_3f2a...' or 'asmnt_7c1b...'"""
    return f"{prefix}{uuid.uuid4().hex[:16]}"


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    """Convert a sqlite3.Row to a plain dict."""
    return dict(row)


def _deserialise_patient_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expand the raw_data JSON blob back into a full patient dict.
    Merges DB columns with raw_data for convenience.
    """
    result = dict(row)
    raw_data_str = result.pop("raw_data", "{}")
    try:
        raw_data = json.loads(raw_data_str)
    except (json.JSONDecodeError, TypeError):
        raw_data = {}
    result.update(raw_data)   # raw_data fields fill in clinical features
    return result


def _deserialise_assessment_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Expand JSON blobs in an assessment row."""
    result = dict(row)

    for json_field in ("ai_diagnosis", "similar_cases"):
        val = result.get(json_field)
        if isinstance(val, str):
            try:
                result[json_field] = json.loads(val)
            except (json.JSONDecodeError, TypeError):
                result[json_field] = {}

    return result


# ─────────────────────────────────────────────
# Patient CRUD
# ─────────────────────────────────────────────

def save_patient(patient: Dict[str, Any]) -> str:
    """
    Insert or update a patient record.

    If the patient dict already has a 'patient_id' key, it will be
    treated as an UPDATE (upsert).  Otherwise a new UUID is generated.

    Parameters
    ----------
    patient : dict
        Must contain clinical feature fields.
        May optionally contain: name, patient_id.

    Returns
    -------
    patient_id : str  — the UUID assigned to this patient
    """
    init_db()

    patient_id = patient.get("patient_id") or _generate_id("pat_")
    name       = str(patient.get("name", "Unknown Patient"))
    age        = int(float(patient.get("age", 0)))
    sex_raw    = patient.get("sex", 0)
    sex_label  = "Male" if int(float(sex_raw)) == 1 else "Female"
    now        = _now_iso()

    # Full patient data as JSON blob (including all clinical features)
    raw_data = json.dumps({k: v for k, v in patient.items()})

    sql = """
    INSERT INTO patients (patient_id, name, age, sex, created_at, updated_at, synced, raw_data)
    VALUES (?, ?, ?, ?, ?, ?, 0, ?)
    ON CONFLICT(patient_id) DO UPDATE SET
        name       = excluded.name,
        age        = excluded.age,
        sex        = excluded.sex,
        updated_at = excluded.updated_at,
        synced     = 0,
        raw_data   = excluded.raw_data
    """

    with _get_connection() as conn:
        conn.execute(sql, (patient_id, name, age, sex_label, now, now, raw_data))
        conn.commit()

    logger.info("Patient saved: id=%s name=%s", patient_id, name)
    return patient_id


def get_patient(patient_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve a single patient by patient_id.

    Returns
    -------
    Full patient dict (DB columns + clinical features), or None.
    """
    init_db()

    with _get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM patients WHERE patient_id = ?", (patient_id,)
        ).fetchone()

    if row is None:
        return None
    return _deserialise_patient_row(_row_to_dict(row))


def get_all_patients(
    limit: int = 100,
    offset: int = 0,
    synced_only: bool = False,
) -> List[Dict[str, Any]]:
    """
    Retrieve paginated list of all patients.

    Parameters
    ----------
    limit       : max records to return
    offset      : skip first N records (for pagination)
    synced_only : if True, return only cloud-synced records

    Returns
    -------
    List of patient dicts, newest first.
    """
    init_db()

    where = "WHERE synced = 1" if synced_only else ""
    sql   = f"SELECT * FROM patients {where} ORDER BY created_at DESC LIMIT ? OFFSET ?"

    with _get_connection() as conn:
        rows = conn.execute(sql, (limit, offset)).fetchall()

    return [_deserialise_patient_row(_row_to_dict(r)) for r in rows]


def search_patients(query: str, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Full-text search on patient name.

    Parameters
    ----------
    query : str  — search string (case-insensitive, partial match)
    limit : int  — max results

    Returns
    -------
    List of matching patient dicts.
    """
    init_db()

    pattern = f"%{query.strip()}%"
    sql = "SELECT * FROM patients WHERE name LIKE ? ORDER BY created_at DESC LIMIT ?"

    with _get_connection() as conn:
        rows = conn.execute(sql, (pattern, limit)).fetchall()

    return [_deserialise_patient_row(_row_to_dict(r)) for r in rows]


def delete_patient(patient_id: str) -> bool:
    """
    Delete a patient and all their assessments.

    Returns
    -------
    True if a record was deleted, False if not found.
    """
    init_db()

    with _get_connection() as conn:
        # Delete assessments first (FK constraint)
        conn.execute(
            "DELETE FROM assessments WHERE patient_id = ?", (patient_id,)
        )
        cursor = conn.execute(
            "DELETE FROM patients WHERE patient_id = ?", (patient_id,)
        )
        conn.commit()
        deleted = cursor.rowcount > 0

    if deleted:
        logger.info("Patient deleted: id=%s", patient_id)
    else:
        logger.warning("Delete called on non-existent patient: %s", patient_id)

    return deleted


# ─────────────────────────────────────────────
# Assessment CRUD
# ─────────────────────────────────────────────

def save_assessment(
    patient_id:    str,
    risk_result:   Optional[Dict[str, Any]]   = None,
    ai_response:   Optional[Dict[str, Any]]   = None,
    similar_cases: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Save a clinical assessment linked to a patient.

    Parameters
    ----------
    patient_id    : str  — must exist in patients table
    risk_result   : RiskResult.to_dict()
    ai_response   : AIResponse.to_dict()
    similar_cases : list of SimilarCase.to_dict()

    Returns
    -------
    assessment_id : str
    """
    init_db()

    assessment_id  = _generate_id("asmnt_")
    risk_level     = risk_result.get("risk_level", "Unknown") if risk_result else "Unknown"
    confidence_pct = risk_result.get("confidence_pct", 0.0)   if risk_result else 0.0
    disease_prob   = risk_result.get("disease_prob", 0.0)      if risk_result else 0.0
    ai_source      = ai_response.get("source", "unknown")      if ai_response else "none"
    now            = _now_iso()

    ai_json      = json.dumps(ai_response or {})
    similar_json = json.dumps(similar_cases or [])

    sql = """
    INSERT INTO assessments
        (assessment_id, patient_id, risk_level, confidence_pct, disease_prob,
         ai_source, ai_diagnosis, similar_cases, created_at, synced)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
    """

    with _get_connection() as conn:
        conn.execute(sql, (
            assessment_id, patient_id, risk_level, confidence_pct, disease_prob,
            ai_source, ai_json, similar_json, now,
        ))
        conn.commit()

    logger.info(
        "Assessment saved: id=%s patient=%s risk=%s",
        assessment_id, patient_id, risk_level,
    )
    return assessment_id


def get_assessments(
    patient_id: str,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """
    Retrieve all assessments for a patient, newest first.

    Returns
    -------
    List of assessment dicts with expanded JSON fields.
    """
    init_db()

    sql = """
    SELECT * FROM assessments
    WHERE patient_id = ?
    ORDER BY created_at DESC
    LIMIT ?
    """

    with _get_connection() as conn:
        rows = conn.execute(sql, (patient_id, limit)).fetchall()

    return [_deserialise_assessment_row(_row_to_dict(r)) for r in rows]


def get_latest_assessment(patient_id: str) -> Optional[Dict[str, Any]]:
    """Return the most recent assessment for a patient, or None."""
    assessments = get_assessments(patient_id, limit=1)
    return assessments[0] if assessments else None


# ─────────────────────────────────────────────
# Sync support
# ─────────────────────────────────────────────

def get_unsynced_records() -> Dict[str, List[Dict[str, Any]]]:
    """
    Return all records not yet pushed to the cloud.

    Returns
    -------
    dict:
        patients    : list of unsynced patient dicts
        assessments : list of unsynced assessment dicts
    """
    init_db()

    with _get_connection() as conn:
        patient_rows = conn.execute(
            "SELECT * FROM patients WHERE synced = 0 ORDER BY created_at ASC"
        ).fetchall()
        assessment_rows = conn.execute(
            "SELECT * FROM assessments WHERE synced = 0 ORDER BY created_at ASC"
        ).fetchall()

    patients    = [_deserialise_patient_row(_row_to_dict(r)) for r in patient_rows]
    assessments = [_deserialise_assessment_row(_row_to_dict(r)) for r in assessment_rows]

    logger.debug("Unsynced: %d patients, %d assessments", len(patients), len(assessments))
    return {"patients": patients, "assessments": assessments}


def mark_synced(record_id: str, table: str = "patients") -> None:
    """
    Mark a record as successfully synced to the cloud.

    Parameters
    ----------
    record_id : str  — patient_id or assessment_id
    table     : str  — "patients" or "assessments"
    """
    init_db()

    if table == "patients":
        id_col = "patient_id"
    elif table == "assessments":
        id_col = "assessment_id"
    else:
        raise ValueError(f"Unknown table: {table}")

    with _get_connection() as conn:
        conn.execute(
            f"UPDATE {table} SET synced = 1 WHERE {id_col} = ?",
            (record_id,),
        )
        conn.commit()

    logger.debug("Marked synced: table=%s id=%s", table, record_id)


# ─────────────────────────────────────────────
# Analytics + export
# ─────────────────────────────────────────────

def get_stats() -> Dict[str, Any]:
    """
    Return summary statistics about the local database.

    Returns
    -------
    dict:
        total_patients      : int
        total_assessments   : int
        unsynced_patients   : int
        unsynced_assessments: int
        high_risk_count     : int
        medium_risk_count   : int
        low_risk_count      : int
        db_size_kb          : float
    """
    init_db()

    with _get_connection() as conn:
        total_patients       = conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
        total_assessments    = conn.execute("SELECT COUNT(*) FROM assessments").fetchone()[0]
        unsynced_patients    = conn.execute("SELECT COUNT(*) FROM patients    WHERE synced=0").fetchone()[0]
        unsynced_assessments = conn.execute("SELECT COUNT(*) FROM assessments WHERE synced=0").fetchone()[0]
        high_risk   = conn.execute("SELECT COUNT(*) FROM assessments WHERE risk_level='High'").fetchone()[0]
        medium_risk = conn.execute("SELECT COUNT(*) FROM assessments WHERE risk_level='Medium'").fetchone()[0]
        low_risk    = conn.execute("SELECT COUNT(*) FROM assessments WHERE risk_level='Low'").fetchone()[0]

    db_size_kb = os.path.getsize(LOCAL_DB_PATH) / 1024 if os.path.exists(LOCAL_DB_PATH) else 0.0

    return {
        "total_patients":       total_patients,
        "total_assessments":    total_assessments,
        "unsynced_patients":    unsynced_patients,
        "unsynced_assessments": unsynced_assessments,
        "high_risk_count":      high_risk,
        "medium_risk_count":    medium_risk,
        "low_risk_count":       low_risk,
        "db_size_kb":           round(db_size_kb, 2),
    }


def export_to_csv(filepath: Optional[str] = None) -> str:
    """
    Export all patient records (with latest risk assessment) to CSV.

    Parameters
    ----------
    filepath : str  — output path (default: data/exports/patients_export.csv)

    Returns
    -------
    Absolute path to the exported CSV file.
    """
    init_db()

    if filepath is None:
        export_dir = Path("data/exports")
        export_dir.mkdir(parents=True, exist_ok=True)
        timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath   = str(export_dir / f"patients_export_{timestamp}.csv")

    patients = get_all_patients(limit=10_000)

    if not patients:
        logger.warning("No patients to export")
        return filepath

    # Flatten for CSV — include latest risk level
    rows = []
    for p in patients:
        latest = get_latest_assessment(p.get("patient_id", ""))
        row = {
            "patient_id":   p.get("patient_id"),
            "name":         p.get("name"),
            "age":          p.get("age"),
            "sex":          p.get("sex"),
            "created_at":   p.get("created_at"),
            "risk_level":   latest.get("risk_level")   if latest else "N/A",
            "confidence":   latest.get("confidence_pct") if latest else "N/A",
            "ai_source":    latest.get("ai_source")    if latest else "N/A",
        }
        # Add clinical features
        for feature in ["cp", "trestbps", "chol", "fbs", "restecg",
                         "thalach", "exang", "oldpeak", "slope", "ca", "thal"]:
            row[feature] = p.get(feature, "N/A")
        rows.append(row)

    fieldnames = list(rows[0].keys()) if rows else []

    with open(filepath, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Exported %d patients to %s", len(rows), filepath)
    return filepath


# ─────────────────────────────────────────────
# Module init
# ─────────────────────────────────────────────
try:
    init_db()
except Exception as _exc:
    logger.warning("Could not initialise local DB on import: %s", _exc)