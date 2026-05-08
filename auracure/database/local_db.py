# =============================================================================
# database/local_db.py
# AuraEcho+ — SQLite Local Storage Engine
#
# Responsibility:
#     Manage local persistence of patients, predictions, and the sync queue.
#     Provides offline-first storage that syncs to cloud when online.
#
# Schema:
#     patients      — patient demographics + feature JSON
#     predictions   — risk results + AI responses linked to patients
#     sync_queue    — pending operations for cloud sync
#
# Public API:
#     init_db()                     → None
#     save_patient(patient)         → int (patient_id)
#     get_patient(patient_id)       → dict
#     get_all_patients()            → list[dict]
#     save_prediction(patient_id, risk, ai) → int
#     get_patient_history(patient_id) → list[dict]
#     add_to_sync_queue(table, record_id, op) → None
#     get_pending_sync(limit)       → list[dict]
#     mark_synced(queue_id)         → None
#     export_to_csv()               → str (path)
# =============================================================================

import os
import json
import sqlite3
import threading
import pandas as pd
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from utils.constants import (
    LOCAL_DB_PATH,
    LOCAL_CSV_BACKUP_PATH,
    DB_TABLE_PATIENTS,
    DB_TABLE_PREDICTIONS,
    DB_TABLE_SYNC_QUEUE,
    DATE_FORMAT,
    SYNC_BATCH_SIZE,
)
from utils.helpers import get_logger, ensure_dir, now_str

logger = get_logger(__name__)

# Thread lock for write operations (SQLite is thread-safe for reads,
# but concurrent writes need serialization in Streamlit)
_db_lock = threading.Lock()


# ─────────────────────────────────────────────
# Connection helper
# ─────────────────────────────────────────────

def _get_connection() -> sqlite3.Connection:
    """
    Get a database connection with row factory and foreign keys enabled.
    Creates the database file if it doesn't exist.
    """
    ensure_dir(os.path.dirname(LOCAL_DB_PATH))
    conn = sqlite3.connect(LOCAL_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# ─────────────────────────────────────────────
# Schema initialization
# ─────────────────────────────────────────────

def init_db() -> None:
    """
    Create all tables if they don't exist.
    Call this once at app startup.
    """
    with _db_lock:
        conn = _get_connection()
        cursor = conn.cursor()

        # Patients table
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {DB_TABLE_PATIENTS} (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL,
                age         INTEGER,
                sex         INTEGER,
                features    TEXT NOT NULL,          -- JSON blob of all clinical features
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            )
        """)

        # Predictions table
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {DB_TABLE_PREDICTIONS} (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id      INTEGER NOT NULL,
                risk_level      TEXT NOT NULL,
                risk_score      REAL NOT NULL,
                predicted_label INTEGER NOT NULL,
                risk_result     TEXT NOT NULL,      -- JSON blob of RiskResult
                ai_response     TEXT,               -- JSON blob of AIResponse (nullable)
                created_at      TEXT NOT NULL,
                FOREIGN KEY (patient_id) REFERENCES {DB_TABLE_PATIENTS}(id) ON DELETE CASCADE
            )
        """)

        # Sync queue table
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {DB_TABLE_SYNC_QUEUE} (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                table_name  TEXT NOT NULL,          -- 'patients' or 'predictions'
                record_id   INTEGER NOT NULL,       -- ID in the source table
                operation   TEXT NOT NULL,          -- 'INSERT' | 'UPDATE' | 'DELETE'
                status      TEXT NOT NULL DEFAULT 'PENDING',
                retry_count INTEGER DEFAULT 0,
                created_at  TEXT NOT NULL,
                synced_at   TEXT,
                error_msg   TEXT
            )
        """)

        # Indexes for performance
        cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_pred_patient ON {DB_TABLE_PREDICTIONS}(patient_id)")
        cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_sync_status ON {DB_TABLE_SYNC_QUEUE}(status)")

        conn.commit()
        conn.close()
        logger.info("Database initialized at %s", LOCAL_DB_PATH)


# ─────────────────────────────────────────────
# Patient operations
# ─────────────────────────────────────────────

def save_patient(patient: Dict[str, Any]) -> int:
    """
    Insert a new patient record.

    Parameters
    ----------
    patient : dict with keys: name, age, sex, and all FEATURE_COLUMNS

    Returns
    -------
    patient_id : int
    """
    with _db_lock:
        conn = _get_connection()
        cursor = conn.cursor()

        now = now_str()
        name = patient.get("name", "Unknown")
        age = patient.get("age")
        sex = patient.get("sex")

        # Store all features as JSON (flexible schema)
        features_json = json.dumps(patient, default=str)

        cursor.execute(f"""
            INSERT INTO {DB_TABLE_PATIENTS}
            (name, age, sex, features, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (name, age, sex, features_json, now, now))

        patient_id = cursor.lastrowid

        # Add to sync queue
        _add_to_sync_queue_internal(cursor, DB_TABLE_PATIENTS, patient_id, "INSERT")

        conn.commit()
        conn.close()
        logger.info("Saved patient id=%d name='%s'", patient_id, name)
        return patient_id


def get_patient(patient_id: int) -> Optional[Dict[str, Any]]:
    """
    Retrieve a patient by ID.
    Returns None if not found.
    """
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute(f"""
        SELECT * FROM {DB_TABLE_PATIENTS} WHERE id = ?
    """, (patient_id,))
    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None

    patient = dict(row)
    patient["features"] = json.loads(patient["features"])
    return patient


def get_all_patients(limit: int = 100) -> List[Dict[str, Any]]:
    """
    Get all patients ordered by created_at descending.
    """
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute(f"""
        SELECT * FROM {DB_TABLE_PATIENTS}
        ORDER BY created_at DESC
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()

    patients = []
    for row in rows:
        p = dict(row)
        p["features"] = json.loads(p["features"])
        patients.append(p)
    return patients


def search_patients(query: str, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Search patients by name.
    """
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute(f"""
        SELECT * FROM {DB_TABLE_PATIENTS}
        WHERE name LIKE ?
        ORDER BY created_at DESC
        LIMIT ?
    """, (f"%{query}%", limit))
    rows = cursor.fetchall()
    conn.close()

    patients = []
    for row in rows:
        p = dict(row)
        p["features"] = json.loads(p["features"])
        patients.append(p)
    return patients


# ─────────────────────────────────────────────
# Prediction operations
# ─────────────────────────────────────────────

def save_prediction(
    patient_id:    int,
    risk_result:   Dict[str, Any],
    ai_response:   Optional[Dict[str, Any]] = None,
) -> int:
    """
    Save a prediction result for a patient.

    Parameters
    ----------
    patient_id  : int
    risk_result : RiskResult.to_dict()
    ai_response : AIResponse.to_dict() or None

    Returns
    -------
    prediction_id : int
    """
    with _db_lock:
        conn = _get_connection()
        cursor = conn.cursor()

        now = now_str()
        risk_json = json.dumps(risk_result, default=str)
        ai_json   = json.dumps(ai_response, default=str) if ai_response else None

        risk_level   = risk_result.get("risk_level", "UNKNOWN")
        risk_score   = risk_result.get("disease_prob", 0.0)
        pred_label   = risk_result.get("predicted_label", 0)

        cursor.execute(f"""
            INSERT INTO {DB_TABLE_PREDICTIONS}
            (patient_id, risk_level, risk_score, predicted_label,
             risk_result, ai_response, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (patient_id, risk_level, risk_score, pred_label, risk_json, ai_json, now))

        prediction_id = cursor.lastrowid

        # Add to sync queue
        _add_to_sync_queue_internal(cursor, DB_TABLE_PREDICTIONS, prediction_id, "INSERT")

        conn.commit()
        conn.close()
        logger.info(
            "Saved prediction id=%d patient=%d risk=%s",
            prediction_id, patient_id, risk_level,
        )
        return prediction_id


def get_patient_history(patient_id: int) -> List[Dict[str, Any]]:
    """
    Get all predictions for a patient, ordered by created_at descending.
    """
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute(f"""
        SELECT * FROM {DB_TABLE_PREDICTIONS}
        WHERE patient_id = ?
        ORDER BY created_at DESC
    """, (patient_id,))
    rows = cursor.fetchall()
    conn.close()

    history = []
    for row in rows:
        h = dict(row)
        h["risk_result"] = json.loads(h["risk_result"])
        h["ai_response"] = json.loads(h["ai_response"]) if h["ai_response"] else None
        history.append(h)
    return history


def get_recent_predictions(limit: int = 20) -> List[Dict[str, Any]]:
    """
    Get recent predictions across all patients with patient name joined.
    """
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute(f"""
        SELECT p.*, pat.name as patient_name, pat.age, pat.sex
        FROM {DB_TABLE_PREDICTIONS} p
        JOIN {DB_TABLE_PATIENTS} pat ON p.patient_id = pat.id
        ORDER BY p.created_at DESC
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()

    results = []
    for row in rows:
        r = dict(row)
        r["risk_result"] = json.loads(r["risk_result"])
        r["ai_response"] = json.loads(r["ai_response"]) if r["ai_response"] else None
        results.append(r)
    return results


# ─────────────────────────────────────────────
# Sync queue operations
# ─────────────────────────────────────────────

def _add_to_sync_queue_internal(
    cursor:    sqlite3.Cursor,
    table:     str,
    record_id: int,
    operation: str,
) -> None:
    """Internal helper to add to sync queue (assumes cursor is active)."""
    now = now_str()
    cursor.execute(f"""
        INSERT INTO {DB_TABLE_SYNC_QUEUE}
        (table_name, record_id, operation, status, created_at)
        VALUES (?, ?, ?, 'PENDING', ?)
    """, (table, record_id, operation, now))


def add_to_sync_queue(table: str, record_id: int, operation: str) -> None:
    """
    Public API to add an item to the sync queue.
    Used by services that modify data outside the main save functions.
    """
    with _db_lock:
        conn = _get_connection()
        cursor = conn.cursor()
        _add_to_sync_queue_internal(cursor, table, record_id, operation)
        conn.commit()
        conn.close()


def get_pending_sync(limit: int = SYNC_BATCH_SIZE) -> List[Dict[str, Any]]:
    """
    Get pending sync items ordered by created_at.
    """
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute(f"""
        SELECT * FROM {DB_TABLE_SYNC_QUEUE}
        WHERE status = 'PENDING'
        ORDER BY created_at ASC
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def mark_synced(queue_id: int) -> None:
    """
    Mark a sync queue item as synced.
    """
    with _db_lock:
        conn = _get_connection()
        cursor = conn.cursor()
        now = now_str()
        cursor.execute(f"""
            UPDATE {DB_TABLE_SYNC_QUEUE}
            SET status = 'SYNCED', synced_at = ?
            WHERE id = ?
        """, (now, queue_id))
        conn.commit()
        conn.close()


def mark_sync_failed(queue_id: int, error: str) -> None:
    """
    Mark a sync queue item as failed and increment retry count.
    """
    with _db_lock:
        conn = _get_connection()
        cursor = conn.cursor()
        cursor.execute(f"""
            UPDATE {DB_TABLE_SYNC_QUEUE}
            SET status = 'FAILED', retry_count = retry_count + 1, error_msg = ?
            WHERE id = ?
        """, (error, queue_id))
        conn.commit()
        conn.close()


def get_sync_stats() -> Dict[str, int]:
    """
    Get counts of sync queue items by status.
    """
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute(f"""
        SELECT status, COUNT(*) as cnt
        FROM {DB_TABLE_SYNC_QUEUE}
        GROUP BY status
    """)
    rows = cursor.fetchall()
    conn.close()
    return {row["status"]: row["cnt"] for row in rows}


# ─────────────────────────────────────────────
# Export / Backup
# ─────────────────────────────────────────────

def export_to_csv() -> str:
    """
    Export all patients and predictions to a CSV backup file.
    Returns the path to the backup file.
    """
    ensure_dir(os.path.dirname(LOCAL_CSV_BACKUP_PATH))

    conn = _get_connection()
    patients_df = pd.read_sql_query(f"SELECT * FROM {DB_TABLE_PATIENTS}", conn)
    predictions_df = pd.read_sql_query(f"SELECT * FROM {DB_TABLE_PREDICTIONS}", conn)
    conn.close()

    # Expand JSON columns
    if "features" in patients_df.columns:
        features_expanded = pd.json_normalize(patients_df["features"].apply(json.loads))
        patients_df = pd.concat([patients_df.drop(columns=["features"]), features_expanded], axis=1)

    if "risk_result" in predictions_df.columns:
        risk_expanded = pd.json_normalize(predictions_df["risk_result"].apply(json.loads))
        predictions_df = pd.concat([predictions_df.drop(columns=["risk_result"]), risk_expanded], axis=1)

    # Merge
    merged = predictions_df.merge(
        patients_df.add_prefix("patient_"),
        left_on="patient_id",
        right_on="patient_id",
        how="left",
    )

    merged.to_csv(LOCAL_CSV_BACKUP_PATH, index=False)
    logger.info("Exported backup to %s (%d rows)", LOCAL_CSV_BACKUP_PATH, len(merged))
    return LOCAL_CSV_BACKUP_PATH


# ─────────────────────────────────────────────
# Module init
# ─────────────────────────────────────────────

try:
    init_db()
except Exception as _exc:
    logger.error("Failed to initialize database: %s", _exc)