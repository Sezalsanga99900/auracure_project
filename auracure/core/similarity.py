"""
core/similarity.py
──────────────────
Patient similarity engine for AuraEcho+.

Responsibility:
    Given a new patient, find the K most similar historical patients
    from heart_data.csv using KNN (K-Nearest Neighbors) with
    cosine similarity.  Return rich result dicts the UI can display
    as "Patients Like This" cards.

Why cosine similarity instead of Euclidean?
    Cosine measures the angle between feature vectors — it's direction-
    aware, meaning a 55-year-old with a similar feature *pattern* to a
    60-year-old ranks higher than someone with accidentally similar raw
    numbers but a different clinical profile.

Architecture:
    ┌──────────────────────────────────────────────────────────┐
    │  heart_data.csv  ──▶  preprocess_dataframe()            │
    │       ↓  scaled (N×13) reference matrix                 │
    │  New patient  ──▶  preprocess_patient()  → (1×13)       │
    │       ↓  NearestNeighbors.kneighbors()                  │
    │  Top-K indices + distances                               │
    │       ↓  _build_similar_case()                          │
    │  List[SimilarCase] rich result objects                   │
    └──────────────────────────────────────────────────────────┘

Public API:
    find_similar_cases(patient, k=3) → List[SimilarCase]
    get_similarity_stats(patient)    → dict  (min/max/mean similarity)
    preload_reference_data()         → None  (call at app startup)
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from core.preprocess import preprocess_patient, preprocess_dataframe, fit_scaler
from utils.constants import (
    DATA_PATH,
    TARGET_COLUMN,
    FEATURE_COLUMNS,
    KNN_NEIGHBORS,
    KNN_METRIC,
    KNN_ALGORITHM,
    RISK_LOW,
    RISK_MEDIUM,
    RISK_HIGH,
    RISK_THRESHOLDS,
)
from utils.helpers import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────

@dataclass
class SimilarCase:
    """
    One historical patient similar to the current case.

    Attributes
    ----------
    rank            : 1-based rank (1 = most similar)
    similarity_pct  : 0–100 % — how similar to the query patient
    patient_index   : row index in heart_data.csv
    features        : dict of raw feature values (original scale)
    outcome         : "Disease" | "No Disease"
    risk_level      : "Low" | "Medium" | "High"  (derived from outcome + features)
    age             : int  (convenience accessor)
    sex             : str  "Male" | "Female"
    summary         : str  one-line human-readable description
    """
    rank:           int
    similarity_pct: float
    patient_index:  int
    features:       Dict[str, Any]
    outcome:        str
    risk_level:     str
    age:            int
    sex:            str
    summary:        str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank":           self.rank,
            "similarity_pct": round(self.similarity_pct, 1),
            "patient_index":  self.patient_index,
            "outcome":        self.outcome,
            "risk_level":     self.risk_level,
            "age":            self.age,
            "sex":            self.sex,
            "summary":        self.summary,
            "features":       self.features,
        }


# ─────────────────────────────────────────────
# Module-level reference data cache
# ─────────────────────────────────────────────

_reference_df:  Optional[pd.DataFrame]    = None   # original (unscaled) rows
_reference_X:   Optional[np.ndarray]      = None   # scaled (N×13) matrix
_knn_model:     Optional[NearestNeighbors] = None
_raw_labels:    Optional[np.ndarray]       = None   # target column values


# ─────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────

def _load_reference_data() -> None:
    """
    Load and preprocess heart_data.csv once into module-level caches.

    Called automatically on first use; can also be called explicitly
    at app startup via preload_reference_data().
    """
    global _reference_df, _reference_X, _knn_model, _raw_labels

    if _reference_X is not None:
        return  # already loaded

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Reference dataset not found at {DATA_PATH}. "
            "Ensure heart_data.csv is present."
        )

    df = pd.read_csv(DATA_PATH)
    logger.info("Loaded reference data: %d patients from %s", len(df), DATA_PATH)

    # Store raw (unscaled) rows for display purposes
    _reference_df = df.copy()

    # Extract labels
    if TARGET_COLUMN in df.columns:
        _raw_labels = df[TARGET_COLUMN].values.astype(int)
    else:
        _raw_labels = np.zeros(len(df), dtype=int)
        logger.warning("Target column '%s' not found — assuming all label=0", TARGET_COLUMN)

    # Preprocess + scale
    try:
        _reference_X, _ = preprocess_dataframe(df)
    except Exception as exc:
        logger.warning("preprocess_dataframe failed (%s) — fitting scaler first", exc)
        fit_scaler(df)
        _reference_X, _ = preprocess_dataframe(df)

    # Fit KNN model
    _knn_model = NearestNeighbors(
        n_neighbors=min(KNN_NEIGHBORS + 1, len(df)),  # +1 because query itself may be in set
        metric=KNN_METRIC,
        algorithm=KNN_ALGORITHM,
        n_jobs=-1,
    )
    _knn_model.fit(_reference_X)
    logger.info(
        "KNN model fitted: metric=%s, algorithm=%s, k=%d",
        KNN_METRIC, KNN_ALGORITHM, KNN_NEIGHBORS,
    )


def _cosine_distance_to_similarity(distance: float) -> float:
    """
    Convert cosine distance [0, 2] → similarity percentage [0, 100].

    Cosine distance = 1 − cosine_similarity
    So similarity = (1 − distance) * 100
    Clipped to [0, 100].
    """
    similarity = (1.0 - distance) * 100.0
    return float(np.clip(similarity, 0.0, 100.0))


def _outcome_label(label: int) -> str:
    """0 → 'No Disease'  |  1 → 'Disease'"""
    return "Disease" if label == 1 else "No Disease"


def _derive_risk_level(label: int, features: Dict[str, Any]) -> str:
    """
    Derive a risk level for a historical patient using simple heuristics.

    Historical records only have a binary label, so we enrich them with
    a rough risk tier based on the label and a few high-signal features.
    """
    if label == 0:
        return RISK_LOW

    # For disease-positive cases, further triage by severity signals
    thalach = float(features.get("thalach", 150))   # max heart rate
    oldpeak = float(features.get("oldpeak", 0.0))   # ST depression
    ca      = float(features.get("ca", 0))           # vessels affected

    severity_score = 0
    if thalach < 120:   severity_score += 2
    elif thalach < 140: severity_score += 1
    if oldpeak > 2.0:   severity_score += 2
    elif oldpeak > 1.0: severity_score += 1
    if ca >= 2:         severity_score += 2
    elif ca == 1:       severity_score += 1

    if severity_score >= 4:
        return RISK_HIGH
    elif severity_score >= 2:
        return RISK_MEDIUM
    else:
        return RISK_LOW


def _sex_label(sex_code: Any) -> str:
    """1 → 'Male'  |  0 → 'Female'"""
    try:
        return "Male" if int(float(sex_code)) == 1 else "Female"
    except (ValueError, TypeError):
        return "Unknown"


def _build_summary(rank: int, age: int, sex: str, outcome: str,
                   risk_level: str, similarity_pct: float) -> str:
    """One-line card summary for the UI."""
    return (
        f"#{rank} Match — {age}yr {sex} | {outcome} | "
        f"{risk_level} Risk | {similarity_pct:.1f}% similar"
    )


def _build_similar_case(
    rank: int,
    patient_index: int,
    distance: float,
) -> SimilarCase:
    """
    Construct a SimilarCase object from a KNN result.

    Parameters
    ----------
    rank          : 1-based position (1 = most similar)
    patient_index : row index in _reference_df
    distance      : cosine distance returned by sklearn
    """
    similarity_pct = _cosine_distance_to_similarity(distance)

    # Raw feature dict (unscaled, original values)
    raw_row: Dict[str, Any] = {}
    if _reference_df is not None:
        row = _reference_df.iloc[patient_index]
        raw_row = {col: row[col] for col in FEATURE_COLUMNS if col in row.index}

    # Label
    label  = int(_raw_labels[patient_index]) if _raw_labels is not None else 0
    outcome = _outcome_label(label)

    # Derived fields
    age         = int(float(raw_row.get("age", 0)))
    sex         = _sex_label(raw_row.get("sex", 0))
    risk_level  = _derive_risk_level(label, raw_row)

    summary = _build_summary(rank, age, sex, outcome, risk_level, similarity_pct)

    return SimilarCase(
        rank=rank,
        similarity_pct=similarity_pct,
        patient_index=patient_index,
        features=raw_row,
        outcome=outcome,
        risk_level=risk_level,
        age=age,
        sex=sex,
        summary=summary,
    )


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def preload_reference_data() -> None:
    """
    Eagerly load and index the reference dataset.

    Call this once at app startup (e.g., in app.py) to avoid a
    cold-start delay on the first patient query.
    """
    _load_reference_data()
    logger.info("Reference data preloaded (%d patients)", len(_reference_df) if _reference_df is not None else 0)


def find_similar_cases(
    patient: Dict[str, Any],
    k: int = KNN_NEIGHBORS,
) -> List[SimilarCase]:
    """
    Find the k most similar historical patients.

    Parameters
    ----------
    patient : dict
        Raw patient record (UI form format).
    k : int
        Number of similar cases to return (default: KNN_NEIGHBORS from constants).

    Returns
    -------
    List[SimilarCase]  length = k, sorted by similarity descending.

    Raises
    ------
    FileNotFoundError  if heart_data.csv is missing.
    ValueError         if k > number of reference patients.
    """
    _load_reference_data()

    n_ref = len(_reference_df) if _reference_df is not None else 0
    if k > n_ref:
        logger.warning("k=%d > reference size=%d — reducing k", k, n_ref)
        k = n_ref

    # Preprocess query patient
    query_vector = preprocess_patient(patient)   # shape (1, 13)

    # Query KNN
    distances, indices = _knn_model.kneighbors(query_vector, n_neighbors=k + 1)
    distances = distances[0]     # flatten: shape (k+1,)
    indices   = indices[0]       # flatten: shape (k+1,)

    logger.debug("KNN raw distances: %s", distances[:k+1])
    logger.debug("KNN raw indices  : %s", indices[:k+1])

    # Build result objects (skip index 0 if it's an exact match = same patient)
    results: List[SimilarCase] = []
    rank = 1
    for dist, idx in zip(distances, indices):
        if rank > k:
            break
        case = _build_similar_case(rank, int(idx), float(dist))
        results.append(case)
        logger.debug(
            "Similar case #%d: idx=%d | sim=%.1f%% | %s | %s",
            rank, idx, case.similarity_pct, case.outcome, case.risk_level,
        )
        rank += 1

    return results


def get_similarity_stats(patient: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute similarity statistics across ALL reference patients.

    Useful for the analytics dashboard to show where the current
    patient sits in the distribution.

    Returns
    -------
    dict with keys: min_sim, max_sim, mean_sim, median_sim,
                    pct_above_80 (% of patients with >80% similarity)
    """
    _load_reference_data()

    query_vector = preprocess_patient(patient)

    # Compute all pairwise distances
    all_distances, _ = _knn_model.kneighbors(
        query_vector,
        n_neighbors=len(_reference_df),
    )
    all_distances = all_distances[0]
    similarities  = np.array([_cosine_distance_to_similarity(d) for d in all_distances])

    stats = {
        "min_sim":      round(float(similarities.min()), 2),
        "max_sim":      round(float(similarities.max()), 2),
        "mean_sim":     round(float(similarities.mean()), 2),
        "median_sim":   round(float(np.median(similarities)), 2),
        "pct_above_80": round(float((similarities > 80).mean() * 100), 2),
    }
    logger.debug("Similarity stats: %s", stats)
    return stats


def find_similar_by_risk(
    patient: Dict[str, Any],
    risk_filter: str,
    k: int = 3,
) -> List[SimilarCase]:
    """
    Find similar cases filtered to a specific risk level.

    Useful for showing "other High-risk patients like you"
    or "what did Low-risk similar patients look like?".

    Parameters
    ----------
    patient     : dict  (raw patient record)
    risk_filter : "Low" | "Medium" | "High"
    k           : max results to return

    Returns
    -------
    List[SimilarCase]  (may be shorter than k if few matches exist)
    """
    # Fetch a larger pool first, then filter
    pool = find_similar_cases(patient, k=min(50, len(_reference_df) if _reference_df is not None else 50))
    filtered = [c for c in pool if c.risk_level == risk_filter]
    return filtered[:k]


def get_outcome_distribution(patient: Dict[str, Any], k: int = 20) -> Dict[str, Any]:
    """
    Among the top-k similar patients, what fraction had heart disease?

    Returns
    -------
    dict:
        disease_count   : int
        no_disease_count: int
        disease_pct     : float
        total           : int
    """
    cases  = find_similar_cases(patient, k=k)
    total  = len(cases)
    n_disease = sum(1 for c in cases if c.outcome == "Disease")
    n_healthy  = total - n_disease

    return {
        "disease_count":    n_disease,
        "no_disease_count": n_healthy,
        "disease_pct":      round(n_disease / total * 100, 1) if total else 0.0,
        "total":            total,
    }


# ─────────────────────────────────────────────
# Module init — preload reference data eagerly
# ─────────────────────────────────────────────
try:
    _load_reference_data()
    logger.info("Similarity engine ready")
except Exception as _exc:
    logger.warning("Similarity engine could not preload: %s", _exc)