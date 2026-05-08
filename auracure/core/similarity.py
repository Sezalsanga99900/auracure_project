# =============================================================================
# core/similarity.py
# AuraEcho+ — Patient Similarity Engine (KNN with Cosine Similarity)
#
# Responsibility:
#     Given a new patient, find the K most similar historical patients
#     from heart_data.csv using KNN with cosine similarity.
#     Return rich result dicts the UI can display as "Patients Like This" cards.
#
# Public API:
#     find_similar_cases(patient, k)        → List[SimilarCase]
#     get_similarity_stats(patient)         → dict
#     find_similar_by_risk(patient, filter) → List[SimilarCase]
#     get_outcome_distribution(patient, k)  → dict
#     get_similar_cases_summary(patient, k) → dict
#     compare_patients(patient_a, patient_b)→ dict
#     preload_reference_data()              → None
# =============================================================================

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from core.preprocess import preprocess_patient, preprocess_dataframe, fit_scaler
from utils.constants import (
    HEART_DATA_PATH,
    TARGET_COLUMN,
    FEATURE_COLUMNS,
    FEATURE_LABELS,
    KNN_N_NEIGHBORS,
    KNN_TOP_DISPLAY,
    KNN_METRIC,
    KNN_ALGORITHM,
    SIMILARITY_SCORE_MIN,
    SIMILARITY_SCORE_MAX,
    SIMILARITY_POOL_SIZE,
    RISK_LEVELS,
    RISK_LABELS,
    RISK_COLORS,
    RISK_ICONS,
)
from utils.helpers import get_logger
from utils.validators import validate_patient, errors_to_str, format_validation_errors

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
    similarity_pct  : 0–100 %
    patient_index   : row index in heart_data.csv
    features        : dict of raw feature values (original scale)
    outcome         : "Disease" | "No Disease"
    risk_level      : "LOW" | "MEDIUM" | "HIGH"  (KEY not label)
    age             : int
    sex             : str  "Male" | "Female" | "Unknown"
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
            "risk_label":     RISK_LABELS.get(self.risk_level, self.risk_level),
            "risk_color":     RISK_COLORS.get(self.risk_level, "#95a5a6"),
            "risk_icon":      RISK_ICONS.get(self.risk_level, "❔"),
            "age":            self.age,
            "sex":            self.sex,
            "summary":        self.summary,
            "features":       self.features,
        }


# ─────────────────────────────────────────────
# Module-level reference data cache
# ─────────────────────────────────────────────

_reference_df:  Optional[pd.DataFrame]     = None
_reference_X:   Optional[np.ndarray]       = None
_knn_model:     Optional[NearestNeighbors] = None
_raw_labels:    Optional[np.ndarray]       = None


# ─────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────

def _load_reference_data() -> None:
    """
    Load and preprocess heart_data.csv once into module-level caches.
    Called automatically on first use or explicitly at startup.
    """
    global _reference_df, _reference_X, _knn_model, _raw_labels

    if _reference_X is not None:
        return   # already loaded

    if not os.path.exists(HEART_DATA_PATH):
        raise FileNotFoundError(
            f"Reference dataset not found at {HEART_DATA_PATH}. "
            "Ensure heart_data.csv is present."
        )

    df = pd.read_csv(HEART_DATA_PATH)
    logger.info(
        "Loaded reference data: %d patients from %s",
        len(df), HEART_DATA_PATH,
    )

    _reference_df = df.copy()

    if TARGET_COLUMN in df.columns:
        _raw_labels = df[TARGET_COLUMN].values.astype(int)
    else:
        _raw_labels = np.zeros(len(df), dtype=int)
        logger.warning(
            "Target column '%s' not found — assuming all label=0",
            TARGET_COLUMN,
        )

    # Preprocess + scale
    try:
        _reference_X, _ = preprocess_dataframe(df)
    except Exception as exc:
        logger.warning(
            "preprocess_dataframe failed (%s) — fitting scaler first", exc
        )
        fit_scaler(df)
        _reference_X, _ = preprocess_dataframe(df)

    # Fit KNN model
    n_neighbors = min(KNN_N_NEIGHBORS + 1, len(df))
    _knn_model = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric=KNN_METRIC,
        algorithm=KNN_ALGORITHM,
        n_jobs=-1,
    )
    _knn_model.fit(_reference_X)
    logger.info(
        "KNN model fitted: metric=%s, algorithm=%s, k=%d",
        KNN_METRIC, KNN_ALGORITHM, KNN_N_NEIGHBORS,
    )


def _cosine_distance_to_similarity(distance: float) -> float:
    """
    Convert cosine distance [0, 2] → similarity percentage [0, 100].
    similarity = (1 − distance) * 100, clipped to [0, 100].
    """
    similarity = (1.0 - distance) * 100.0
    return float(np.clip(similarity, SIMILARITY_SCORE_MIN, SIMILARITY_SCORE_MAX))


def _outcome_label(label: int) -> str:
    """0 → 'No Disease'  |  1 → 'Disease'"""
    return "Disease" if label == 1 else "No Disease"


def _derive_risk_level(label: int, features: Dict[str, Any]) -> str:
    """
    Derive a risk level KEY for a historical patient.

    FIXED: Dead code removed — RISK_LEVELS thresholds are now
           actually used via normalized severity score.

    Returns "LOW" | "MEDIUM" | "HIGH"
    """
    if label == 0:
        return "LOW"

    thalach = float(features.get("thalach", 150))
    oldpeak = float(features.get("oldpeak", 0.0))
    ca      = float(features.get("ca", 0))

    severity_score = 0
    if thalach < 120:    severity_score += 2
    elif thalach < 140:  severity_score += 1
    if oldpeak > 2.0:    severity_score += 2
    elif oldpeak > 1.0:  severity_score += 1
    if ca >= 2:          severity_score += 2
    elif ca == 1:        severity_score += 1

    # FIXED: Normalize score to [0, 1] and use RISK_LEVELS thresholds
    max_score  = 6
    normalized = severity_score / max_score

    for level in ["LOW", "MEDIUM", "HIGH"]:
        lo, hi = RISK_LEVELS[level]
        if lo <= normalized < hi:
            return level
    return "HIGH"


def _sex_label(sex_code: Any) -> str:
    """Convert 1/0 → 'Male'/'Female'."""
    try:
        val = int(float(sex_code))
        if val == 1:
            return "Male"
        elif val == 0:
            return "Female"
        return "Unknown"
    except (ValueError, TypeError):
        return "Unknown"


def _build_summary(
    rank:           int,
    age:            int,
    sex:            str,
    outcome:        str,
    risk_level:     str,
    similarity_pct: float,
) -> str:
    """
    One-line card summary for the UI.

    FIXED: Removed double 'Risk' text.
           RISK_LABELS already contains 'Risk' in the label.
    """
    risk_label_display = RISK_LABELS.get(risk_level, risk_level)
    return (
        f"#{rank} Match — {age}yr {sex} | {outcome} | "
        f"{risk_label_display} | {similarity_pct:.1f}% similar"
    )


def _build_similar_case(
    rank:          int,
    patient_index: int,
    distance:      float,
) -> SimilarCase:
    """
    Construct a SimilarCase object from a KNN result.
    """
    similarity_pct = _cosine_distance_to_similarity(distance)

    raw_row: Dict[str, Any] = {}
    if _reference_df is not None:
        row     = _reference_df.iloc[patient_index]
        raw_row = {
            col: row[col]
            for col in FEATURE_COLUMNS
            if col in row.index
        }

    label   = int(_raw_labels[patient_index]) if _raw_labels is not None else 0
    outcome = _outcome_label(label)

    age        = int(float(raw_row.get("age", 0)))
    sex        = _sex_label(raw_row.get("sex", 0))
    risk_level = _derive_risk_level(label, raw_row)
    summary    = _build_summary(rank, age, sex, outcome, risk_level, similarity_pct)

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
    Eagerly load and index the reference dataset at app startup.
    """
    _load_reference_data()
    n = len(_reference_df) if _reference_df is not None else 0
    logger.info("Reference data preloaded (%d patients)", n)


def find_similar_cases(
    patient: Dict[str, Any],
    k: int = KNN_N_NEIGHBORS,
) -> List[SimilarCase]:
    """
    Find the k most similar historical patients.

    FIXED:
    - validate_patient() called with correct signature
    - preprocess_patient() called without invalid validate=False param
    - Self-skip logic fixed (rank gap bug removed)
    - f-string anti-pattern removed from debug log

    Parameters
    ----------
    patient : dict — raw patient record (UI form format)
    k       : number of similar cases to return

    Returns
    -------
    List[SimilarCase] sorted by similarity descending.
    """
    # Validate input — uses check_structure=True (ADDED to validators.py)
    ok, errors = validate_patient(patient, check_structure=True)
    if not ok:
        raise ValueError(
            f"Invalid patient input for similarity search:\n"
            f"{format_validation_errors(errors)}"
        )

    _load_reference_data()

    n_ref = len(_reference_df) if _reference_df is not None else 0
    if k > n_ref:
        logger.warning("k=%d > reference size=%d — reducing k", k, n_ref)
        k = n_ref

    # Preprocess query patient
    # FIXED: no validate=False parameter (not in preprocess_patient signature)
    query_vector = preprocess_patient(patient, validate=True)

    # Fetch k+1 neighbors to allow for self-skip
    n_query = min(k + 1, n_ref)
    distances, indices = _knn_model.kneighbors(
        query_vector, n_neighbors=n_query
    )
    distances = distances[0]
    indices   = indices[0]

    logger.debug("KNN raw distances: %s", distances.tolist())
    logger.debug("KNN raw indices  : %s", indices.tolist())

    # FIXED: self-skip logic — no rank gap, no f-string anti-pattern
    results: List[SimilarCase] = []
    rank = 1

    for dist, idx in zip(distances, indices):
        if rank > k:
            break
        # Skip exact match (query patient found in reference set)
        if float(dist) < 1e-9:
            logger.debug("Skipping self-match at index %d", idx)
            continue   # do NOT increment rank

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

    Returns
    -------
    dict: min_sim, max_sim, mean_sim, median_sim, pct_above_80
    """
    _load_reference_data()

    ok, errors = validate_patient(patient, check_structure=True)
    if not ok:
        raise ValueError(
            f"Invalid patient input for stats:\n{errors_to_str(errors)}"
        )

    # FIXED: no validate=False
    query_vector = preprocess_patient(patient, validate=True)

    n_ref = len(_reference_df) if _reference_df is not None else 0

    # NOTE: Queries all N reference patients for full distribution stats.
    # Acceptable for Cleveland dataset (N=303).
    all_distances, _ = _knn_model.kneighbors(
        query_vector, n_neighbors=n_ref
    )
    all_distances = all_distances[0]
    similarities  = np.array([
        _cosine_distance_to_similarity(d) for d in all_distances
    ])

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
    patient:     Dict[str, Any],
    risk_filter: str,
    k:           int = KNN_TOP_DISPLAY,
) -> List[SimilarCase]:
    """
    Find similar cases filtered to a specific risk level KEY.

    FIXED: Uses SIMILARITY_POOL_SIZE constant (was hardcoded 50).

    Parameters
    ----------
    patient     : dict
    risk_filter : "LOW" | "MEDIUM" | "HIGH"
    k           : max results to return
    """
    if risk_filter not in RISK_LEVELS:
        raise ValueError(
            f"risk_filter must be one of {list(RISK_LEVELS.keys())}, "
            f"got {risk_filter!r}"
        )

    n_ref = len(_reference_df) if _reference_df is not None else SIMILARITY_POOL_SIZE
    # FIXED: use SIMILARITY_POOL_SIZE constant
    pool     = find_similar_cases(patient, k=min(SIMILARITY_POOL_SIZE, n_ref))
    filtered = [c for c in pool if c.risk_level == risk_filter]
    return filtered[:k]


def get_outcome_distribution(
    patient: Dict[str, Any],
    k: int = 20,
) -> Dict[str, Any]:
    """
    Among the top-k similar patients, what fraction had heart disease?

    Returns
    -------
    dict: disease_count, no_disease_count, disease_pct, total
    """
    cases     = find_similar_cases(patient, k=k)
    total     = len(cases)
    n_disease = sum(1 for c in cases if c.outcome == "Disease")
    n_healthy = total - n_disease

    return {
        "disease_count":    n_disease,
        "no_disease_count": n_healthy,
        "disease_pct":      round(n_disease / total * 100, 1) if total else 0.0,
        "total":            total,
    }


def get_similar_cases_summary(
    patient: Dict[str, Any],
    k: int = KNN_TOP_DISPLAY,
) -> Dict[str, Any]:
    """
    ADDED: High-level summary of similar cases for the results panel.
    Returns aggregated stats + top-k cases in one call.
    Used by ui/results_panel.py as a single API call.
    """
    cases        = find_similar_cases(patient, k=k)
    outcome_dist = get_outcome_distribution(patient, k=k)

    avg_sim = (
        round(sum(c.similarity_pct for c in cases) / len(cases), 1)
        if cases else 0.0
    )

    return {
        "cases":            [c.to_dict() for c in cases],
        "outcome_dist":     outcome_dist,
        "top_match_sim":    cases[0].similarity_pct if cases else 0.0,
        "avg_similarity":   avg_sim,
        "dominant_outcome": (
            "Disease" if outcome_dist["disease_pct"] >= 50
            else "No Disease"
        ),
    }


def compare_patients(
    patient_a: Dict[str, Any],
    patient_b: Dict[str, Any],
) -> Dict[str, Any]:
    """
    ADDED: Compute cosine similarity between two specific patients.
    Useful for doctor's side-by-side comparison view.

    Returns
    -------
    dict: similarity_pct, interpretation
    """
    from sklearn.metrics.pairwise import cosine_similarity

    _load_reference_data()

    vec_a = preprocess_patient(patient_a, validate=True)
    vec_b = preprocess_patient(patient_b, validate=True)

    sim     = float(cosine_similarity(vec_a, vec_b)[0][0])
    sim_pct = round(sim * 100, 1)

    interpretation = (
        "Very similar profiles"      if sim_pct >= 80
        else "Moderately similar"    if sim_pct >= 60
        else "Different clinical profiles"
    )

    return {
        "similarity_pct":  sim_pct,
        "interpretation":  interpretation,
    }


# ─────────────────────────────────────────────
# Module init — preload reference data eagerly
# ─────────────────────────────────────────────
try:
    _load_reference_data()
    logger.info(
        "Similarity engine ready (k=%d, metric=%s)",
        KNN_N_NEIGHBORS, KNN_METRIC,
    )
except Exception as _exc:
    logger.warning("Similarity engine could not preload: %s", _exc)