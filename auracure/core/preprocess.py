"""
core/preprocess.py
──────────────────
Patient data preprocessing pipeline for AuraEcho+.

Responsibility:
    Transform raw patient dictionaries (as submitted from the UI form)
    into clean, scaled NumPy arrays ready for the risk model and the
    similarity engine.

Pipeline stages:
    1. Type coercion   — cast strings → int/float
    2. Missing-value   — fill with column medians / mode
    3. Encoding        — map categorical labels → integers
    4. Range clamping  — clip values to clinically valid bounds
    5. Scaling         — MinMax scale to [0, 1] per feature
    6. Column ordering — guarantee consistent feature vector order

Public API:
    preprocess_patient(patient_dict)  → np.ndarray  shape (1, 13)
    preprocess_dataframe(df)          → np.ndarray  shape (N, 13)
    get_feature_names()               → List[str]
    inverse_transform(array)          → np.ndarray  (unscale)
    fit_scaler(df)                    → fitted scaler  (call once at startup)
"""

import os
import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from utils.constants import (
    FEATURE_COLUMNS,
    FEATURE_RANGES,
    CATEGORICAL_ENCODINGS,
    DATA_PATH,
    SCALER_PATH,
    TARGET_COLUMN,
)
from utils.helpers import get_logger

warnings.filterwarnings("ignore", category=UserWarning)
logger = get_logger(__name__)

# ─────────────────────────────────────────────
# Column medians / modes used for imputation
# (populated when fit_scaler() or _load_scaler() runs)
# ─────────────────────────────────────────────
_FILL_VALUES: Dict[str, Any] = {}
_scaler: Optional[MinMaxScaler] = None


# ─────────────────────────────────────────────
# Encoding maps  (centralised in constants but
# mirrored here for fast lookup)
# ─────────────────────────────────────────────

# Maps human-readable UI strings → integer codes
_ENCODE: Dict[str, Dict[str, int]] = CATEGORICAL_ENCODINGS


# ─────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────

def _coerce_types(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Stage 1 — Type coercion.

    Convert every field to its expected Python numeric type.
    Non-numeric strings that survive the encoding step are set to NaN
    so the imputer can handle them in stage 2.
    """
    coerced: Dict[str, Any] = {}

    for col in FEATURE_COLUMNS:
        raw_val = raw.get(col)

        # ── Categorical fields: map label → int ──────────────────────
        if col in _ENCODE:
            mapping = _ENCODE[col]
            if isinstance(raw_val, str):
                # Try exact match, then case-insensitive
                encoded = mapping.get(raw_val) or mapping.get(raw_val.lower())
                coerced[col] = float(encoded) if encoded is not None else np.nan
            elif raw_val is None:
                coerced[col] = np.nan
            else:
                coerced[col] = float(raw_val)   # already numeric

        # ── Continuous / ordinal fields ──────────────────────────────
        else:
            try:
                coerced[col] = float(raw_val) if raw_val is not None else np.nan
            except (ValueError, TypeError):
                coerced[col] = np.nan

    return coerced


def _impute_missing(coerced: Dict[str, Any]) -> Dict[str, Any]:
    """
    Stage 2 — Missing value imputation.

    Uses pre-computed _FILL_VALUES (medians for continuous,
    mode for categorical).  Falls back to the feature midpoint
    if _FILL_VALUES hasn't been populated yet.
    """
    imputed = dict(coerced)

    for col in FEATURE_COLUMNS:
        val = imputed.get(col)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            fill = _FILL_VALUES.get(col)
            if fill is None:
                # Fallback: midpoint of valid range
                lo, hi = FEATURE_RANGES.get(col, (0, 1))
                fill = (lo + hi) / 2.0
                logger.debug("No fill value for %s; using midpoint %.2f", col, fill)
            imputed[col] = fill
            logger.debug("Imputed missing field '%s' with %.4f", col, fill)

    return imputed


def _clamp_ranges(imputed: Dict[str, Any]) -> Dict[str, Any]:
    """
    Stage 3 — Range clamping.

    Clip each feature to its clinically valid [min, max] so that
    extreme outliers don't distort the scaler.
    """
    clamped = dict(imputed)

    for col in FEATURE_COLUMNS:
        if col in FEATURE_RANGES:
            lo, hi = FEATURE_RANGES[col]
            val = clamped[col]
            clamped[col] = float(np.clip(val, lo, hi))

    return clamped


def _to_feature_array(patient_dict: Dict[str, Any]) -> np.ndarray:
    """
    Convert a fully processed dict → ordered 1-D NumPy array.

    Column order is determined by FEATURE_COLUMNS so the scaler
    and the model always see features in the same position.
    """
    return np.array([patient_dict[col] for col in FEATURE_COLUMNS], dtype=np.float64)


# ─────────────────────────────────────────────
# Scaler management
# ─────────────────────────────────────────────

def _compute_fill_values(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute per-column imputation values from a reference DataFrame.

    Categorical → mode  |  Continuous → median
    """
    fills: Dict[str, Any] = {}
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            continue
        if col in _ENCODE:
            mode_vals = df[col].mode()
            fills[col] = float(mode_vals.iloc[0]) if len(mode_vals) else 0.0
        else:
            fills[col] = float(df[col].median())
    return fills


def fit_scaler(df: Optional[pd.DataFrame] = None) -> MinMaxScaler:
    """
    Fit (or re-fit) the MinMaxScaler on *df*.

    If *df* is None, loads DATA_PATH automatically.

    Side effects
    ------------
    - Populates the module-level _scaler and _FILL_VALUES.
    - Persists the scaler to SCALER_PATH (pickle) for future runs.

    Returns
    -------
    Fitted MinMaxScaler instance.
    """
    global _scaler, _FILL_VALUES

    if df is None:
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"Training data not found at {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
        logger.info("Loaded training data from %s (%d rows)", DATA_PATH, len(df))

    # Keep only feature columns (drop target if present)
    feature_df = df[[c for c in FEATURE_COLUMNS if c in df.columns]].copy()

    # Compute fill values BEFORE fitting the scaler
    _FILL_VALUES = _compute_fill_values(feature_df)
    logger.debug("Fill values computed: %s", _FILL_VALUES)

    # Fit scaler
    _scaler = MinMaxScaler(feature_range=(0, 1))
    _scaler.fit(feature_df[FEATURE_COLUMNS])
    logger.info("MinMaxScaler fitted on %d samples, %d features", len(feature_df), len(FEATURE_COLUMNS))

    # Persist to disk
    scaler_dir = Path(SCALER_PATH).parent
    scaler_dir.mkdir(parents=True, exist_ok=True)
    with open(SCALER_PATH, "wb") as fh:
        pickle.dump({"scaler": _scaler, "fill_values": _FILL_VALUES}, fh)
    logger.info("Scaler persisted to %s", SCALER_PATH)

    return _scaler


def _load_scaler() -> MinMaxScaler:
    """
    Load a previously fitted scaler from SCALER_PATH.
    If the file doesn't exist, call fit_scaler() to create one.
    """
    global _scaler, _FILL_VALUES

    if _scaler is not None:
        return _scaler                          # already loaded in memory

    if os.path.exists(SCALER_PATH):
        with open(SCALER_PATH, "rb") as fh:
            payload = pickle.load(fh)
        _scaler = payload["scaler"]
        _FILL_VALUES = payload.get("fill_values", {})
        logger.info("Scaler loaded from %s", SCALER_PATH)
    else:
        logger.warning("Scaler not found at %s — fitting from scratch", SCALER_PATH)
        fit_scaler()

    return _scaler


def _scale(array_2d: np.ndarray) -> np.ndarray:
    """Apply MinMaxScaler transform to a (N, 13) array."""
    scaler = _load_scaler()
    return scaler.transform(array_2d)


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def get_feature_names() -> List[str]:
    """Return the ordered list of feature column names."""
    return list(FEATURE_COLUMNS)


def preprocess_patient(patient: Dict[str, Any]) -> np.ndarray:
    """
    Full preprocessing pipeline for a single patient dict.

    Steps
    -----
    1. Type coercion (strings → numbers, labels → codes)
    2. Missing-value imputation  
    3. Range clamping
    4. MinMax scaling to [0, 1]

    Parameters
    ----------
    patient : dict
        Raw patient record from the UI form or sample_input.json.
        Keys must include all FEATURE_COLUMNS (extras are ignored).

    Returns
    -------
    np.ndarray  shape (1, 13)  — ready for model.predict() / KNN
    """
    logger.debug("Preprocessing patient: %s", {k: v for k, v in patient.items() if k in FEATURE_COLUMNS})

    # Pipeline stages
    coerced  = _coerce_types(patient)
    imputed  = _impute_missing(coerced)
    clamped  = _clamp_ranges(imputed)

    # Convert to array
    raw_arr = _to_feature_array(clamped).reshape(1, -1)   # shape (1, 13)

    # Scale
    scaled_arr = _scale(raw_arr)

    logger.debug("Preprocessed vector: %s", scaled_arr)
    return scaled_arr


def preprocess_dataframe(df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Preprocess an entire DataFrame of patients (e.g., the training set).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain all FEATURE_COLUMNS.  May also contain TARGET_COLUMN.

    Returns
    -------
    (X: np.ndarray shape (N, 13),  y: np.ndarray shape (N,) or None)
        y is None if TARGET_COLUMN is not present in df.
    """
    df = df.copy()

    # Extract labels before processing features
    y: Optional[np.ndarray] = None
    if TARGET_COLUMN in df.columns:
        y = df[TARGET_COLUMN].values.astype(int)

    # Run each row through the pipeline
    records = df.to_dict(orient="records")
    processed_rows = []

    for rec in records:
        coerced = _coerce_types(rec)
        imputed  = _impute_missing(coerced)
        clamped  = _clamp_ranges(imputed)
        processed_rows.append(_to_feature_array(clamped))

    raw_X = np.vstack(processed_rows)          # shape (N, 13)
    scaled_X = _scale(raw_X)

    logger.info("preprocess_dataframe: processed %d rows", len(scaled_X))
    return scaled_X, y


def inverse_transform(scaled_array: np.ndarray) -> np.ndarray:
    """
    Reverse the MinMax scaling — useful for displaying readable values
    after internal computations.

    Parameters
    ----------
    scaled_array : np.ndarray  shape (N, 13) or (1, 13)

    Returns
    -------
    np.ndarray  same shape, values in original clinical units
    """
    scaler = _load_scaler()
    return scaler.inverse_transform(scaled_array)


def get_preprocessing_summary(patient: Dict[str, Any]) -> Dict[str, Any]:
    """
    Developer / debug helper.

    Returns a human-readable dict showing the value of each feature
    at every stage of the pipeline.

    Useful for the 'Data Entry' debug pane in the UI.
    """
    coerced = _coerce_types(patient)
    imputed  = _impute_missing(coerced)
    clamped  = _clamp_ranges(imputed)
    raw_arr  = _to_feature_array(clamped).reshape(1, -1)
    scaled   = _scale(raw_arr)[0]

    summary = {}
    for i, col in enumerate(FEATURE_COLUMNS):
        summary[col] = {
            "raw":     patient.get(col),
            "coerced": coerced.get(col),
            "imputed": imputed.get(col),
            "clamped": clamped.get(col),
            "scaled":  round(float(scaled[i]), 4),
        }
    return summary


# ─────────────────────────────────────────────
# Module init — load scaler eagerly on import
# ─────────────────────────────────────────────
try:
    _load_scaler()
except Exception as _exc:
    logger.warning("Could not load scaler on import: %s", _exc)