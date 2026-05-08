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
    3. Range clamping  — clip values to clinically valid bounds
    4. Scaling         — MinMax scale to [0, 1] per feature
    5. Column ordering — guarantee consistent feature vector order

Public API:
    preprocess_patient(patient_dict, validate)  → np.ndarray  shape (1, 13)
    preprocess_dataframe(df)                    → np.ndarray  shape (N, 13)
    get_feature_names()                         → List[str]
    inverse_transform(array)                    → np.ndarray
    fit_scaler(df)                              → fitted scaler
    get_preprocessing_summary(patient)          → Dict
    validate_before_preprocess(patient)         → Tuple[bool, List[str]]
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
    CATEGORICAL_FEATURES,
    HEART_DATA_PATH,
    SCALER_SAVE_PATH,
    TARGET_COLUMN,
)
from utils.helpers import get_logger

warnings.filterwarnings("ignore", category=UserWarning)
logger = get_logger(__name__)

# ─────────────────────────────────────────────
# Module-level state
# ─────────────────────────────────────────────
_FILL_VALUES: Dict[str, Any] = {}
_scaler: Optional[MinMaxScaler] = None


# ─────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────

def _coerce_types(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Stage 1 — Type coercion.

    FIXED: Encoding lookup no longer uses falsy-zero bug.
           mapping.get(raw_val) or mapping.get(raw_val.lower())
           → explicit None check so code=0 is not skipped.
    """
    coerced: Dict[str, Any] = {}

    for col in FEATURE_COLUMNS:
        raw_val = raw.get(col)

        if col in CATEGORICAL_ENCODINGS:
            mapping = CATEGORICAL_ENCODINGS[col]
            if isinstance(raw_val, str):
                # FIXED: explicit None check (avoids falsy 0 bug)
                encoded = mapping.get(raw_val)
                if encoded is None:
                    encoded = mapping.get(raw_val.lower())
                coerced[col] = float(encoded) if encoded is not None else np.nan
            elif raw_val is None:
                coerced[col] = np.nan
            else:
                coerced[col] = float(raw_val)
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
    mode for categorical). Falls back to the feature midpoint
    if _FILL_VALUES has not been populated yet.
    """
    imputed = dict(coerced)

    for col in FEATURE_COLUMNS:
        val = imputed.get(col)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            fill = _FILL_VALUES.get(col)
            if fill is None:
                lo, hi = FEATURE_RANGES.get(col, (0, 1))
                fill = (lo + hi) / 2.0
                logger.debug(
                    "No fill value for '%s'; using midpoint %.2f", col, fill
                )
            imputed[col] = fill
            logger.debug("Imputed missing field '%s' with %.4f", col, fill)

    return imputed


def _clamp_ranges(imputed: Dict[str, Any]) -> Dict[str, Any]:
    """
    Stage 3 — Range clamping.

    FIXED: Only applies to NUMERICAL features that appear in FEATURE_RANGES.
           Categorical features are skipped (already encoded to valid ints).
    """
    clamped = dict(imputed)

    for col in FEATURE_COLUMNS:
        # Numerical only — categorical already encoded to valid int
        if col in FEATURE_RANGES:
            lo, hi = FEATURE_RANGES[col]
            clamped[col] = float(np.clip(clamped[col], lo, hi))

    return clamped


def _to_feature_array(patient_dict: Dict[str, Any]) -> np.ndarray:
    """
    Convert a fully processed dict → ordered 1-D NumPy array.
    Column order is determined by FEATURE_COLUMNS.
    """
    return np.array(
        [patient_dict[col] for col in FEATURE_COLUMNS],
        dtype=np.float64,
    )


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
        if col in CATEGORICAL_ENCODINGS:
            mode_vals = df[col].mode()
            fills[col] = float(mode_vals.iloc[0]) if len(mode_vals) else 0.0
        else:
            fills[col] = float(df[col].median())
    return fills


def fit_scaler(df: Optional[pd.DataFrame] = None) -> MinMaxScaler:
    """
    Fit (or re-fit) the MinMaxScaler on *df*.

    FIXED: Runs the full encoding pipeline on all rows before fitting
           the scaler — previously fitted on raw (possibly string) data.

    If *df* is None, loads HEART_DATA_PATH automatically.

    Side effects
    ------------
    - Populates module-level _scaler and _FILL_VALUES.
    - Persists scaler + fill values to SCALER_SAVE_PATH (pickle).

    Returns
    -------
    Fitted MinMaxScaler instance.
    """
    global _scaler, _FILL_VALUES

    if df is None:
        if not os.path.exists(HEART_DATA_PATH):
            raise FileNotFoundError(
                f"Training data not found at {HEART_DATA_PATH}"
            )
        df = pd.read_csv(HEART_DATA_PATH)
        logger.info(
            "Loaded training data from %s (%d rows)",
            HEART_DATA_PATH, len(df),
        )

    feature_df = df[
        [c for c in FEATURE_COLUMNS if c in df.columns]
    ].copy()

    # Compute fill values from RAW df (before encoding)
    _FILL_VALUES = _compute_fill_values(feature_df)
    logger.debug("Fill values computed: %s", _FILL_VALUES)

    # FIXED: Run full pipeline on every row before fitting scaler
    records = feature_df.to_dict(orient="records")
    processed = []
    for rec in records:
        coerced = _coerce_types(rec)
        imputed  = _impute_missing(coerced)
        clamped  = _clamp_ranges(imputed)
        processed.append(_to_feature_array(clamped))

    clean_X = np.vstack(processed)   # shape (N, 13) — fully numeric

    # Fit scaler on clean processed array
    _scaler = MinMaxScaler(feature_range=(0, 1))
    _scaler.fit(clean_X)
    logger.info(
        "MinMaxScaler fitted on %d samples, %d features",
        len(clean_X), len(FEATURE_COLUMNS),
    )

    # Persist to disk
    scaler_dir = Path(SCALER_SAVE_PATH).parent
    scaler_dir.mkdir(parents=True, exist_ok=True)
    with open(SCALER_SAVE_PATH, "wb") as fh:
        # NOTE: Only load pickle files from trusted sources.
        pickle.dump({"scaler": _scaler, "fill_values": _FILL_VALUES}, fh)
    logger.info("Scaler persisted to %s", SCALER_SAVE_PATH)

    return _scaler


def _load_scaler() -> MinMaxScaler:
    """
    Load a previously fitted scaler from SCALER_SAVE_PATH.
    If the file does not exist, call fit_scaler() to create one.
    """
    global _scaler, _FILL_VALUES

    if _scaler is not None:
        return _scaler

    if os.path.exists(SCALER_SAVE_PATH):
        with open(SCALER_SAVE_PATH, "rb") as fh:
            # NOTE: Only load from trusted sources.
            payload = pickle.load(fh)
        _scaler      = payload["scaler"]
        _FILL_VALUES = payload.get("fill_values", {})
        logger.info("Scaler loaded from %s", SCALER_SAVE_PATH)
    else:
        logger.warning(
            "Scaler not found at %s — fitting from scratch", SCALER_SAVE_PATH
        )
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


def validate_before_preprocess(
    patient: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    """
    ADDED: Quick sanity check before running the pipeline.

    Returns (ok, list_of_warnings).
    Warnings do not block processing — pipeline handles missing values.
    """
    warnings_list: List[str] = []

    for col in FEATURE_COLUMNS:
        if col not in patient:
            warnings_list.append(f"Missing field: '{col}'")
        elif patient[col] is None:
            warnings_list.append(f"Null value for: '{col}'")

    return len(warnings_list) == 0, warnings_list


def preprocess_patient(
    patient: Dict[str, Any],
    validate: bool = True,           # ADDED: optional pre-validation
) -> np.ndarray:
    """
    Full preprocessing pipeline for a single patient dict.

    ADDED: validate parameter — if True, runs validate_before_preprocess()
           and logs warnings before processing.

    Steps
    -----
    1. Type coercion
    2. Missing-value imputation
    3. Range clamping
    4. MinMax scaling to [0, 1]

    Parameters
    ----------
    patient  : dict — raw patient record from UI form or sample_input.json
    validate : bool — whether to run pre-validation (default True)

    Returns
    -------
    np.ndarray  shape (1, 13)
    """
    if validate:
        ok, warnings_list = validate_before_preprocess(patient)
        if not ok:
            logger.warning(
                "preprocess_patient: %d field warning(s): %s",
                len(warnings_list), warnings_list,
            )

    logger.debug(
        "Preprocessing patient: %s",
        {k: v for k, v in patient.items() if k in FEATURE_COLUMNS},
    )

    coerced  = _coerce_types(patient)
    imputed  = _impute_missing(coerced)
    clamped  = _clamp_ranges(imputed)
    raw_arr  = _to_feature_array(clamped).reshape(1, -1)   # shape (1, 13)
    scaled   = _scale(raw_arr)

    logger.debug("Preprocessed vector: %s", scaled)
    return scaled


def preprocess_dataframe(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Preprocess an entire DataFrame of patients (e.g., the training set).

    Parameters
    ----------
    df : pd.DataFrame — must contain all FEATURE_COLUMNS.
                        May also contain TARGET_COLUMN.

    Returns
    -------
    (X: np.ndarray shape (N, 13),  y: np.ndarray shape (N,) or None)
        y is None if TARGET_COLUMN is not present in df.
    """
    df = df.copy()

    y: Optional[np.ndarray] = None
    if TARGET_COLUMN in df.columns:
        y = df[TARGET_COLUMN].values.astype(int)

    records = df.to_dict(orient="records")
    processed_rows = []

    logger.info(
        "preprocess_dataframe: processing %d rows (row-by-row pipeline)",
        len(records),
    )

    for rec in records:
        coerced = _coerce_types(rec)
        imputed  = _impute_missing(coerced)
        clamped  = _clamp_ranges(imputed)
        processed_rows.append(_to_feature_array(clamped))

    raw_X    = np.vstack(processed_rows)   # shape (N, 13)
    scaled_X = _scale(raw_X)

    logger.info("preprocess_dataframe: processed %d rows", len(scaled_X))
    return scaled_X, y


def inverse_transform(scaled_array: np.ndarray) -> np.ndarray:
    """
    Reverse the MinMax scaling.

    Parameters
    ----------
    scaled_array : np.ndarray  shape (N, 13) or (1, 13)

    Returns
    -------
    np.ndarray  same shape, values in original clinical units.
    """
    scaler = _load_scaler()
    return scaler.inverse_transform(scaled_array)


def get_preprocessing_summary(patient: Dict[str, Any]) -> Dict[str, Any]:
    """
    Developer / debug helper.

    Returns a human-readable dict showing the value of each feature
    at every stage of the pipeline.
    Useful for the Data Entry debug pane in the UI.

    ADDED to Public API docstring (was missing from module header).
    """
    coerced = _coerce_types(patient)
    imputed  = _impute_missing(coerced)
    clamped  = _clamp_ranges(imputed)
    raw_arr  = _to_feature_array(clamped).reshape(1, -1)
    scaled   = _scale(raw_arr)[0]

    summary: Dict[str, Any] = {}
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