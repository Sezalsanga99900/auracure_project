"""
core/risk_model.py
──────────────────
Cardiac risk scoring engine for AuraEcho+.

Responsibility:
    Train and serve a Random Forest classifier that turns a preprocessed
    patient feature vector into:
        • A binary prediction    (disease / no disease)
        • A risk level label     (Low / Medium / High)
        • A confidence score     (0 – 100 %)
        • Feature importances    (which vitals drove the decision)
        • A plain-language explanation

Architecture:
    ┌──────────────────────────────────────────────┐
    │  Raw patient dict                            │
    │        ↓  preprocess_patient()               │
    │  Scaled (1×13) array                         │
    │        ↓  RandomForestClassifier.predict()   │
    │  Probability vector  [P_no_disease, P_disease]│
    │        ↓  _probability_to_risk_level()       │
    │  RiskResult dataclass                        │
    └──────────────────────────────────────────────┘

Public API:
    load_model()           → fitted RandomForest (cached)
    predict_risk(patient)  → RiskResult
    train_model(df)        → fitted model + metrics dict
    get_feature_importances() → List[(feature, importance)]
    explain_prediction(patient) → str  (plain-English narrative)
"""

import os
import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from core.preprocess import preprocess_patient, preprocess_dataframe, fit_scaler
from utils.constants import (
    DATA_PATH,
    MODEL_PATH,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    RISK_LOW,
    RISK_MEDIUM,
    RISK_HIGH,
    RISK_THRESHOLDS,
    RF_N_ESTIMATORS,
    RF_MAX_DEPTH,
    RF_RANDOM_STATE,
    CONFIDENCE_LOW_THRESHOLD,
    CONFIDENCE_HIGH_THRESHOLD,
)
from utils.helpers import get_logger, score_to_risk_level

warnings.filterwarnings("ignore")
logger = get_logger(__name__)


# ─────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────

@dataclass
class RiskResult:
    """
    Everything the UI and AI modules need about a single patient assessment.

    Attributes
    ----------
    risk_level       : "Low" | "Medium" | "High"
    confidence_pct   : float   0–100
    disease_prob     : float   0.0–1.0  (raw model probability)
    predicted_label  : int     0 = no disease, 1 = disease
    feature_contributions : list of (feature_name, importance_score)
    top_risk_factors : list of str  (human-readable top 3 drivers)
    explanation      : str  (one-paragraph plain-language summary)
    model_version    : str
    """
    risk_level:            str
    confidence_pct:        float
    disease_prob:          float
    predicted_label:       int
    feature_contributions: List[Tuple[str, float]] = field(default_factory=list)
    top_risk_factors:      List[str]                = field(default_factory=list)
    explanation:           str                      = ""
    model_version:         str                      = "rf_v1"

    # Derived convenience properties
    @property
    def is_high_risk(self) -> bool:
        return self.risk_level == RISK_HIGH

    @property
    def badge_color(self) -> str:
        return {"Low": "green", "Medium": "orange", "High": "red"}.get(self.risk_level, "gray")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_level":            self.risk_level,
            "confidence_pct":        round(self.confidence_pct, 1),
            "disease_prob":          round(self.disease_prob, 4),
            "predicted_label":       self.predicted_label,
            "top_risk_factors":      self.top_risk_factors,
            "explanation":           self.explanation,
            "model_version":         self.model_version,
            "feature_contributions": [
                {"feature": f, "importance": round(v, 4)}
                for f, v in self.feature_contributions
            ],
        }


# ─────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────

# Human-readable names for the 13 Cleveland features
_FEATURE_LABELS: Dict[str, str] = {
    "age":      "Age",
    "sex":      "Sex",
    "cp":       "Chest Pain Type",
    "trestbps": "Resting Blood Pressure",
    "chol":     "Serum Cholesterol",
    "fbs":      "Fasting Blood Sugar",
    "restecg":  "Resting ECG Result",
    "thalach":  "Max Heart Rate Achieved",
    "exang":    "Exercise-Induced Angina",
    "oldpeak":  "ST Depression (Exercise)",
    "slope":    "Slope of Peak ST Segment",
    "ca":       "Number of Major Vessels (Fluoroscopy)",
    "thal":     "Thalassemia Type",
}

_RISK_EXPLANATIONS: Dict[str, str] = {
    RISK_LOW: (
        "The model detected minimal cardiac risk indicators. "
        "Routine follow-up and preventive lifestyle measures are recommended."
    ),
    RISK_MEDIUM: (
        "Moderate cardiac risk factors are present. "
        "Closer monitoring, further diagnostic tests, and lifestyle modifications "
        "are advisable. Consult a cardiologist if symptoms persist."
    ),
    RISK_HIGH: (
        "Significant cardiac risk factors detected. "
        "Immediate cardiology consultation is strongly recommended. "
        "Do not delay further evaluation and possible intervention."
    ),
}

# Module-level cached model
_model: Optional[RandomForestClassifier] = None


def _probability_to_risk_level(disease_prob: float) -> Tuple[str, float]:
    """
    Map raw disease probability → (risk_level, confidence_pct).

    Thresholds from constants.RISK_THRESHOLDS:
        < LOW_THRESHOLD   → Low     (confidence = 1 - disease_prob)
        < HIGH_THRESHOLD  → Medium  (confidence based on distance from 0.5)
        ≥ HIGH_THRESHOLD  → High    (confidence = disease_prob)
    """
    lo = RISK_THRESHOLDS["low_max"]      # e.g. 0.35
    hi = RISK_THRESHOLDS["high_min"]     # e.g. 0.65

    if disease_prob < lo:
        risk = RISK_LOW
        confidence = (1.0 - disease_prob) * 100.0
    elif disease_prob < hi:
        risk = RISK_MEDIUM
        # Confidence = how far from the uncertain midpoint (0.5)
        confidence = (abs(disease_prob - 0.5) / 0.5) * 60.0 + 40.0   # 40–100%
    else:
        risk = RISK_HIGH
        confidence = disease_prob * 100.0

    return risk, round(min(confidence, 99.9), 1)


def _get_feature_contributions(
    model: RandomForestClassifier,
) -> List[Tuple[str, float]]:
    """
    Return feature importances sorted descending.

    Returns
    -------
    List of (readable_feature_label, importance_score)
    """
    importances = model.feature_importances_
    pairs = [
        (_FEATURE_LABELS.get(col, col), float(imp))
        for col, imp in zip(FEATURE_COLUMNS, importances)
    ]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs


def _top_risk_factors(contributions: List[Tuple[str, float]], n: int = 3) -> List[str]:
    """Extract the top-n feature names from a contributions list."""
    return [feat for feat, _ in contributions[:n]]


def _build_explanation(
    risk_level: str,
    top_factors: List[str],
    confidence: float,
    disease_prob: float,
) -> str:
    """
    Compose a plain-English explanation paragraph.
    """
    base = _RISK_EXPLANATIONS.get(risk_level, "")
    factors_str = ", ".join(top_factors) if top_factors else "multiple clinical features"
    conf_desc = (
        "high confidence" if confidence >= CONFIDENCE_HIGH_THRESHOLD
        else "moderate confidence" if confidence >= CONFIDENCE_LOW_THRESHOLD
        else "low confidence"
    )
    return (
        f"Assessment ({conf_desc}, {confidence:.1f}%): {base} "
        f"Key contributing factors: {factors_str}. "
        f"Disease probability estimated at {disease_prob*100:.1f}%."
    )


# ─────────────────────────────────────────────
# Model persistence
# ─────────────────────────────────────────────

def _save_model(model: RandomForestClassifier) -> None:
    model_dir = Path(MODEL_PATH).parent
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as fh:
        pickle.dump(model, fh)
    logger.info("Model saved to %s", MODEL_PATH)


def _load_model_from_disk() -> Optional[RandomForestClassifier]:
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as fh:
        model = pickle.load(fh)
    logger.info("Model loaded from %s", MODEL_PATH)
    return model


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train_model(df: Optional[pd.DataFrame] = None) -> Tuple[RandomForestClassifier, Dict]:
    """
    Train a Random Forest on *df* (or DATA_PATH if None).

    Steps
    -----
    1. Fit the MinMaxScaler on the full dataset.
    2. Preprocess all rows (preprocess_dataframe).
    3. Train/test split (80/20 stratified).
    4. Fit RandomForestClassifier.
    5. Evaluate and return metrics.
    6. Persist model to MODEL_PATH.

    Returns
    -------
    (fitted_model, metrics_dict)
        metrics_dict keys: accuracy, precision, recall, f1, roc_auc
    """
    global _model

    if df is None:
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"Training data not found at {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
        logger.info("Loaded %d rows from %s", len(df), DATA_PATH)

    # Fit scaler first (needed by preprocess_dataframe)
    fit_scaler(df)

    # Preprocess
    X, y = preprocess_dataframe(df)
    if y is None:
        raise ValueError(f"Training data must contain target column '{TARGET_COLUMN}'")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RF_RANDOM_STATE, stratify=y
    )
    logger.info(
        "Train/test split: %d train, %d test (stratified)",
        len(X_train), len(X_test),
    )

    # Train
    model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        random_state=RF_RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    logger.info(
        "RandomForest trained: %d trees, max_depth=%s",
        RF_N_ESTIMATORS, RF_MAX_DEPTH,
    )

    # Evaluate
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1":        round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc":   round(float(roc_auc_score(y_test, y_proba)), 4),
        "train_size": int(len(X_train)),
        "test_size":  int(len(X_test)),
    }
    logger.info("Metrics: %s", metrics)

    # Cache + persist
    _model = model
    _save_model(model)

    return model, metrics


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def load_model() -> RandomForestClassifier:
    """
    Return the fitted model, loading or training it if necessary.

    Priority:
    1. Module-level cache (_model)
    2. Disk (MODEL_PATH)
    3. Train from scratch using DATA_PATH
    """
    global _model

    if _model is not None:
        return _model

    _model = _load_model_from_disk()
    if _model is not None:
        return _model

    logger.warning("No model on disk — training from scratch")
    _model, metrics = train_model()
    logger.info("Auto-trained model metrics: %s", metrics)
    return _model


def predict_risk(patient: Dict[str, Any]) -> RiskResult:
    """
    Full risk assessment pipeline for one patient.

    Parameters
    ----------
    patient : dict
        Raw patient record (same format as UI form / sample_input.json).

    Returns
    -------
    RiskResult  — see dataclass definition above.
    """
    model = load_model()

    # Preprocess
    X = preprocess_patient(patient)     # shape (1, 13)

    # Predict
    proba         = model.predict_proba(X)[0]   # [P_class0, P_class1]
    disease_prob  = float(proba[1])
    predicted_lbl = int(model.predict(X)[0])

    # Map → risk level
    risk_level, confidence = _probability_to_risk_level(disease_prob)

    # Feature importance breakdown
    contributions = _get_feature_contributions(model)
    top_factors   = _top_risk_factors(contributions)

    # Plain-English explanation
    explanation = _build_explanation(risk_level, top_factors, confidence, disease_prob)

    result = RiskResult(
        risk_level=risk_level,
        confidence_pct=confidence,
        disease_prob=disease_prob,
        predicted_label=predicted_lbl,
        feature_contributions=contributions,
        top_risk_factors=top_factors,
        explanation=explanation,
    )

    logger.info(
        "Risk assessment — level=%s | prob=%.3f | confidence=%.1f%%",
        risk_level, disease_prob, confidence,
    )
    return result


def get_feature_importances() -> List[Tuple[str, float]]:
    """
    Return global feature importances from the loaded model.

    Returns
    -------
    List of (readable_label, importance) sorted descending.
    """
    model = load_model()
    return _get_feature_contributions(model)


def explain_prediction(patient: Dict[str, Any]) -> str:
    """
    Convenience wrapper — returns just the explanation string.
    Useful for the AI prompt builder.
    """
    result = predict_risk(patient)
    return result.explanation


def batch_predict(patients: List[Dict[str, Any]]) -> List[RiskResult]:
    """
    Score multiple patients efficiently.

    Uses the model once (avoids repeated disk loads) and processes
    each patient through the full pipeline.

    Parameters
    ----------
    patients : list of patient dicts

    Returns
    -------
    List[RiskResult]  same order as input
    """
    model = load_model()        # ensure loaded once
    return [predict_risk(p) for p in patients]


# ─────────────────────────────────────────────
# Module init — load/train model eagerly
# ─────────────────────────────────────────────
try:
    load_model()
    logger.info("Risk model ready")
except Exception as _exc:
    logger.warning("Risk model could not be loaded at import time: %s", _exc)