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

Public API:
    load_model()                → fitted RandomForest (cached)
    predict_risk(patient)       → RiskResult
    train_model(df)             → fitted model + metrics dict
    get_feature_importances()   → List[(feature, importance)]
    explain_prediction(patient) → str
    batch_predict(patients)     → List[RiskResult]
    get_model_metadata()        → Dict
    retrain_if_stale(days)      → bool
"""

import os
import time
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
from sklearn.model_selection import train_test_split, cross_val_score

from core.preprocess import preprocess_patient, preprocess_dataframe, fit_scaler
from utils.constants import (
    HEART_DATA_PATH,
    MODEL_SAVE_PATH,
    FEATURE_COLUMNS,
    FEATURE_LABELS,
    TARGET_COLUMN,
    RISK_LABELS,
    RISK_LEVELS,
    RISK_COLORS,
    RISK_ICONS,
    RF_N_ESTIMATORS,
    RF_MAX_DEPTH,
    RF_RANDOM_STATE,
    CONFIDENCE_LOW_THRESHOLD,
    CONFIDENCE_HIGH_THRESHOLD,
)
from utils.helpers import get_logger, normalize_risk_level

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
    risk_level           : "LOW" | "MEDIUM" | "HIGH"  (key, not label)
    risk_label           : "Low Risk" | "Medium Risk" | "High Risk"
    confidence_pct       : float  0–100
    disease_prob         : float  0.0–1.0
    predicted_label      : int    0 = no disease, 1 = disease
    feature_contributions: list of (feature_name, importance_score)
    top_risk_factors     : list of str
    explanation          : str
    model_version        : str
    """
    risk_level:            str
    risk_label:            str
    confidence_pct:        float
    disease_prob:          float
    predicted_label:       int
    feature_contributions: List[Tuple[str, float]] = field(default_factory=list)
    top_risk_factors:      List[str]                = field(default_factory=list)
    explanation:           str                      = ""
    model_version:         str                      = "rf_v1"

    @property
    def is_high_risk(self) -> bool:
        return self.risk_level == "HIGH"

    @property
    def badge_color(self) -> str:
        """
        FIXED: Uses RISK_COLORS hex values from constants
               instead of hardcoded 'green'/'orange'/'red' strings.
        """
        return RISK_COLORS.get(self.risk_level, "#808080")

    @property
    def badge_icon(self) -> str:
        """ADDED: Convenience icon accessor for UI and prompt_builder."""
        return RISK_ICONS.get(self.risk_level, "❔")

    def to_dict(self) -> Dict[str, Any]:
        """
        FIXED: Now stores both risk_level KEY and risk_label STRING.
               Also includes badge_icon and badge_color for prompt_builder.
        """
        return {
            "risk_level":   self.risk_level,          # "HIGH"
            "risk_label":   self.risk_label,           # "High Risk"
            "confidence_pct":        round(self.confidence_pct, 1),
            "disease_prob":          round(self.disease_prob, 4),
            "predicted_label":       self.predicted_label,
            "top_risk_factors":      self.top_risk_factors,
            "explanation":           self.explanation,
            "model_version":         self.model_version,
            "badge_color":           self.badge_color,  # ADDED
            "badge_icon":            self.badge_icon,   # ADDED
            "feature_contributions": [
                {"feature": f, "importance": round(v, 4)}
                for f, v in self.feature_contributions
            ],
        }


# ─────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────

_RISK_EXPLANATIONS: Dict[str, str] = {
    "LOW": (
        "The model detected minimal cardiac risk indicators. "
        "Routine follow-up and preventive lifestyle measures are recommended."
    ),
    "MEDIUM": (
        "Moderate cardiac risk factors are present. "
        "Closer monitoring, further diagnostic tests, and lifestyle modifications "
        "are advisable. Consult a cardiologist if symptoms persist."
    ),
    "HIGH": (
        "Significant cardiac risk factors detected. "
        "Immediate cardiology consultation is strongly recommended. "
        "Do not delay further evaluation and possible intervention."
    ),
}

# Module-level cached model
_model: Optional[RandomForestClassifier] = None


def _probability_to_risk_level(disease_prob: float) -> Tuple[str, float]:
    """
    Map raw disease probability → (risk_level_key, confidence_pct).

    FIXED: Consistent confidence formula — no cliff at boundaries.
           Returns risk_level KEY ("LOW"|"MEDIUM"|"HIGH"), not label.
    """
    lo = RISK_LEVELS["LOW"][1]      # 0.35
    hi = RISK_LEVELS["HIGH"][0]     # 0.65

    if disease_prob < lo:
        risk = "LOW"
        # Scale: 0.0 → 100%, 0.35 → 65%
        confidence = (1.0 - disease_prob) * 100.0

    elif disease_prob < hi:
        risk = "MEDIUM"
        # FIXED: Normalize within band [0.35, 0.65] — no boundary cliff
        band_width = hi - lo           # 0.30
        mid        = (lo + hi) / 2    # 0.50
        distance   = abs(disease_prob - mid)
        confidence = 50.0 + (distance / (band_width / 2)) * 30.0

    else:
        risk = "HIGH"
        # Scale: 0.65 → 65%, 1.0 → 100%
        confidence = disease_prob * 100.0

    return risk, round(min(confidence, 99.9), 1)


def _get_feature_contributions(
    model: RandomForestClassifier,
    scaled_input: Optional[np.ndarray] = None,
) -> List[Tuple[str, float]]:
    """
    Return feature importances sorted descending.

    FIXED: If scaled_input provided, weights global importances by
           patient's actual feature values for patient-specific contributions.
           Uses FEATURE_LABELS from constants (no duplication).

    Returns
    -------
    List of (readable_feature_label, importance_score)
    """
    global_importances = model.feature_importances_

    if scaled_input is not None:
        patient_vals = scaled_input[0]            # shape (13,)
        weighted     = global_importances * patient_vals
        total        = weighted.sum()
        importances  = weighted / total if total > 0 else global_importances
    else:
        importances = global_importances

    pairs = [
        (FEATURE_LABELS.get(col, col), float(imp))
        for col, imp in zip(FEATURE_COLUMNS, importances)
    ]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs


def _top_risk_factors(
    contributions: List[Tuple[str, float]],
    n: int = 3,
) -> List[str]:
    """Extract the top-n feature names from a contributions list."""
    return [feat for feat, _ in contributions[:n]]


def _build_explanation(
    risk_level:   str,
    top_factors:  List[str],
    confidence:   float,
    disease_prob: float,
) -> str:
    """Compose a plain-English explanation paragraph."""
    base        = _RISK_EXPLANATIONS.get(risk_level, "")
    factors_str = ", ".join(top_factors) if top_factors else "multiple clinical features"
    conf_desc   = (
        "high confidence"     if confidence >= CONFIDENCE_HIGH_THRESHOLD
        else "moderate confidence" if confidence >= CONFIDENCE_LOW_THRESHOLD
        else "low confidence"
    )
    return (
        f"Assessment ({conf_desc}, {confidence:.1f}%): {base} "
        f"Key contributing factors: {factors_str}. "
        f"Disease probability estimated at {disease_prob * 100:.1f}%."
    )


# ─────────────────────────────────────────────
# Model persistence
# ─────────────────────────────────────────────

def _save_model(model: RandomForestClassifier) -> None:
    model_dir = Path(MODEL_SAVE_PATH).parent
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(MODEL_SAVE_PATH, "wb") as fh:
        pickle.dump(model, fh)
    logger.info("Model saved to %s", MODEL_SAVE_PATH)


def _load_model_from_disk() -> Optional[RandomForestClassifier]:
    if not os.path.exists(MODEL_SAVE_PATH):
        return None
    with open(MODEL_SAVE_PATH, "rb") as fh:
        # NOTE: Only load pickle files from trusted sources.
        model = pickle.load(fh)
    logger.info("Model loaded from %s", MODEL_SAVE_PATH)
    return model


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train_model(
    df: Optional[pd.DataFrame] = None,
) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    """
    Train a Random Forest on *df* (or HEART_DATA_PATH if None).

    ADDED: 5-fold cross-validation metrics (cv_roc_auc_mean/std).

    Returns
    -------
    (fitted_model, metrics_dict)
    """
    global _model

    if df is None:
        if not os.path.exists(HEART_DATA_PATH):
            raise FileNotFoundError(
                f"Training data not found at {HEART_DATA_PATH}"
            )
        df = pd.read_csv(HEART_DATA_PATH)
        logger.info("Loaded %d rows from %s", len(df), HEART_DATA_PATH)

    # Fit scaler first
    fit_scaler(df)

    # Preprocess
    X, y = preprocess_dataframe(df)
    if y is None:
        raise ValueError(
            f"Training data must contain target column '{TARGET_COLUMN}'"
        )

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=RF_RANDOM_STATE,
        stratify=y,
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

    # Evaluate — hold-out metrics
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics: Dict[str, Any] = {
        "accuracy":   round(float(accuracy_score(y_test, y_pred)), 4),
        "precision":  round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall":     round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1":         round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc":    round(float(roc_auc_score(y_test, y_proba)), 4),
        "train_size": int(len(X_train)),
        "test_size":  int(len(X_test)),
    }

    # ADDED: 5-fold cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
    metrics["cv_roc_auc_mean"] = round(float(cv_scores.mean()), 4)
    metrics["cv_roc_auc_std"]  = round(float(cv_scores.std()), 4)
    logger.info(
        "5-fold CV ROC-AUC: %.4f ± %.4f",
        cv_scores.mean(), cv_scores.std(),
    )

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
    2. Disk  (MODEL_SAVE_PATH)
    3. Train from scratch using HEART_DATA_PATH
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

    FIXED:
    - risk_level now stores KEY ("HIGH") not label ("High Risk")
    - risk_label stored separately for display
    - feature_contributions are patient-specific (weighted)
    - badge_color and badge_icon included in to_dict()

    Parameters
    ----------
    patient : dict — raw patient record (UI form / sample_input.json)

    Returns
    -------
    RiskResult
    """
    model = load_model()

    X = preprocess_patient(patient)          # shape (1, 13)

    proba         = model.predict_proba(X)[0]
    disease_prob  = float(proba[1])
    predicted_lbl = int(model.predict(X)[0])

    # FIXED: returns KEY not label
    risk_level, confidence = _probability_to_risk_level(disease_prob)
    risk_label = RISK_LABELS[risk_level]     # "High Risk" etc.

    # FIXED: patient-specific weighted contributions
    contributions = _get_feature_contributions(model, scaled_input=X)
    top_factors   = _top_risk_factors(contributions)
    explanation   = _build_explanation(
        risk_level, top_factors, confidence, disease_prob
    )

    result = RiskResult(
        risk_level=risk_level,
        risk_label=risk_label,
        confidence_pct=confidence,
        disease_prob=disease_prob,
        predicted_label=predicted_lbl,
        feature_contributions=contributions,
        top_risk_factors=top_factors,
        explanation=explanation,
    )

    logger.info(
        "Risk assessment — level=%s | label=%s | prob=%.3f | confidence=%.1f%%",
        risk_level, risk_label, disease_prob, confidence,
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


def batch_predict(
    patients: List[Dict[str, Any]],
) -> List["RiskResult"]:
    """
    Score multiple patients efficiently.

    FIXED: Model loaded once — no repeated load_model() calls
           per patient (was calling it indirectly via predict_risk).

    Parameters
    ----------
    patients : list of patient dicts

    Returns
    -------
    List[RiskResult]  same order as input
    """
    if not patients:
        return []

    model = load_model()    # load once
    results: List[RiskResult] = []

    for patient in patients:
        X = preprocess_patient(patient)
        proba         = model.predict_proba(X)[0]
        disease_prob  = float(proba[1])
        predicted_lbl = int(model.predict(X)[0])

        risk_level, confidence = _probability_to_risk_level(disease_prob)
        risk_label    = RISK_LABELS[risk_level]
        contributions = _get_feature_contributions(model, scaled_input=X)
        top_factors   = _top_risk_factors(contributions)
        explanation   = _build_explanation(
            risk_level, top_factors, confidence, disease_prob
        )

        results.append(RiskResult(
            risk_level=risk_level,
            risk_label=risk_label,
            confidence_pct=confidence,
            disease_prob=disease_prob,
            predicted_label=predicted_lbl,
            feature_contributions=contributions,
            top_risk_factors=top_factors,
            explanation=explanation,
        ))

    logger.info("batch_predict: scored %d patients", len(results))
    return results


def get_model_metadata() -> Dict[str, Any]:
    """
    ADDED: Return metadata about the currently loaded model.
    Used by ui/system_status.py.
    """
    model = load_model()
    importances = _get_feature_contributions(model)

    return {
        "model_type":    type(model).__name__,
        "n_estimators":  model.n_estimators,
        "max_depth":     model.max_depth,
        "n_features":    model.n_features_in_,
        "model_version": "rf_v1",
        "top_features":  importances[:3],
        "model_path":    MODEL_SAVE_PATH,
        "model_exists":  os.path.exists(MODEL_SAVE_PATH),
    }


def retrain_if_stale(max_age_days: int = 30) -> bool:
    """
    ADDED: Retrain model if saved file is older than max_age_days.
    Returns True if retrained, False if still fresh.
    """
    if not os.path.exists(MODEL_SAVE_PATH):
        train_model()
        return True

    age_seconds = time.time() - os.path.getmtime(MODEL_SAVE_PATH)
    age_days    = age_seconds / 86400

    if age_days > max_age_days:
        logger.info("Model is %.1f days old — retraining", age_days)
        train_model()
        return True

    logger.info("Model is %.1f days old — still fresh", age_days)
    return False


# ─────────────────────────────────────────────
# Module init — load/train model eagerly
# ─────────────────────────────────────────────
try:
    load_model()
    logger.info("Risk model ready")
except Exception as _exc:
    logger.warning("Risk model could not be loaded at import time: %s", _exc)