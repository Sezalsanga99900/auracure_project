# =============================================================================
# tests/test_risk_model.py
# AuraEcho+ — Risk Model Engine Tests
#
# Coverage:
#     • RiskResult dataclass (fields, properties, to_dict)
#     • Probability to risk level mapping (boundaries, confidence)
#     • Feature contributions (global vs patient-specific weighting)
#     • Model training (metrics, CV, persistence)
#     • Prediction (single, batch, explain)
#     • Metadata and retrain logic
#     • Edge cases and error handling
#
# Run:
#     pytest tests/test_risk_model.py -v
# =============================================================================

import pytest
import os
import time
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from sklearn.ensemble import RandomForestClassifier

from core.risk_model import (
    RiskResult,
    _probability_to_risk_level,
    _get_feature_contributions,
    _top_risk_factors,
    _build_explanation,
    train_model,
    load_model,
    predict_risk,
    get_feature_importances,
    explain_prediction,
    batch_predict,
    get_model_metadata,
    retrain_if_stale,
    _model,
)
from utils.constants import (
    RISK_LEVELS,
    RISK_LABELS,
    RISK_COLORS,
    RISK_ICONS,
    FEATURE_COLUMNS,
    HEART_DATA_PATH,
    MODEL_SAVE_PATH,
    RF_N_ESTIMATORS,
    RF_MAX_DEPTH,
    CONFIDENCE_LOW_THRESHOLD,
    CONFIDENCE_HIGH_THRESHOLD,
)
from core.preprocess import preprocess_patient


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def trained_model():
    """
    Train model once per test session.
    Reused by all tests to avoid repeated training overhead.
    """
    if not os.path.exists(HEART_DATA_PATH):
        pytest.skip("heart_data.csv not found")
    
    model, metrics = train_model()
    return model


@pytest.fixture
def sample_patient():
    """Returns a valid patient dictionary."""
    return {
        "name": "Test Patient",
        "age": 55,
        "sex": 1,
        "cp": 2,
        "trestbps": 130,
        "chol": 240,
        "fbs": 0,
        "restecg": 1,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 1.5,
        "slope": 1,
        "ca": 1,
        "thal": 2,
    }


@pytest.fixture
def risk_result(sample_patient, trained_model):
    """Returns a RiskResult for the sample patient."""
    return predict_risk(sample_patient)


# ─────────────────────────────────────────────────────────────────────────────
# RiskResult Dataclass Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRiskResult:
    def test_risk_result_fields(self, risk_result):
        """Test all fields are present and correctly typed."""
        assert isinstance(risk_result.risk_level, str)
        assert isinstance(risk_result.risk_label, str)
        assert isinstance(risk_result.confidence_pct, float)
        assert isinstance(risk_result.disease_prob, float)
        assert isinstance(risk_result.predicted_label, int)
        assert isinstance(risk_result.feature_contributions, list)
        assert isinstance(risk_result.top_risk_factors, list)
        assert isinstance(risk_result.explanation, str)
        assert isinstance(risk_result.model_version, str)

    def test_risk_result_level_is_key(self, risk_result):
        """
        FIXED: risk_level must be KEY ("HIGH"), not label ("High Risk").
        """
        assert risk_result.risk_level in RISK_LEVELS
        assert risk_result.risk_level in ["LOW", "MEDIUM", "HIGH"]

    def test_risk_result_label_is_display(self, risk_result):
        """risk_label must be the display string."""
        assert risk_result.risk_label in RISK_LABELS.values()
        # Verify consistency
        assert RISK_LABELS[risk_result.risk_level] == risk_result.risk_label

    def test_risk_result_is_high_risk_property(self):
        """Test is_high_risk property."""
        result_high = RiskResult(
            risk_level="HIGH", risk_label="High Risk",
            confidence_pct=80.0, disease_prob=0.8, predicted_label=1
        )
        assert result_high.is_high_risk is True

        result_low = RiskResult(
            risk_level="LOW", risk_label="Low Risk",
            confidence_pct=90.0, disease_prob=0.1, predicted_label=0
        )
        assert result_low.is_high_risk is False

    def test_risk_result_badge_color_property(self, risk_result):
        """
        FIXED: badge_color must use RISK_COLORS hex values.
        """
        expected_color = RISK_COLORS[risk_result.risk_level]
        assert risk_result.badge_color == expected_color
        assert risk_result.badge_color.startswith("#")

    def test_risk_result_badge_icon_property(self, risk_result):
        """badge_icon must use RISK_ICONS."""
        expected_icon = RISK_ICONS[risk_result.risk_level]
        assert risk_result.badge_icon == expected_icon

    def test_risk_result_to_dict_structure(self, risk_result):
        """
        FIXED: to_dict must include risk_label, badge_color, badge_icon.
        """
        d = risk_result.to_dict()
        
        # Required keys
        assert "risk_level" in d
        assert "risk_label" in d
        assert "confidence_pct" in d
        assert "disease_prob" in d
        assert "predicted_label" in d
        assert "top_risk_factors" in d
        assert "explanation" in d
        assert "model_version" in d
        assert "badge_color" in d
        assert "badge_icon" in d
        assert "feature_contributions" in d

        # Verify values
        assert d["risk_level"] == risk_result.risk_level
        assert d["risk_label"] == risk_result.risk_label
        assert d["badge_color"] == risk_result.badge_color
        assert d["badge_icon"] == risk_result.badge_icon

    def test_risk_result_to_dict_rounding(self, risk_result):
        """Test numeric rounding in to_dict."""
        d = risk_result.to_dict()
        # confidence_pct rounded to 1 decimal
        assert d["confidence_pct"] == round(risk_result.confidence_pct, 1)
        # disease_prob rounded to 4 decimals
        assert d["disease_prob"] == round(risk_result.disease_prob, 4)


# ─────────────────────────────────────────────────────────────────────────────
# Probability to Risk Level Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestProbabilityToRiskLevel:
    def test_low_risk_boundary(self):
        """Test LOW risk range [0.0, 0.35)."""
        level, conf = _probability_to_risk_level(0.0)
        assert level == "LOW"
        
        level, conf = _probability_to_risk_level(0.34)
        assert level == "LOW"
        
        level, conf = _probability_to_risk_level(0.3499)
        assert level == "LOW"

    def test_medium_risk_boundary(self):
        """Test MEDIUM risk range [0.35, 0.65)."""
        level, conf = _probability_to_risk_level(0.35)
        assert level == "MEDIUM"
        
        level, conf = _probability_to_risk_level(0.50)
        assert level == "MEDIUM"
        
        level, conf = _probability_to_risk_level(0.64)
        assert level == "MEDIUM"

    def test_high_risk_boundary(self):
        """Test HIGH risk range [0.65, 1.0]."""
        level, conf = _probability_to_risk_level(0.65)
        assert level == "HIGH"
        
        level, conf = _probability_to_risk_level(0.80)
        assert level == "HIGH"
        
        level, conf = _probability_to_risk_level(1.0)
        assert level == "HIGH"

    def test_confidence_low_risk(self):
        """Test LOW risk confidence formula: (1 - prob) * 100."""
        _, conf = _probability_to_risk_level(0.0)
        assert conf == 100.0
        
        _, conf = _probability_to_risk_level(0.20)
        assert conf == 80.0
        
        _, conf = _probability_to_risk_level(0.34)
        assert abs(conf - 66.0) < 0.1  # (1 - 0.34) * 100 = 66.0

    def test_confidence_medium_risk_smooth(self):
        """
        FIXED: MEDIUM confidence should be smooth, no cliff at boundaries.
        Formula: 50 + (distance / (band/2)) * 30
        Band: [0.35, 0.65], mid=0.50, width=0.30
        """
        # At midpoint (0.50), distance=0, conf=50%
        _, conf = _probability_to_risk_level(0.50)
        assert conf == 50.0
        
        # At lower boundary (0.35), distance=0.15, conf=80%
        _, conf = _probability_to_risk_level(0.35)
        assert abs(conf - 80.0) < 0.1
        
        # At upper boundary (0.65), distance=0.15, conf=80%
        _, conf = _probability_to_risk_level(0.6499)
        assert abs(conf - 80.0) < 0.1

    def test_confidence_high_risk(self):
        """Test HIGH risk confidence formula: prob * 100."""
        _, conf = _probability_to_risk_level(0.65)
        assert conf == 65.0
        
        _, conf = _probability_to_risk_level(0.80)
        assert conf == 80.0
        
        _, conf = _probability_to_risk_level(1.0)
        assert conf == 99.9  # Capped at 99.9


# ─────────────────────────────────────────────────────────────────────────────
# Feature Contributions Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureContributions:
    def test_global_contributions(self, trained_model):
        """Test global feature contributions (no patient input)."""
        contributions = _get_feature_contributions(trained_model)
        
        assert len(contributions) == len(FEATURE_COLUMNS)
        # Sorted descending
        importances = [imp for _, imp in contributions]
        assert importances == sorted(importances, reverse=True)
        # Sum to 1.0
        assert abs(sum(importances) - 1.0) < 1e-6

    def test_patient_specific_contributions(self, trained_model, sample_patient):
        """
        FIXED: Patient-specific contributions should weight by feature values.
        """
        scaled_input = preprocess_patient(sample_patient)
        
        global_contribs = _get_feature_contributions(trained_model)
        patient_contribs = _get_feature_contributions(trained_model, scaled_input=scaled_input)
        
        # Contributions should differ when weighted by patient values
        global_imp = {f: i for f, i in global_contribs}
        patient_imp = {f: i for f, i in patient_contribs}
        
        # At least some features should have different importance
        differences = [
            abs(global_imp[f] - patient_imp[f])
            for f in FEATURE_COLUMNS
        ]
        assert sum(differences) > 0.01  # Should be noticeably different

    def test_patient_zero_input_fallback(self, trained_model):
        """
        If patient input is all zeros, should fall back to global importances.
        """
        zero_input = np.zeros((1, len(FEATURE_COLUMNS)))
        
        global_contribs = _get_feature_contributions(trained_model)
        zero_contribs = _get_feature_contributions(trained_model, scaled_input=zero_input)
        
        # Should be identical (fallback triggered)
        for (f1, i1), (f2, i2) in zip(global_contribs, zero_contribs):
            assert f1 == f2
            assert abs(i1 - i2) < 1e-9

    def test_top_risk_factors(self):
        """Test _top_risk_factors extraction."""
        contributions = [
            ("Age", 0.20),
            ("Cholesterol", 0.15),
            ("Blood Pressure", 0.10),
            ("Heart Rate", 0.05),
        ]
        
        top = _top_risk_factors(contributions, n=2)
        assert top == ["Age", "Cholesterol"]
        
        top = _top_risk_factors(contributions, n=3)
        assert top == ["Age", "Cholesterol", "Blood Pressure"]


# ─────────────────────────────────────────────────────────────────────────────
# Prediction Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPrediction:
    def test_predict_risk_returns_result(self, sample_patient):
        """Test predict_risk returns RiskResult."""
        result = predict_risk(sample_patient)
        assert isinstance(result, RiskResult)

    def test_predict_risk_consistency(self, sample_patient):
        """Test prediction is deterministic."""
        result1 = predict_risk(sample_patient)
        result2 = predict_risk(sample_patient)
        
        assert result1.risk_level == result2.risk_level
        assert result1.disease_prob == result2.disease_prob
        assert result1.predicted_label == result2.predicted_label

    def test_predict_risk_explanation_generated(self, risk_result):
        """Test explanation is generated and contains key info."""
        assert len(risk_result.explanation) > 0
        assert "Assessment" in risk_result.explanation
        assert "confidence" in risk_result.explanation.lower()
        assert "probability" in risk_result.explanation.lower()

    def test_explain_prediction_wrapper(self, sample_patient):
        """Test explain_prediction returns explanation string."""
        explanation = explain_prediction(sample_patient)
        assert isinstance(explanation, str)
        assert len(explanation) > 0

    def test_get_feature_importances(self):
        """Test get_feature_importances returns sorted list."""
        importances = get_feature_importances()
        assert len(importances) == len(FEATURE_COLUMNS)
        values = [v for _, v in importances]
        assert values == sorted(values, reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# Batch Prediction Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestBatchPrediction:
    def test_batch_predict_returns_list(self, sample_patient):
        """Test batch_predict returns list of RiskResults."""
        patients = [sample_patient, sample_patient.copy()]
        results = batch_predict(patients)
        
        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, RiskResult) for r in results)

    def test_batch_predict_order_preserved(self, sample_patient):
        """Test batch results maintain input order."""
        patient1 = sample_patient.copy()
        patient1["age"] = 30  # Likely low risk
        
        patient2 = sample_patient.copy()
        patient2["age"] = 70  # Likely higher risk
        
        results = batch_predict([patient1, patient2])
        
        # Order should match input
        assert results[0].age == 30
        assert results[1].age == 70

    def test_batch_predict_empty_list(self):
        """Test batch_predict with empty list."""
        results = batch_predict([])
        assert results == []

    def test_batch_predict_loads_model_once(self, sample_patient, monkeypatch):
        """
        FIXED: batch_predict should load model only once.
        """
        load_count = 0
        original_load = load_model
        
        def mock_load():
            nonlocal load_count
            load_count += 1
            return original_load()
        
        monkeypatch.setattr("core.risk_model.load_model", mock_load)
        
        patients = [sample_patient, sample_patient.copy(), sample_patient.copy()]
        batch_predict(patients)
        
        assert load_count == 1


# ─────────────────────────────────────────────────────────────────────────────
# Training Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTraining:
    def test_train_model_returns_model_and_metrics(self):
        """Test train_model returns tuple of (model, metrics)."""
        if not os.path.exists(HEART_DATA_PATH):
            pytest.skip("heart_data.csv not found")
        
        model, metrics = train_model()
        
        assert isinstance(model, RandomForestClassifier)
        assert isinstance(metrics, dict)

    def test_train_model_metrics_keys(self):
        """Test metrics dict contains expected keys."""
        if not os.path.exists(HEART_DATA_PATH):
            pytest.skip("heart_data.csv not found")
        
        _, metrics = train_model()
        
        required_keys = [
            "accuracy", "precision", "recall", "f1", "roc_auc",
            "train_size", "test_size",
            "cv_roc_auc_mean", "cv_roc_auc_std",  # FIXED: CV metrics added
        ]
        
        for key in required_keys:
            assert key in metrics, f"Missing metric key: {key}"

    def test_train_model_cv_metrics_valid(self):
        """Test CV metrics are reasonable values."""
        if not os.path.exists(HEART_DATA_PATH):
            pytest.skip("heart_data.csv not found")
        
        _, metrics = train_model()
        
        assert 0.0 <= metrics["cv_roc_auc_mean"] <= 1.0
        assert metrics["cv_roc_auc_std"] >= 0.0
        assert metrics["cv_roc_auc_std"] < 0.5  # Std should be reasonable

    def test_train_model_persists_to_disk(self):
        """Test train_model saves model to MODEL_SAVE_PATH."""
        if not os.path.exists(HEART_DATA_PATH):
            pytest.skip("heart_data.csv not found")
        
        # Remove existing model
        if os.path.exists(MODEL_SAVE_PATH):
            os.remove(MODEL_SAVE_PATH)
        
        train_model()
        
        assert os.path.exists(MODEL_SAVE_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# Metadata and Maintenance Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMetadataAndMaintenance:
    def test_get_model_metadata(self):
        """Test get_model_metadata returns expected structure."""
        metadata = get_model_metadata()
        
        required_keys = [
            "model_type", "n_estimators", "max_depth",
            "n_features", "model_version", "top_features",
            "model_path", "model_exists",
        ]
        
        for key in required_keys:
            assert key in metadata
        
        assert metadata["model_type"] == "RandomForestClassifier"
        assert metadata["n_estimators"] == RF_N_ESTIMATORS
        assert metadata["max_depth"] == RF_MAX_DEPTH
        assert metadata["n_features"] == len(FEATURE_COLUMNS)
        assert len(metadata["top_features"]) == 3

    def test_retrain_if_stale_model_missing(self, monkeypatch):
        """Test retrain_if_stale trains if model doesn't exist."""
        monkeypatch.setattr("os.path.exists", lambda x: False)
        
        trained = False
        def mock_train():
            nonlocal trained
            trained = True
            return None, {}
        
        monkeypatch.setattr("core.risk_model.train_model", mock_train)
        
        result = retrain_if_stale(max_age_days=30)
        
        assert result is True
        assert trained is True

    def test_retrain_if_stale_fresh_model(self, monkeypatch):
        """Test retrain_if_stale returns False if model is fresh."""
        monkeypatch.setattr("os.path.exists", lambda x: True)
        monkeypatch.setattr("time.time", lambda: 1000000)
        monkeypatch.setattr("os.path.getmtime", lambda x: 999999)  # 1 second old
        
        trained = False
        def mock_train():
            nonlocal trained
            trained = True
            return None, {}
        
        monkeypatch.setattr("core.risk_model.train_model", mock_train)
        
        result = retrain_if_stale(max_age_days=30)
        
        assert result is False
        assert trained is False

    def test_retrain_if_stale_old_model(self, monkeypatch):
        """Test retrain_if_stale retrains if model is stale."""
        monkeypatch.setattr("os.path.exists", lambda x: True)
        monkeypatch.setattr("time.time", lambda: 1000000)
        # Model is 40 days old
        monkeypatch.setattr("os.path.getmtime", lambda x: 1000000 - (40 * 86400))
        
        trained = False
        def mock_train():
            nonlocal trained
            trained = True
            return None, {}
        
        monkeypatch.setattr("core.risk_model.train_model", mock_train)
        
        result = retrain_if_stale(max_age_days=30)
        
        assert result is True
        assert trained is True


# ─────────────────────────────────────────────────────────────────────────────
# Integration Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegration:
    def test_full_prediction_pipeline(self, sample_patient):
        """Test end-to-end prediction pipeline."""
        result = predict_risk(sample_patient)
        
        # Verify all components
        assert result.risk_level in RISK_LEVELS
        assert result.risk_label == RISK_LABELS[result.risk_level]
        assert 0.0 <= result.disease_prob <= 1.0
        assert 0.0 <= result.confidence_pct <= 100.0
        assert result.predicted_label in [0, 1]
        assert len(result.feature_contributions) == len(FEATURE_COLUMNS)
        assert len(result.top_risk_factors) == 3
        assert len(result.explanation) > 0
        
        # Verify to_dict
        d = result.to_dict()
        assert d["risk_level"] == result.risk_level
        assert d["risk_label"] == result.risk_label
        assert d["badge_color"] == RISK_COLORS[result.risk_level]
        assert d["badge_icon"] == RISK_ICONS[result.risk_level]

    def test_prediction_with_missing_fields(self, sample_patient):
        """Test prediction handles missing fields via imputation."""
        incomplete_patient = sample_patient.copy()
        del incomplete_patient["chol"]
        del incomplete_patient["thalach"]
        
        # Should not raise, imputation handles missing
        result = predict_risk(incomplete_patient)
        assert isinstance(result, RiskResult)