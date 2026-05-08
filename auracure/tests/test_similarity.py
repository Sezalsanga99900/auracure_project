# =============================================================================
# tests/test_similarity.py
# AuraEcho+ — Similarity Engine Tests
#
# Coverage:
#     • SimilarCase dataclass
#     • Reference data loading and caching
#     • Cosine distance to similarity conversion
#     • Risk derivation (severity score → RISK_LEVELS thresholds)
#     • Summary building (no double "Risk" text)
#     • find_similar_cases (validation, self-skip, rank gaps, k handling)
#     • get_similarity_stats
#     • find_similar_by_risk (uses SIMILARITY_POOL_SIZE)
#     • get_outcome_distribution
#     • get_similar_cases_summary
#     • compare_patients
#     • Error handling and edge cases
#
# Run:
#     pytest tests/test_similarity.py -v
# =============================================================================

import pytest
import os
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from core.similarity import (
    SimilarCase,
    _load_reference_data,
    _cosine_distance_to_similarity,
    _derive_risk_level,
    _build_summary,
    find_similar_cases,
    get_similarity_stats,
    find_similar_by_risk,
    get_outcome_distribution,
    get_similar_cases_summary,
    compare_patients,
    preload_reference_data,
    _reference_df,
    _reference_X,
    _knn_model,
    _raw_labels,
)
from utils.constants import (
    HEART_DATA_PATH,
    RISK_LEVELS,
    RISK_LABELS,
    SIMILARITY_POOL_SIZE,
    KNN_N_NEIGHBORS,
    FEATURE_COLUMNS,
)
from utils.validators import validate_patient


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def reference_data_loaded():
    """
    Load reference data once per test session.
    Ensures _reference_X, _knn_model, etc. are populated.
    """
    if not os.path.exists(HEART_DATA_PATH):
        pytest.skip("heart_data.csv not found")
    
    preload_reference_data()
    
    # Verify loaded
    assert _reference_df is not None
    assert _reference_X is not None
    assert _knn_model is not None
    assert _raw_labels is not None
    
    return {
        "df": _reference_df,
        "X": _reference_X,
        "model": _knn_model,
        "labels": _raw_labels,
    }


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
def sample_patient_in_reference(reference_data_loaded):
    """
    Returns a patient that exists in the reference dataset.
    Useful for testing self-skip logic.
    """
    # Get first row from reference
    row = reference_data_loaded["df"].iloc[0]
    patient = {col: row[col] for col in FEATURE_COLUMNS}
    patient["name"] = "Reference Patient"
    return patient


# ─────────────────────────────────────────────────────────────────────────────
# SimilarCase Dataclass Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSimilarCase:
    def test_similar_case_fields(self):
        """Test all fields are present."""
        case = SimilarCase(
            rank=1,
            similarity_pct=85.5,
            patient_index=42,
            features={"age": 55, "sex": 1},
            outcome="Disease",
            risk_level="HIGH",
            age=55,
            sex="Male",
            summary="#1 Match — 55yr Male | Disease | High Risk | 85.5% similar",
        )
        
        assert case.rank == 1
        assert case.similarity_pct == 85.5
        assert case.patient_index == 42
        assert case.outcome == "Disease"
        assert case.risk_level == "HIGH"
        assert case.age == 55
        assert case.sex == "Male"

    def test_similar_case_to_dict(self):
        """Test to_dict returns expected structure."""
        case = SimilarCase(
            rank=1,
            similarity_pct=85.5,
            patient_index=42,
            features={"age": 55},
            outcome="Disease",
            risk_level="HIGH",
            age=55,
            sex="Male",
        )
        
        d = case.to_dict()
        
        assert d["rank"] == 1
        assert d["similarity_pct"] == 85.5
        assert d["patient_index"] == 42
        assert d["outcome"] == "Disease"
        assert d["risk_level"] == "HIGH"
        assert d["risk_label"] == RISK_LABELS["HIGH"]
        assert d["risk_color"] is not None
        assert d["risk_icon"] is not None
        assert d["age"] == 55
        assert d["sex"] == "Male"
        assert "features" in d


# ─────────────────────────────────────────────────────────────────────────────
# Internal Helper Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestInternalHelpers:
    def test_cosine_distance_to_similarity(self):
        """Test distance → similarity conversion."""
        # distance=0 → similarity=100
        assert _cosine_distance_to_similarity(0.0) == 100.0
        
        # distance=1 → similarity=0
        assert _cosine_distance_to_similarity(1.0) == 0.0
        
        # distance=0.5 → similarity=50
        assert _cosine_distance_to_similarity(0.5) == 50.0
        
        # Clipping: distance < 0 → similarity=100
        assert _cosine_distance_to_similarity(-0.5) == 100.0
        
        # Clipping: distance > 1 → similarity=0
        assert _cosine_distance_to_similarity(1.5) == 0.0

    def test_derive_risk_level_no_disease(self):
        """Test label=0 always returns LOW."""
        features = {"thalach": 100, "oldpeak": 3.0, "ca": 2}  # High severity
        level = _derive_risk_level(0, features)
        assert level == "LOW"

    def test_derive_risk_level_uses_thresholds(self):
        """
        FIXED: _derive_risk_level must use RISK_LEVELS thresholds
        via normalized severity score, not hardcoded values.
        """
        # Max severity: thalach<120 (+2), oldpeak>2.0 (+2), ca>=2 (+2) = 6
        # Normalized = 6/6 = 1.0 → HIGH
        features = {"thalach": 100, "oldpeak": 3.0, "ca": 2}
        level = _derive_risk_level(1, features)
        assert level == "HIGH"
        
        # Medium severity: thalach<140 (+1), oldpeak>1.0 (+1), ca=1 (+1) = 3
        # Normalized = 3/6 = 0.5 → MEDIUM (since 0.35 <= 0.5 < 0.65)
        features = {"thalach": 130, "oldpeak": 1.5, "ca": 1}
        level = _derive_risk_level(1, features)
        assert level == "MEDIUM"
        
        # Low severity: all normal = 0
        # Normalized = 0 → LOW
        features = {"thalach": 160, "oldpeak": 0.0, "ca": 0}
        level = _derive_risk_level(1, features)
        assert level == "LOW"

    def test_build_summary_no_double_risk(self):
        """
        FIXED: Summary must not contain "Risk Risk".
        RISK_LABELS already contains "Risk", so we shouldn't append it.
        """
        summary = _build_summary(
            rank=1,
            age=55,
            sex="Male",
            outcome="Disease",
            risk_level="HIGH",
            similarity_pct=85.5,
        )
        
        assert "Risk Risk" not in summary
        assert "High Risk" in summary
        assert "#1 Match" in summary
        assert "85.5% similar" in summary


# ─────────────────────────────────────────────────────────────────────────────
# Find Similar Cases Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFindSimilarCases:
    def test_returns_correct_count(self, sample_patient, reference_data_loaded):
        """Test returns k cases."""
        k = 5
        cases = find_similar_cases(sample_patient, k=k)
        
        assert len(cases) == k
        assert all(isinstance(c, SimilarCase) for c in cases)

    def test_ranks_are_sequential(self, sample_patient, reference_data_loaded):
        """Test ranks are 1, 2, 3... without gaps."""
        cases = find_similar_cases(sample_patient, k=10)
        
        ranks = [c.rank for c in cases]
        assert ranks == list(range(1, len(cases) + 1))

    def test_self_skip_no_rank_gap(self, sample_patient_in_reference, reference_data_loaded):
        """
        FIXED: When query patient is in reference set, self should be skipped
        without creating rank gaps. Ranks should still be 1, 2, 3...
        """
        cases = find_similar_cases(sample_patient_in_reference, k=5)
        
        # Ranks should be sequential
        ranks = [c.rank for c in cases]
        assert ranks == list(range(1, len(cases) + 1))
        
        # Similarity should be high but not 100% (self was skipped)
        # First case should be very similar but not exact match
        assert cases[0].similarity_pct < 100.0

    def test_similarity_sorted_descending(self, sample_patient, reference_data_loaded):
        """Test results sorted by similarity descending."""
        cases = find_similar_cases(sample_patient, k=10)
        
        similarities = [c.similarity_pct for c in cases]
        assert similarities == sorted(similarities, reverse=True)

    def test_k_greater_than_reference(self, sample_patient, reference_data_loaded):
        """Test k > n_ref reduces k gracefully."""
        n_ref = len(reference_data_loaded["df"])
        k = n_ref + 100
        
        cases = find_similar_cases(sample_patient, k=k)
        
        # Should return at most n_ref cases
        assert len(cases) <= n_ref

    def test_validation_with_check_structure(self, reference_data_loaded):
        """
        FIXED: validate_patient called with check_structure=True.
        Should raise ValueError if required fields missing.
        """
        incomplete_patient = {"age": 55}  # Missing many fields
        
        with pytest.raises(ValueError, match="Invalid patient input"):
            find_similar_cases(incomplete_patient, k=3)

    def test_validation_error_message(self, reference_data_loaded):
        """Test validation error includes formatted errors."""
        invalid_patient = {"age": 200}  # Out of range
        
        with pytest.raises(ValueError, match="Age"):
            find_similar_cases(invalid_patient, k=3)

    def test_results_contain_expected_fields(self, sample_patient, reference_data_loaded):
        """Test each case has all expected fields."""
        cases = find_similar_cases(sample_patient, k=3)
        
        for case in cases:
            assert case.rank > 0
            assert 0.0 <= case.similarity_pct <= 100.0
            assert case.patient_index >= 0
            assert isinstance(case.features, dict)
            assert case.outcome in ["Disease", "No Disease"]
            assert case.risk_level in RISK_LEVELS
            assert isinstance(case.age, int)
            assert isinstance(case.sex, str)
            assert isinstance(case.summary, str)


# ─────────────────────────────────────────────────────────────────────────────
# Similarity Stats Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSimilarityStats:
    def test_stats_structure(self, sample_patient, reference_data_loaded):
        """Test stats dict has expected keys."""
        stats = get_similarity_stats(sample_patient)
        
        required_keys = ["min_sim", "max_sim", "mean_sim", "median_sim", "pct_above_80"]
        for key in required_keys:
            assert key in stats

    def test_stats_values_reasonable(self, sample_patient, reference_data_loaded):
        """Test stats values are in expected ranges."""
        stats = get_similarity_stats(sample_patient)
        
        assert 0.0 <= stats["min_sim"] <= 100.0
        assert 0.0 <= stats["max_sim"] <= 100.0
        assert stats["min_sim"] <= stats["mean_sim"] <= stats["max_sim"]
        assert 0.0 <= stats["pct_above_80"] <= 100.0

    def test_stats_validation(self, reference_data_loaded):
        """Test stats validates input."""
        incomplete = {"age": 55}
        
        with pytest.raises(ValueError):
            get_similarity_stats(incomplete)


# ─────────────────────────────────────────────────────────────────────────────
# Find By Risk Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFindByRisk:
    def test_filters_by_risk(self, sample_patient, reference_data_loaded):
        """Test returns only cases matching risk_filter."""
        cases = find_similar_by_risk(sample_patient, risk_filter="HIGH", k=5)
        
        for case in cases:
            assert case.risk_level == "HIGH"

    def test_uses_similarity_pool_size(self, sample_patient, reference_data_loaded, monkeypatch):
        """
        FIXED: Must use SIMILARITY_POOL_SIZE constant, not hardcoded 50.
        """
        # Patch SIMILARITY_POOL_SIZE to a small value
        monkeypatch.setattr("core.similarity.SIMILARITY_POOL_SIZE", 5)
        
        # Mock find_similar_cases to check k value
        called_with_k = []
        original_find = find_similar_cases
        
        def mock_find(patient, k):
            called_with_k.append(k)
            return original_find(patient, k)
        
        monkeypatch.setattr("core.similarity.find_similar_cases", mock_find)
        
        find_similar_by_risk(sample_patient, risk_filter="LOW", k=3)
        
        # Should have called with k=5 (SIMILARITY_POOL_SIZE)
        assert called_with_k[0] == 5

    def test_invalid_risk_filter(self, sample_patient, reference_data_loaded):
        """Test raises error for invalid risk_filter."""
        with pytest.raises(ValueError, match="risk_filter must be one of"):
            find_similar_by_risk(sample_patient, risk_filter="INVALID", k=3)

    def test_returns_up_to_k(self, sample_patient, reference_data_loaded):
        """Test returns at most k results after filtering."""
        cases = find_similar_by_risk(sample_patient, risk_filter="MEDIUM", k=3)
        
        assert len(cases) <= 3
        for case in cases:
            assert case.risk_level == "MEDIUM"


# ─────────────────────────────────────────────────────────────────────────────
# Outcome Distribution Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestOutcomeDistribution:
    def test_distribution_structure(self, sample_patient, reference_data_loaded):
        """Test returns expected dict structure."""
        dist = get_outcome_distribution(sample_patient, k=20)
        
        required_keys = ["disease_count", "no_disease_count", "disease_pct", "total"]
        for key in required_keys:
            assert key in dist

    def test_distribution_values(self, sample_patient, reference_data_loaded):
        """Test values are consistent."""
        dist = get_outcome_distribution(sample_patient, k=20)
        
        assert dist["total"] == 20
        assert dist["disease_count"] + dist["no_disease_count"] == dist["total"]
        assert 0.0 <= dist["disease_pct"] <= 100.0
        
        # Verify percentage calculation
        expected_pct = round(dist["disease_count"] / dist["total"] * 100, 1)
        assert dist["disease_pct"] == expected_pct

    def test_distribution_empty(self, reference_data_loaded):
        """Test handles empty results."""
        # This shouldn't happen with valid data, but test the logic
        with patch("core.similarity.find_similar_cases", return_value=[]):
            dist = get_outcome_distribution({"age": 55}, k=5)
            
            assert dist["total"] == 0
            assert dist["disease_count"] == 0
            assert dist["disease_pct"] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Summary and Compare Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSummaryAndCompare:
    def test_summary_structure(self, sample_patient, reference_data_loaded):
        """Test summary returns expected structure."""
        summary = get_similar_cases_summary(sample_patient, k=5)
        
        required_keys = [
            "cases", "outcome_dist", "top_match_sim",
            "avg_similarity", "dominant_outcome",
        ]
        for key in required_keys:
            assert key in summary
        
        assert len(summary["cases"]) == 5
        assert isinstance(summary["outcome_dist"], dict)
        assert isinstance(summary["top_match_sim"], float)
        assert isinstance(summary["avg_similarity"], float)
        assert summary["dominant_outcome"] in ["Disease", "No Disease"]

    def test_summary_avg_calculation(self, sample_patient, reference_data_loaded):
        """Test avg_similarity is correctly calculated."""
        summary = get_similar_cases_summary(sample_patient, k=5)
        
        cases = summary["cases"]
        expected_avg = round(sum(c["similarity_pct"] for c in cases) / len(cases), 1)
        
        assert summary["avg_similarity"] == expected_avg

    def test_compare_patients(self, sample_patient, reference_data_loaded):
        """Test compare_patients returns similarity and interpretation."""
        patient_a = sample_patient
        patient_b = sample_patient.copy()
        patient_b["age"] = 56  # Slightly different
        
        result = compare_patients(patient_a, patient_b)
        
        assert "similarity_pct" in result
        assert "interpretation" in result
        assert 0.0 <= result["similarity_pct"] <= 100.0
        assert isinstance(result["interpretation"], str)
        
        # Very similar patients should have high similarity
        assert result["similarity_pct"] > 80.0

    def test_compare_patients_interpretation_levels(self, reference_data_loaded):
        """Test interpretation matches similarity thresholds."""
        # Mock different similarity levels
        test_cases = [
            (85.0, "Very similar profiles"),
            (70.0, "Moderately similar"),
            (40.0, "Different clinical profiles"),
        ]
        
        for sim, expected_interp in test_cases:
            with patch("core.similarity.cosine_similarity", return_value=[[sim / 100.0]]):
                result = compare_patients({"age": 55}, {"age": 55})
                assert result["interpretation"] == expected_interp


# ─────────────────────────────────────────────────────────────────────────────
# Preload and Error Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPreloadAndErrors:
    def test_preload_reference_data(self, reference_data_loaded):
        """Test preload loads all caches."""
        assert _reference_df is not None
        assert _reference_X is not None
        assert _knn_model is not None
        assert _raw_labels is not None
        
        # Verify shapes
        assert len(_reference_df) == len(_reference_X)
        assert _reference_X.shape[1] == len(FEATURE_COLUMNS)

    def test_missing_heart_data(self):
        """Test raises error if heart_data.csv missing."""
        with patch("os.path.exists", return_value=False):
            with pytest.raises(FileNotFoundError, match="Reference dataset not found"):
                _load_reference_data()

    def test_invalid_patient_validation(self, reference_data_loaded):
        """Test invalid patient raises ValueError."""
        with pytest.raises(ValueError):
            find_similar_cases({"invalid": "data"}, k=3)