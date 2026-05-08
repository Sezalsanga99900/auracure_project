# =============================================================================
# tests/test_ai.py
# AuraEcho+ — AI Layer Tests
#
# Coverage:
#     • AIResponse dataclass and helpers (parse_sections, fallback, safe_error)
#     • Prompt builder (diagnosis, followup, summary, validation, risk normalization)
#     • Offline AI (Ollama availability, call with retries, stream, fallback)
#     • Online AI (Groq/OpenAI availability, key validation, retries, stream)
#     • Integration (prompt → AI → response parsing)
#
# Run:
#     pytest tests/test_ai.py -v
# =============================================================================

import pytest
import os
import json
import time
import re
from unittest.mock import patch, MagicMock, Mock
from typing import Any, Dict, List

# AI modules
from ai.ai_response import (
    AIResponse,
    parse_sections,
    build_fallback_response,
    safe_error_msg,
)
from ai.prompt_builder import (
    PromptPackage,
    build_diagnosis_prompt,
    build_followup_prompt,
    build_summary_prompt,
    build_alert_prompt,
    validate_prompt_package,
    _normalize_risk_for_prompt,
)
from ai.offline_ai import (
    is_ollama_available,
    get_ollama_status,
    _call_ollama,
    analyze_patient as analyze_offline,
    stream_analysis as stream_offline,
    warmup_model,
    get_model_info as get_ollama_info,
)
from ai.online_ai import (
    is_groq_available,
    is_openai_available,
    get_api_status,
    _call_groq,
    _call_openai,
    analyze_patient as analyze_online,
    stream_analysis as stream_online,
    test_api_connection,
    get_model_info as get_online_info,
)
from utils.constants import (
    RISK_LEVELS,
    RISK_LABELS,
    RISK_COLORS,
    RISK_ICONS,
    CONNECTIVITY_RETRIES,
    LLM_MAX_TOKENS,
)
from utils.helpers import normalize_risk_level


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

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
def sample_risk_result():
    """
    Returns a risk result dict matching RiskResult.to_dict().
    FIXED: Includes risk_level KEY, risk_label, badge_icon, badge_color.
    """
    return {
        "risk_level": "HIGH",           # KEY
        "risk_label": "High Risk",      # LABEL
        "confidence_pct": 85.5,
        "disease_prob": 0.78,
        "predicted_label": 1,
        "top_risk_factors": ["Age", "Cholesterol", "Chest Pain"],
        "explanation": "Assessment (high confidence)...",
        "badge_color": "#e74c3c",
        "badge_icon": "🚨",
        "feature_contributions": [
            {"feature": "Age", "importance": 0.20},
            {"feature": "Cholesterol", "importance": 0.15},
        ],
    }


@pytest.fixture
def sample_similar_cases():
    """Returns a list of similar case dicts."""
    return [
        {
            "rank": 1,
            "similarity_pct": 92.5,
            "patient_index": 42,
            "outcome": "Disease",
            "risk_level": "HIGH",
            "risk_label": "High Risk",
            "risk_color": "#e74c3c",
            "risk_icon": "🚨",
            "age": 58,
            "sex": "Male",
            "summary": "#1 Match — 58yr Male | Disease | High Risk | 92.5% similar",
            "features": {"age": 58, "sex": 1},
        },
        {
            "rank": 2,
            "similarity_pct": 87.3,
            "patient_index": 15,
            "outcome": "No Disease",
            "risk_level": "MEDIUM",
            "risk_label": "Medium Risk",
            "risk_color": "#f39c12",
            "risk_icon": "⚠️",
            "age": 52,
            "sex": "Female",
            "summary": "#2 Match — 52yr Female | No Disease | Medium Risk | 87.3% similar",
            "features": {"age": 52, "sex": 0},
        },
    ]


@pytest.fixture
def mock_ollama_response():
    """Returns a mock Ollama API response."""
    return {
        "model": "llama3",
        "message": {
            "role": "assistant",
            "content": """## 🔍 Clinical Assessment
Patient shows elevated cardiac risk.

## ⚠️ Key Risk Indicators
• Age and cholesterol are concerning.

## 💊 Treatment Recommendations
• Lifestyle changes recommended.
""",
        },
        "prompt_eval_count": 150,
        "eval_count": 80,
        "done": True,
    }


@pytest.fixture
def mock_groq_completion():
    """Returns a mock Groq completion object."""
    mock = MagicMock()
    mock.choices = [MagicMock()]
    mock.choices[0].message.content = """## 🔍 Clinical Assessment
Groq analysis complete.

## ⚠️ Key Risk Indicators
• Elevated risk factors detected.
"""
    mock.usage = MagicMock()
    mock.usage.prompt_tokens = 120
    mock.usage.completion_tokens = 60
    return mock


@pytest.fixture
def mock_openai_completion():
    """Returns a mock OpenAI completion object."""
    mock = MagicMock()
    mock.choices = [MagicMock()]
    mock.choices[0].message.content = """## 🔍 Clinical Assessment
OpenAI analysis complete.

## 💊 Treatment Recommendations
• Follow standard protocols.
"""
    mock.usage = MagicMock()
    mock.usage.prompt_tokens = 130
    mock.usage.completion_tokens = 70
    return mock


# ─────────────────────────────────────────────────────────────────────────────
# AIResponse and Helpers Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAIResponseAndHelpers:
    def test_ai_response_fields(self):
        """Test AIResponse dataclass fields."""
        resp = AIResponse(
            content="Test content",
            source="test",
            model="test-model",
            prompt_tokens=100,
            output_tokens=50,
            latency_ms=250.5,
            success=True,
            sections={"Assessment": "Test"},
        )
        
        assert resp.content == "Test content"
        assert resp.source == "test"
        assert resp.model == "test-model"
        assert resp.prompt_tokens == 100
        assert resp.output_tokens == 50
        assert resp.latency_ms == 250.5
        assert resp.success is True
        assert resp.sections == {"Assessment": "Test"}

    def test_ai_response_to_dict(self):
        """Test to_dict returns expected structure."""
        resp = AIResponse(
            content="Test",
            source="test",
            model="test",
            latency_ms=123.456,
        )
        
        d = resp.to_dict()
        
        assert d["content"] == "Test"
        assert d["latency_ms"] == 123.5  # Rounded
        assert d["success"] is True

    def test_ai_response_is_empty_property(self):
        """
        FIXED: is_empty checks success flag as well as content.
        """
        # Success with content → not empty
        resp = AIResponse(content="Content", source="test", model="test", success=True)
        assert resp.is_empty is False
        
        # Success with whitespace → empty
        resp = AIResponse(content="   ", source="test", model="test", success=True)
        assert resp.is_empty is True
        
        # Failure → empty regardless of content
        resp = AIResponse(content="Error details", source="test", model="test", success=False)
        assert resp.is_empty is True

    def test_parse_sections_basic(self):
        """Test section parsing."""
        text = """## 🔍 Clinical Assessment
Patient is healthy.

## ⚠️ Key Risk Indicators
• None identified.
"""
        sections = parse_sections(text)
        
        assert "Clinical Assessment" in sections
        assert sections["Clinical Assessment"] == "Patient is healthy."
        assert "Key Risk Indicators" in sections
        assert "None identified" in sections["Key Risk Indicators"]

    def test_parse_sections_emoji_regex_preserves_punctuation(self):
        """
        FIXED: Emoji regex must preserve punctuation and digits.
        Old code stripped '&' and other chars.
        """
        text = """## 🏥 Referral & Follow-up
Refer to cardiology.

## 📊 Test Results 2024
Results are normal.
"""
        sections = parse_sections(text)
        
        # FIXED: "Referral & Follow-up" should stay intact
        assert "Referral & Follow-up" in sections
        assert "Referral  Followup" not in sections
        
        # FIXED: "Test Results 2024" should keep digits
        assert "Test Results 2024" in sections

    def test_build_fallback_response_structure(self):
        """Test fallback response structure."""
        resp = build_fallback_response(
            error="Ollama not running",
            patient_name="John Doe",
            risk_level="HIGH",
        )
        
        assert resp.success is False
        assert resp.error == "Ollama not running"
        assert resp.source == "rule_based_fallback"
        assert "John Doe" in resp.content
        assert "High Risk" in resp.content
        assert "🚨" in resp.content
        assert len(resp.sections) > 0

    def test_build_fallback_response_risk_normalization(self):
        """
        FIXED: Fallback must normalize risk level (key vs label).
        """
        # Pass label instead of key
        resp = build_fallback_response(
            error="Test",
            risk_level="High Risk",  # Label
        )
        
        # Should still find HIGH and use correct icon/label
        assert "High Risk" in resp.content
        assert "🚨" in resp.content
        
        # Pass key
        resp2 = build_fallback_response(
            error="Test",
            risk_level="HIGH",  # Key
        )
        
        assert resp2.content == resp.content  # Same output

    def test_safe_error_msg_sanitization(self):
        """
        FIXED: safe_error_msg must sanitize technical errors.
        """
        # Rate limit
        msg = safe_error_msg(Exception("429 Rate limit exceeded"), "Groq")
        assert "rate limit" in msg.lower()
        assert "429" not in msg  # No raw codes
        
        # Timeout
        msg = safe_error_msg(Exception("Connection timed out"), "Ollama")
        assert "timed out" in msg.lower()
        
        # API key
        msg = safe_error_msg(Exception("Invalid API key: sk-xxx"), "OpenAI")
        assert "api key invalid" in msg.lower()
        assert "sk-xxx" not in msg  # No key leakage
        
        # Generic
        msg = safe_error_msg(Exception("Internal server error"), "AI")
        assert "temporarily unavailable" in msg.lower()
        assert "internal server error" not in msg.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Prompt Builder Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPromptBuilder:
    def test_build_diagnosis_prompt_returns_package(self, sample_patient, sample_risk_result, sample_similar_cases):
        """Test diagnosis prompt returns PromptPackage."""
        pkg = build_diagnosis_prompt(
            patient=sample_patient,
            risk_result=sample_risk_result,
            similar_cases=sample_similar_cases,
        )
        
        assert isinstance(pkg, PromptPackage)
        assert pkg.prompt_type == "diagnosis"
        assert len(pkg.system_prompt) > 0
        assert len(pkg.user_prompt) > 0
        assert pkg.token_estimate > 0

    def test_diagnosis_prompt_risk_normalization(self, sample_patient, sample_risk_result):
        """
        FIXED: Prompt builder must normalize risk level.
        """
        # Pass label instead of key
        risk_with_label = sample_risk_result.copy()
        risk_with_label["risk_level"] = "High Risk"
        
        pkg = build_diagnosis_prompt(sample_patient, risk_result=risk_with_label)
        
        # System prompt should still get HIGH context
        assert "URGENT assessment" in pkg.system_prompt
        
        # Metadata should have normalized key
        assert pkg.metadata["risk_level"] == "HIGH"

    def test_diagnosis_prompt_sex_code_none_safety(self, sample_patient, sample_risk_result):
        """
        FIXED: sex_code None should not crash.
        """
        patient = sample_patient.copy()
        patient["sex"] = None
        
        # Should not raise
        pkg = build_diagnosis_prompt(patient, risk_result=sample_risk_result)
        
        assert "Sex: Unknown" in pkg.user_prompt

    def test_build_followup_prompt_truncation(self, sample_patient):
        """
        FIXED: Follow-up prompt uses helpers.truncate for clean truncation.
        """
        history = [
            {"role": "user", "content": "x" * 500},
            {"role": "assistant", "content": "y" * 500},
        ]
        
        pkg = build_followup_prompt(
            question="Test?",
            patient=sample_patient,
            conversation_history=history,
        )
        
        # Content should be truncated cleanly
        assert len(pkg.user_prompt) < 2000

    def test_validate_prompt_package_valid(self, sample_patient, sample_risk_result):
        """Test validation passes for valid package."""
        pkg = build_diagnosis_prompt(sample_patient, sample_risk_result)
        
        ok, msg = validate_prompt_package(pkg)
        
        assert ok is True
        assert msg == ""

    def test_validate_prompt_package_empty_system(self):
        """Test validation fails for empty system prompt."""
        pkg = PromptPackage(
            system_prompt="",
            user_prompt="Test",
            prompt_type="diagnosis",
        )
        
        ok, msg = validate_prompt_package(pkg)
        
        assert ok is False
        assert "System prompt is empty" in msg

    def test_validate_prompt_package_token_limit(self, sample_patient, sample_risk_result):
        """Test validation fails if tokens exceed limit."""
        pkg = build_diagnosis_prompt(sample_patient, sample_risk_result)
        
        # Artificially inflate token estimate
        pkg.token_estimate = LLM_MAX_TOKENS + 1000
        
        ok, msg = validate_prompt_package(pkg)
        
        assert ok is False
        assert "too long" in msg.lower()

    def test_normalize_risk_for_prompt_helper(self, sample_risk_result):
        """Test _normalize_risk_for_prompt handles key/label."""
        # Key input
        level, label, icon = _normalize_risk_for_prompt(sample_risk_result)
        assert level == "HIGH"
        assert label == "High Risk"
        assert icon == "🚨"
        
        # Label input
        risk_label_input = sample_risk_result.copy()
        risk_label_input["risk_level"] = "High Risk"
        level, label, icon = _normalize_risk_for_prompt(risk_label_input)
        assert level == "HIGH"
        
        # None input
        level, label, icon = _normalize_risk_for_prompt(None)
        assert level == "MEDIUM"  # Default


# ─────────────────────────────────────────────────────────────────────────────
# Offline AI Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestOfflineAI:
    @patch("ai.offline_ai.requests.get")
    def test_is_ollama_available_success(self, mock_get):
        """Test Ollama availability check."""
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "models": [{"name": "llama3:latest"}]
        }
        
        assert is_ollama_available() is True

    @patch("ai.offline_ai.requests.get")
    def test_is_ollama_available_model_missing(self, mock_get):
        """Test availability fails if model not found."""
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "models": [{"name": "mistral:latest"}]
        }
        
        assert is_ollama_available() is False

    @patch("ai.offline_ai.requests.post")
    def test_call_ollama_retry_logic(self, mock_post):
        """
        FIXED: _call_ollama must implement retry logic.
        """
        # Fail twice, succeed on third
        mock_post.side_effect = [
            Exception("Connection refused"),
            Exception("Connection refused"),
            MagicMock(json=lambda: {"message": {"content": "OK"}}),
        ]
        
        with patch("ai.offline_ai.is_ollama_available", return_value=True):
            result = _call_ollama("sys", "user", retries=3)
        
        assert mock_post.call_count == 3
        assert result["message"]["content"] == "OK"

    @patch("ai.offline_ai.requests.post")
    def test_call_ollama_max_retries_exceeded(self, mock_post):
        """Test call fails after max retries."""
        mock_post.side_effect = Exception("Connection refused")
        
        with patch("ai.offline_ai.is_ollama_available", return_value=True):
            with pytest.raises(Exception):
                _call_ollama("sys", "user", retries=2)
        
        assert mock_post.call_count == 2

    @patch("ai.offline_ai._call_ollama")
    def test_analyze_patient_success(self, mock_call, sample_patient, sample_risk_result, sample_similar_cases):
        """Test analyze_patient returns AIResponse."""
        mock_call.return_value = {
            "message": {"content": "## 🔍 Assessment\nPatient OK."},
            "prompt_eval_count": 100,
            "eval_count": 50,
        }
        
        with patch("ai.offline_ai.is_ollama_available", return_value=True):
            resp = analyze_offline(
                sample_patient,
                risk_result=sample_risk_result,
                similar_cases=sample_similar_cases,
            )
        
        assert resp.success is True
        assert resp.source == "offline_llama3"
        assert "Assessment" in resp.sections

    @patch("ai.offline_ai.is_ollama_available")
    def test_analyze_patient_fallback(self, mock_avail, sample_patient, sample_risk_result):
        """Test analyze_patient falls back when Ollama unavailable."""
        mock_avail.return_value = False
        
        resp = analyze_offline(sample_patient, risk_result=sample_risk_result)
        
        assert resp.success is False
        assert resp.source == "rule_based_fallback"
        assert "Ollama not running" in resp.error

    @patch("ai.offline_ai._call_ollama")
    def test_stream_analysis_safe_errors(self, mock_call, sample_patient):
        """
        FIXED: stream_analysis must use safe_error_msg.
        """
        mock_call.side_effect = Exception("429 Rate limit")
        
        with patch("ai.offline_ai.is_ollama_available", return_value=True):
            chunks = list(stream_offline(sample_patient))
        
        # Should yield sanitized error
        assert len(chunks) > 0
        assert "rate limit" in chunks[0].lower()
        assert "429" not in chunks[0]

    @patch("ai.offline_ai._call_ollama")
    def test_warmup_model(self, mock_call):
        """Test warmup_model calls Ollama."""
        mock_call.return_value = {"message": {"content": "Ready"}}
        
        with patch("ai.offline_ai.is_ollama_available", return_value=True):
            result = warmup_model()
        
        assert result is True
        mock_call.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# Online AI Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestOnlineAI:
    def test_is_groq_available_key_validation(self, monkeypatch):
        """
        FIXED: is_groq_available must validate key format.
        """
        # Valid key
        monkeypatch.setenv("GROQ_API_KEY", "gsk_1234567890abcdefghij")
        with patch("ai.online_ai.GROQ_AVAILABLE", True):
            assert is_groq_available() is True
        
        # Invalid prefix
        monkeypatch.setenv("GROQ_API_KEY", "sk_1234567890abcdefghij")
        with patch("ai.online_ai.GROQ_AVAILABLE", True):
            assert is_groq_available() is False
        
        # Too short
        monkeypatch.setenv("GROQ_API_KEY", "gsk_short")
        with patch("ai.online_ai.GROQ_AVAILABLE", True):
            assert is_groq_available() is False

    def test_is_openai_available_key_validation(self, monkeypatch):
        """
        FIXED: is_openai_available must validate key format.
        """
        # Valid key
        monkeypatch.setenv("OPENAI_API_KEY", "sk-1234567890abcdefghij")
        with patch("ai.online_ai.OPENAI_AVAILABLE", True):
            assert is_openai_available() is True
        
        # Invalid prefix
        monkeypatch.setenv("OPENAI_API_KEY", "gsk-1234567890abcdefghij")
        with patch("ai.online_ai.OPENAI_AVAILABLE", True):
            assert is_openai_available() is False

    @patch("ai.online_ai._get_groq_client")
    def test_call_groq_retry_logic(self, mock_client, mock_groq_completion):
        """
        FIXED: _call_groq must have retry decorator.
        """
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            Exception("503 Service Unavailable"),
            mock_groq_completion,
        ]
        mock_client.return_value = client
        
        result = _call_groq("sys", "user")
        
        assert client.chat.completions.create.call_count == 2
        assert result == mock_groq_completion

    @patch("ai.online_ai._get_openai_client")
    def test_call_openai_retry_logic(self, mock_client, mock_openai_completion):
        """
        FIXED: _call_openai must have retry decorator.
        """
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            Exception("Rate limit"),
            mock_openai_completion,
        ]
        mock_client.return_value = client
        
        result = _call_openai("sys", "user")
        
        assert client.chat.completions.create.call_count == 2
        assert result == mock_openai_completion

    @patch("ai.online_ai._call_groq")
    def test_analyze_patient_groq_success(self, mock_groq, sample_patient, sample_risk_result, mock_groq_completion):
        """Test analyze_patient uses Groq when available."""
        mock_groq.return_value = mock_groq_completion
        
        with patch("ai.online_ai.is_groq_available", return_value=True):
            resp = analyze_online(sample_patient, risk_result=sample_risk_result)
        
        assert resp.success is True
        assert resp.source == "online_groq"
        assert "Groq analysis" in resp.content

    @patch("ai.online_ai._call_groq")
    @patch("ai.online_ai._call_openai")
    def test_analyze_patient_fallback_to_openai(
        self, mock_openai, mock_groq, sample_patient, sample_risk_result, mock_openai_completion
    ):
        """Test analyze_patient falls back to OpenAI if Groq fails."""
        mock_groq.side_effect = Exception("Groq error")
        mock_openai.return_value = mock_openai_completion
        
        with patch("ai.online_ai.is_groq_available", return_value=True), \
             patch("ai.online_ai.is_openai_available", return_value=True):
            resp = analyze_online(sample_patient, risk_result=sample_risk_result)
        
        assert resp.success is True
        assert resp.source == "online_openai"
        mock_groq.assert_called_once()
        mock_openai.assert_called_once()

    @patch("ai.online_ai._call_groq")
    def test_stream_analysis_safe_errors(self, mock_groq, sample_patient):
        """
        FIXED: stream_analysis must use safe_error_msg.
        """
        mock_groq.side_effect = Exception("Connection timeout")
        
        with patch("ai.online_ai.is_groq_available", return_value=True):
            chunks = list(stream_online(sample_patient))
        
        assert len(chunks) > 0
        assert "timed out" in chunks[0].lower()
        assert "timeout" not in chunks[0].lower() or "connection" not in chunks[0].lower()

    @patch("ai.online_ai._call_groq")
    def test_test_api_connection_success(self, mock_groq, mock_groq_completion):
        """Test test_api_connection returns success."""
        mock_groq.return_value = mock_groq_completion
        
        with patch("ai.online_ai.is_groq_available", return_value=True):
            result = test_api_connection("groq")
        
        assert result["success"] is True
        assert result["provider"] == "groq"
        assert result["latency_ms"] >= 0

    def test_get_model_info(self):
        """Test get_model_info returns structure."""
        info = get_online_info()
        
        assert "primary_provider" in info
        assert "groq_available" in info
        assert "openai_available" in info


# ─────────────────────────────────────────────────────────────────────────────
# Integration Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegration:
    def test_prompt_to_response_parsing(self, sample_patient, sample_risk_result):
        """Test full flow: prompt → mock response → parsed sections."""
        # Build prompt
        pkg = build_diagnosis_prompt(sample_patient, sample_risk_result)
        
        # Mock response content
        content = """## 🔍 Clinical Assessment
Patient requires attention.

## ⚠️ Key Risk Indicators
• Elevated cholesterol.

## 💊 Treatment Recommendations
• Start statin therapy.
"""
        
        # Parse sections
        sections = parse_sections(content)
        
        assert "Clinical Assessment" in sections
        assert "Key Risk Indicators" in sections
        assert "Treatment Recommendations" in sections
        assert "Patient requires attention" in sections["Clinical Assessment"]
        assert "Elevated cholesterol" in sections["Key Risk Indicators"]
        assert "statin" in sections["Treatment Recommendations"]

    def test_fallback_response_parses_correctly(self):
        """Test fallback response content is parseable."""
        resp = build_fallback_response(error="Test", risk_level="MEDIUM")
        
        assert resp.success is False
        assert len(resp.sections) > 0
        assert "Clinical Assessment" in resp.sections
        assert "Treatment Recommendations" in resp.sections

    def test_risk_level_consistency_across_layers(self, sample_patient, sample_risk_result):
        """
        Test risk level handling is consistent from risk_result → prompt → response.
        """
        # Risk result has KEY
        assert sample_risk_result["risk_level"] == "HIGH"
        
        # Prompt builder normalizes
        pkg = build_diagnosis_prompt(sample_patient, sample_risk_result)
        assert pkg.metadata["risk_level"] == "HIGH"
        assert "URGENT" in pkg.system_prompt
        
        # Fallback handles both
        resp_key = build_fallback_response("Err", risk_level="HIGH")
        resp_label = build_fallback_response("Err", risk_level="High Risk")
        
        # Both should produce same risk info
        assert "High Risk" in resp_key.content
        assert "High Risk" in resp_label.content
        assert "🚨" in resp_key.content
        assert "🚨" in resp_label.content