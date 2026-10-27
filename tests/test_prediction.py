"""
Tests for Real-Time Prediction, Thresholding, and Risk Tiers.
"""

import os
import pytest
from src.features import get_default_applicant_dict
from src.predict import predict_applicant_risk, get_risk_tier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


def test_predict_applicant_risk_output_structure():
    """Verify prediction result format, probability ranges, and types."""
    applicant = get_default_applicant_dict()
    result = predict_applicant_risk(applicant, model_dir=MODELS_DIR, threshold=0.50)

    assert "default_probability" in result
    assert "repayment_probability" in result
    assert "predicted_class" in result
    assert "decision" in result
    assert "risk_tier" in result
    assert "risk_factors" in result
    assert "protective_factors" in result
    assert "engineered_features" in result

    # Verify probability constraints
    assert 0.0 <= result["default_probability"] <= 1.0
    assert 0.0 <= result["repayment_probability"] <= 1.0
    assert pytest.approx(result["default_probability"] + result["repayment_probability"], 0.001) == 1.0
    assert result["predicted_class"] in (0, 1)


def test_predict_threshold_effect():
    """Verify that lowering threshold flags higher-risk loans as default."""
    applicant = get_default_applicant_dict()
    # High threshold should approve
    res_high = predict_applicant_risk(applicant, model_dir=MODELS_DIR, threshold=0.95)
    assert res_high["predicted_class"] == 0
    assert res_high["decision"] == "Approve Loan"

    # Low threshold should decline
    res_low = predict_applicant_risk(applicant, model_dir=MODELS_DIR, threshold=0.05)
    assert res_low["predicted_class"] == 1
    assert res_low["decision"] == "Decline Loan"


def test_risk_tier_mapping():
    """Verify risk tier assignment logic."""
    label1, color1, _ = get_risk_tier(0.08)
    assert "Low Risk" in label1

    label2, color2, _ = get_risk_tier(0.22)
    assert "Moderate Risk" in label2

    label3, color3, _ = get_risk_tier(0.42)
    assert "Elevated Risk" in label3

    label4, color4, _ = get_risk_tier(0.68)
    assert "High / Critical Risk" in label4
