"""
Tests for SHAP Explainability and Local Attribution.
"""

import os
import pytest
from src.features import get_default_applicant_dict
from src.explain import ModelExplainer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


def test_model_explainer_local_attribution():
    """Verify local SHAP attribution produces top factors and plain summary."""
    explainer = ModelExplainer.get_instance(MODELS_DIR)
    applicant = get_default_applicant_dict()
    explanation = explainer.explain_instance(applicant, top_n=8)

    assert "base_value" in explanation
    assert "top_factors" in explanation
    assert len(explanation["top_factors"]) == 8
    assert "risk_drivers" in explanation
    assert "protective_drivers" in explanation
    assert "natural_language_summary" in explanation
    assert isinstance(explanation["natural_language_summary"], str)
    assert len(explanation["natural_language_summary"]) > 10

    # Ensure each factor has valid direction and numeric value
    for factor in explanation["top_factors"]:
        assert "feature" in factor
        assert "shap_value" in factor
        assert factor["direction"] in ("Risk-Increasing", "Risk-Reducing")
        assert not factor["shap_value"] is None
