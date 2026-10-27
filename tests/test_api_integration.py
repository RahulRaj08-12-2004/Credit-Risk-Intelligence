"""
End-to-End Integration Tests: Data -> Preprocessor -> Model -> SHAP.
"""

import os
import json
import pytest
import pandas as pd
import numpy as np

from src.features import get_default_applicant_dict, add_engineered_features
from src.predict import predict_applicant_risk, ModelManager
from src.explain import ModelExplainer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")


def test_end_to_end_pipeline_consistency():
    """Verify that predictions from raw dictionary match transformed batch predictions."""
    applicant = get_default_applicant_dict()
    manager = ModelManager.get_instance(MODELS_DIR)

    # 1. Prediction via predict_applicant_risk
    res = predict_applicant_risk(applicant, model_dir=MODELS_DIR, threshold=0.50)

    # 2. Prediction manually via preprocessor + model
    df_raw = pd.DataFrame([applicant])
    df_feat = add_engineered_features(df_raw)
    X_proc = manager.preprocessor.transform(df_feat)
    raw_probs = manager.model.predict_proba(X_proc)[0]

    assert pytest.approx(res["default_probability"], 0.0001) == float(raw_probs[1])
    assert pytest.approx(res["repayment_probability"], 0.0001) == float(raw_probs[0])


def test_reports_and_metrics_consistency():
    """Verify evaluation reports exist and contain consistent metrics."""
    metrics_path = os.path.join(REPORTS_DIR, "evaluation_metrics.json")
    threshold_path = os.path.join(REPORTS_DIR, "threshold_analysis.json")
    eda_path = os.path.join(REPORTS_DIR, "eda_summary.json")

    assert os.path.exists(metrics_path)
    assert os.path.exists(threshold_path)
    assert os.path.exists(eda_path)

    with open(metrics_path) as f:
        metrics = json.load(f)
    assert "Random Forest" in metrics
    assert "Logistic Regression" in metrics
    assert "Decision Tree" in metrics
    assert metrics["Random Forest"]["roc_auc"] > 0.60
