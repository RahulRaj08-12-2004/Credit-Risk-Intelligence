"""
Tests for Model Artifact Loading and Metadata Validation.
"""

import os
import json
import pytest
import joblib

from src.predict import ModelManager

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


def test_model_artifacts_exist():
    """Verify all required serialized artifacts are present on disk."""
    required_files = [
        "best_random_forest.joblib",
        "baseline_logistic_regression.joblib",
        "baseline_decision_tree.joblib",
        "preprocessor.joblib",
        "feature_names.json",
        "model_metadata.json",
        "background_sample.parquet",
    ]
    for filename in required_files:
        filepath = os.path.join(MODELS_DIR, filename)
        assert os.path.exists(filepath), f"Missing artifact: {filename}"


def test_model_manager_singleton_loads():
    """Verify ModelManager initializes and loads estimator and preprocessor."""
    manager = ModelManager(MODELS_DIR)
    assert manager.model is not None
    assert manager.preprocessor is not None
    assert isinstance(manager.feature_names, list)
    assert len(manager.feature_names) > 0
    assert isinstance(manager.metadata, dict)
    assert manager.metadata["model_name"] == "RandomForestClassifier"


def test_model_metadata_contents():
    """Verify essential evaluation fields in metadata."""
    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)

    assert "test_roc_auc" in meta
    assert "test_pr_auc" in meta
    assert "optimal_cost_threshold" in meta
    assert meta["test_samples"] == 10000
    assert meta["train_samples"] == 40000
