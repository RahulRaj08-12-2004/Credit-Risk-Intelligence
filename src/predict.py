"""
Prediction Module for Credit Risk Intelligence Platform.
Provides real-time scoring, caching, input validation, and risk tier categorization.
"""

import os
import json
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import joblib

from src.features import (
    validate_input_record,
    add_engineered_features,
    ALL_FEATURE_COLUMNS,
    FEATURE_METADATA,
)


class ModelManager:
    """
    Cached loader and manager for machine learning models and preprocessing pipelines.
    """
    _instance: Optional["ModelManager"] = None

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.metadata = None
        self.load_artifacts()

    @classmethod
    def get_instance(cls, model_dir: str) -> "ModelManager":
        if cls._instance is None:
            cls._instance = cls(model_dir)
        return cls._instance

    def load_artifacts(self) -> None:
        """Loads serialized models, preprocessing pipelines, and schemas."""
        model_path = os.path.join(self.model_dir, "best_random_forest.joblib")
        preprocessor_path = os.path.join(self.model_dir, "preprocessor.joblib")
        features_path = os.path.join(self.model_dir, "feature_names.json")
        metadata_path = os.path.join(self.model_dir, "model_metadata.json")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model artifact missing at {model_path}. Train the model first.")
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"Preprocessor artifact missing at {preprocessor_path}.")

        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)

        if os.path.exists(features_path):
            with open(features_path, "r") as f:
                self.feature_names = json.load(f)
        else:
            self.feature_names = []

        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}


def get_risk_tier(prob: float) -> Tuple[str, str, str]:
    """Maps default probability to an institutional risk tier."""
    if prob < 0.15:
        return "Tier 1 - Low Risk", "#10B981", "Favorable risk profile. Low expected default probability."
    elif prob < 0.30:
        return "Tier 2 - Moderate Risk", "#F59E0B", "Acceptable risk profile under standard pricing."
    elif prob < 0.50:
        return "Tier 3 - Elevated Risk", "#F97316", "Heightened credit risk. Recommend risk-based interest surcharge."
    else:
        return "Tier 4 - High / Critical Risk", "#EF4444", "Substantial default hazard. Rejection or collateral recommended."


def predict_applicant_risk(
    applicant_dict: Dict[str, Any],
    model_dir: str,
    threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Evaluates loan applicant data and produces real-time credit default risk predictions.
    """
    manager = ModelManager.get_instance(model_dir)
    sanitized = validate_input_record(applicant_dict)

    # Convert to DataFrame
    df_raw = pd.DataFrame([sanitized])
    
    # Domain feature engineering
    df_feat = add_engineered_features(df_raw)

    # Transform through fitted ColumnTransformer
    X_proc = manager.preprocessor.transform(df_feat)

    # Predict default probabilities
    probs = manager.model.predict_proba(X_proc)[0]
    default_prob = float(probs[1])
    repay_prob = float(probs[0])

    # Determine threshold (from parameter or model metadata default)
    if threshold is None:
        threshold = manager.metadata.get("optimal_cost_threshold", 0.35)

    is_default = int(default_prob >= threshold)
    decision = "Decline Loan" if is_default == 1 else "Approve Loan"
    tier_label, tier_color, tier_desc = get_risk_tier(default_prob)

    # Summarize key credit drivers
    risk_factors = []
    protective_factors = []

    fico = df_feat["fico_score"].iloc[0]
    dti = df_feat["dti"].iloc[0]
    inq = df_feat["inq_last_6mths"].iloc[0]
    util = df_feat["revol_util"].iloc[0]

    if fico < 660:
        risk_factors.append(f"Subprime FICO score ({fico:.0f})")
    elif fico >= 720:
        protective_factors.append(f"Prime FICO score ({fico:.0f})")

    if dti > 25.0:
        risk_factors.append(f"Elevated debt-to-income ratio ({dti:.1f}%)")
    elif dti < 15.0:
        protective_factors.append(f"Conservative debt-to-income ratio ({dti:.1f}%)")

    if inq > 1:
        risk_factors.append(f"Multiple recent credit inquiries ({inq:.0f} in past 6 mos)")
    else:
        protective_factors.append("Low recent credit inquiry velocity")

    if util > 70.0:
        risk_factors.append(f"High credit card utilization ({util:.1f}%)")
    elif util < 30.0:
        protective_factors.append(f"Healthy revolving credit headroom ({util:.1f}% util)")

    return {
        "default_probability": round(default_prob, 4),
        "repayment_probability": round(repay_prob, 4),
        "predicted_class": is_default,
        "decision": decision,
        "risk_tier": tier_label,
        "risk_color": tier_color,
        "risk_description": tier_desc,
        "threshold_used": round(threshold, 3),
        "risk_factors": risk_factors,
        "protective_factors": protective_factors,
        "input_features": sanitized,
        "engineered_features": {
            "fico_score": round(float(df_feat["fico_score"].iloc[0]), 1),
            "installment_to_income": round(float(df_feat["installment_to_income"].iloc[0]), 4),
            "credit_limit_utilization": round(float(df_feat["credit_limit_utilization"].iloc[0]), 4),
        },
    }
