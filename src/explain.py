"""
Explainability Module for Credit Risk Intelligence Platform.
Integrates SHAP (SHapley Additive exPlanations) for global feature importance
and real-time local applicant prediction attribution.
"""

import os
import json
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib

from src.features import (
    validate_input_record,
    add_engineered_features,
)
from src.predict import ModelManager


class ModelExplainer:
    """
    Cached SHAP TreeExplainer for the trained Random Forest model.
    """
    _instance: Optional["ModelExplainer"] = None

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.manager = ModelManager.get_instance(model_dir)
        self.explainer = None
        self.feature_names = self.manager.feature_names
        self._init_explainer()

    @classmethod
    def get_instance(cls, model_dir: str) -> "ModelExplainer":
        if cls._instance is None:
            cls._instance = cls(model_dir)
        return cls._instance

    def _init_explainer(self) -> None:
        """Initializes shap.TreeExplainer."""
        bg_path = os.path.join(self.model_dir, "background_sample.parquet")
        if os.path.exists(bg_path):
            bg_df = pd.read_parquet(bg_path)
            bg_df = add_engineered_features(bg_df)
            bg_proc = self.manager.preprocessor.transform(bg_df)
            self.explainer = shap.TreeExplainer(self.manager.model, data=bg_proc[:50])
        else:
            self.explainer = shap.TreeExplainer(self.manager.model)

    def explain_instance(self, applicant_dict: Dict[str, Any], top_n: int = 10) -> Dict[str, Any]:
        """
        Computes local SHAP values for an applicant and produces a structured breakdown
        of risk-increasing versus risk-reducing factors.
        """
        sanitized = validate_input_record(applicant_dict)
        df_raw = pd.DataFrame([sanitized])
        df_feat = add_engineered_features(df_raw)
        X_proc = self.manager.preprocessor.transform(df_feat)

        shap_values = self.explainer.shap_values(X_proc, check_additivity=False)

        # Handle binary classification shapes in shap
        # SHAP TreeExplainer for binary RF can return a list of 2 arrays [class_0, class_1]
        # or a 3D array (n_samples, n_features, n_classes) or 2D array
        if isinstance(shap_values, list):
            sv = shap_values[1][0]  # SHAP for class 1 (default)
        elif len(shap_values.shape) == 3:
            sv = shap_values[0, :, 1]
        else:
            sv = shap_values[0]

        # Expected base value
        ev = self.explainer.expected_value
        if isinstance(ev, (list, np.ndarray)):
            base_val = float(ev[1]) if len(ev) > 1 else float(ev[0])
        else:
            base_val = float(ev)

        # Map to feature names
        feature_contributions = []
        for i, val in enumerate(sv):
            fname = self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}"
            fval = float(X_proc[0, i])
            feature_contributions.append({
                "feature": fname,
                "shap_value": float(val),
                "feature_value": round(fval, 3),
                "direction": "Risk-Increasing" if val > 0 else "Risk-Reducing",
            })

        # Sort by absolute impact
        feature_contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        top_factors = feature_contributions[:top_n]
        risk_drivers = [f for f in feature_contributions if f["shap_value"] > 0][:5]
        protective_drivers = [f for f in feature_contributions if f["shap_value"] < 0][:5]

        # Plain language summary
        natural_summary = self._generate_plain_language_summary(risk_drivers, protective_drivers)

        return {
            "base_value": round(base_val, 4),
            "top_factors": top_factors,
            "risk_drivers": risk_drivers,
            "protective_drivers": protective_drivers,
            "natural_language_summary": natural_summary,
        }

    def _generate_plain_language_summary(
        self,
        risk_drivers: List[Dict[str, Any]],
        protective_drivers: List[Dict[str, Any]]
    ) -> str:
        """Translates technical SHAP values into an intuitive underwriter explanation."""
        risk_names = [d["feature"].replace("num__", "").replace("cat__", "") for d in risk_drivers[:3]]
        prot_names = [d["feature"].replace("num__", "").replace("cat__", "") for d in protective_drivers[:3]]

        summary_parts = []
        if risk_names:
            summary_parts.append(f"Elevated default risk is driven primarily by: {', '.join(risk_names)}.")
        if prot_names:
            summary_parts.append(f"Favorable credit factors offsetting risk include: {', '.join(prot_names)}.")
        if not summary_parts:
            summary_parts.append("Applicant profile exhibits balanced credit characteristics close to portfolio baseline.")

        return " ".join(summary_parts)


def generate_global_shap_artifacts(
    model: Any,
    preprocessor: Any,
    feature_names: List[str],
    X_sample: pd.DataFrame,
    output_dir: str
) -> None:
    """
    Computes global SHAP values over a sample and creates summary and bar importance plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.rcParams.update({"font.sans-serif": "DejaVu Sans", "figure.autolayout": True})

    X_feat = add_engineered_features(X_sample)
    X_proc = preprocessor.transform(X_feat)

    # Use 300 instances for reliable global importance computation
    sample_size = min(300, X_proc.shape[0])
    X_eval = X_proc[:sample_size]

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_eval, check_additivity=False)

    if isinstance(shap_vals, list):
        sv = shap_vals[1]
    elif len(shap_vals.shape) == 3:
        sv = shap_vals[:, :, 1]
    else:
        sv = shap_vals

    # 1. SHAP Summary Plot (beeswarm)
    plt.figure(figsize=(10, 7), dpi=200)
    shap.summary_plot(
        sv,
        X_eval,
        feature_names=feature_names,
        max_display=15,
        show=False,
    )
    plt.title("SHAP Beeswarm: Global Feature Impact on Default Probability", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_summary.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # 2. SHAP Bar Feature Importance Plot
    mean_abs_shap = np.mean(np.abs(sv), axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:15]
    top_features = [feature_names[i] for i in top_indices]
    top_values = mean_abs_shap[top_indices]

    plt.figure(figsize=(9, 6), dpi=200)
    y_pos = np.arange(len(top_features))
    plt.barh(y_pos, top_values[::-1], color="#1E3A8A", align="center", alpha=0.85)
    plt.yticks(y_pos, top_features[::-1], fontsize=10)
    plt.xlabel("Mean |SHAP Value| (Average Impact on Model Output Magnitude)", fontsize=11, fontweight="bold")
    plt.title("SHAP Global Feature Importance Ranking (Top 15)", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_bar_importance.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # Save ranking to JSON
    ranking = [
        {"rank": i + 1, "feature": feature_names[idx], "mean_abs_shap": round(float(mean_abs_shap[idx]), 5)}
        for i, idx in enumerate(np.argsort(mean_abs_shap)[::-1])
    ]
    with open(os.path.join(output_dir, "..", "reports", "shap_feature_importance.json"), "w") as f:
        json.dump(ranking, f, indent=2)
