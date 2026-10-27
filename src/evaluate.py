"""
Model Evaluation and Advanced Analytics Module for Credit Risk Intelligence.
Calculates performance metrics, threshold optimization, confusion matrices,
calibration, and business impact analysis.
"""

import os
import json
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50
) -> Dict[str, Any]:
    """
    Computes comprehensive binary classification metrics for default prediction.
    """
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))

    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "threshold": float(threshold),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "brier_score": round(brier, 4),
        "specificity": round(specificity, 4),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "support": {
            "total": int(len(y_true)),
            "actual_defaults": int(y_true.sum()),
            "actual_non_defaults": int(len(y_true) - y_true.sum()),
            "predicted_defaults": int(y_pred.sum()),
            "predicted_non_defaults": int(len(y_pred) - y_pred.sum()),
        },
    }


def compute_threshold_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_fn: float = 5000.0,
    cost_fp: float = 1200.0,
    steps: int = 50
) -> List[Dict[str, Any]]:
    """
    Sweeps decision threshold from 0.02 to 0.98.
    Calculates precision, recall, F1, and simulated business underwriting financial impact.
    - cost_fn: Average loss when defaulting loan is approved ($5,000)
    - cost_fp: Foregone interest margin when creditworthy borrower is rejected ($1,200)
    """
    thresholds = np.linspace(0.02, 0.98, steps)
    results = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        acc = float(accuracy_score(y_true, y_pred))

        # Total expected business loss = (False Negatives * Default Loss) + (False Positives * Opportunity Cost)
        total_loss = (fn * cost_fn) + (fp * cost_fp)
        # Net portfolio benefit compared to zero screening
        net_saved = (tp * cost_fn) - (fp * cost_fp)

        results.append({
            "threshold": round(float(t), 3),
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "total_business_loss": round(total_loss, 2),
            "net_capital_saved": round(net_saved, 2),
        })

    return results


def generate_evaluation_plots(
    models_dict: Dict[str, Any],
    X_test_dict: Dict[str, Any],
    y_test: np.ndarray,
    output_dir: str
) -> None:
    """
    Generates and saves publication-grade evaluation figures:
    - ROC Curves (multi-model comparison)
    - Precision-Recall Curves (multi-model comparison)
    - Confusion Matrix (Random Forest)
    - Model Comparison Bar Chart
    - Calibration Curves
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams.update({"font.sans-serif": "DejaVu Sans", "figure.autolayout": True})

    # 1. ROC Curves
    plt.figure(figsize=(8, 6), dpi=200)
    colors = {"Random Forest": "#1f77b4", "Logistic Regression": "#ff7f0e", "Decision Tree": "#2ca02c"}
    
    for name, model in models_dict.items():
        X_test = X_test_dict[name]
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", color=colors.get(name, "#333333"), linewidth=2.2)

    plt.plot([0, 1], [0, 1], "k--", label="Random Chance (AUC = 0.500)", alpha=0.6)
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11, fontweight="bold")
    plt.title("ROC Curve: Loan Default Classification", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=200)
    plt.close()

    # 2. Precision-Recall Curves
    plt.figure(figsize=(8, 6), dpi=200)
    baseline_pr = float(y_test.sum() / len(y_test))
    plt.axhline(y=baseline_pr, color="gray", linestyle="--", label=f"No-Skill Baseline ({baseline_pr:.3f})", alpha=0.6)

    for name, model in models_dict.items():
        X_test = X_test_dict[name]
        y_prob = model.predict_proba(X_test)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        plt.plot(rec, prec, label=f"{name} (PR-AUC = {pr_auc:.3f})", color=colors.get(name, "#333333"), linewidth=2.2)

    plt.xlabel("Recall (Sensitivity)", fontsize=11, fontweight="bold")
    plt.ylabel("Precision (Positive Predictive Value)", fontsize=11, fontweight="bold")
    plt.title("Precision-Recall Curve: Imbalanced Loan Default", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pr_curve.png"), dpi=200)
    plt.close()

    # 3. Confusion Matrix for Random Forest (primary model)
    rf_model = models_dict["Random Forest"]
    rf_X_test = X_test_dict["Random Forest"]
    rf_prob = rf_model.predict_proba(rf_X_test)[:, 1]
    rf_pred = (rf_prob >= 0.50).astype(int)
    cm = confusion_matrix(y_test, rf_pred)

    plt.figure(figsize=(6, 5), dpi=200)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Fully Paid (0)", "Default (1)"],
        yticklabels=["Fully Paid (0)", "Default (1)"],
        annot_kws={"size": 13, "weight": "bold"},
    )
    plt.xlabel("Predicted Class", fontsize=11, fontweight="bold")
    plt.ylabel("Actual True Class", fontsize=11, fontweight="bold")
    plt.title("Confusion Matrix: Random Forest (Threshold = 0.50)", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=200)
    plt.close()

    # 4. Calibration Curves
    plt.figure(figsize=(8, 6), dpi=200)
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated", alpha=0.7)
    for name, model in models_dict.items():
        X_test = X_test_dict[name]
        y_prob = model.predict_proba(X_test)[:, 1]
        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
        plt.plot(prob_pred, prob_true, "s-", label=name, color=colors.get(name, "#333333"), linewidth=2)

    plt.xlabel("Mean Predicted Probability", fontsize=11, fontweight="bold")
    plt.ylabel("Fraction of Positives (Actual Default Rate)", fontsize=11, fontweight="bold")
    plt.title("Reliability Diagram (Probability Calibration)", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "calibration_curve.png"), dpi=200)
    plt.close()

    # 5. Model Comparison Bar Chart
    comparison_data = []
    for name, model in models_dict.items():
        X_test = X_test_dict[name]
        y_prob = model.predict_proba(X_test)[:, 1]
        m = compute_metrics(y_test, y_prob, threshold=0.50)
        comparison_data.append({
            "Model": name,
            "ROC-AUC": m["roc_auc"],
            "PR-AUC": m["pr_auc"],
            "F1-Score": m["f1_score"],
            "Recall": m["recall"],
            "Precision": m["precision"],
        })

    comp_df = pd.DataFrame(comparison_data)
    melted = pd.melt(comp_df, id_vars=["Model"], var_name="Metric", value_name="Score")

    plt.figure(figsize=(10, 5), dpi=200)
    ax = sns.barplot(data=melted, x="Metric", y="Score", hue="Model", palette=["#1f77b4", "#ff7f0e", "#2ca02c"])
    plt.ylim(0, 1.0)
    plt.title("Model Performance Benchmark Comparison (Test Set)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Evaluation Metric", fontsize=11, fontweight="bold")
    plt.ylabel("Metric Score", fontsize=11, fontweight="bold")
    plt.legend(frameon=True, loc="upper right")
    for p in ax.patches:
        h = p.get_height()
        if h > 0.05:
            ax.annotate(f"{h:.2f}", (p.get_x() + p.get_width() / 2.0, h),
                        ha="center", va="bottom", fontsize=8, xytext=(0, 2), textcoords="offset points")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison.png"), dpi=200)
    plt.close()
