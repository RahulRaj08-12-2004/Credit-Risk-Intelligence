"""
EDA Generation Script for Credit Risk Intelligence platform.
Calculates statistical summaries and outputs publication-grade EDA visualizations.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.features import add_engineered_features


def generate_eda_artifacts(raw_csv_path: str, output_img_dir: str, output_report_dir: str):
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_report_dir, exist_ok=True)

    df = pd.read_csv(raw_csv_path)
    df = add_engineered_features(df)

    # 1. Target Distribution
    plt.figure(figsize=(7, 5), dpi=200)
    counts = df["default"].value_counts()
    percentages = df["default"].value_counts(normalize=True) * 100
    labels = ["Fully Paid (0)", "Default (1)"]
    palette = ["#10B981", "#EF4444"]
    bars = plt.bar(labels, counts.values, color=palette, width=0.55, edgecolor="#1E293B", linewidth=1.2)
    plt.title("Loan Outcome Class Distribution (LendingClub)", fontsize=13, fontweight="bold", pad=12)
    plt.ylabel("Number of Loans", fontsize=11, fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for i, bar in enumerate(bars):
        height = bar.get_height()
        pct = percentages[counts.index[i]]
        plt.annotate(
            f"{height:,}\n({pct:.1f}%)",
            (bar.get_x() + bar.get_width() / 2, height / 2),
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            fontsize=11,
        )

    plt.tight_layout()
    plt.savefig(os.path.join(output_img_dir, "eda_target_distribution.png"), dpi=200)
    plt.close()

    # 2. FICO Score vs Default Rate (binned)
    df["fico_bin"] = pd.cut(df["fico_score"], bins=np.arange(640, 860, 20))
    fico_grp = df.groupby("fico_bin", observed=False)["default"].agg(["count", "mean"]).reset_index()
    fico_grp["default_rate_pct"] = fico_grp["mean"] * 100

    plt.figure(figsize=(9, 5), dpi=200)
    x_labels = [f"{int(b.left)}-{int(b.right)}" for b in fico_grp["fico_bin"]]
    plt.plot(x_labels, fico_grp["default_rate_pct"], marker="o", color="#3B82F6", linewidth=2.5, markersize=7)
    plt.fill_between(x_labels, fico_grp["default_rate_pct"], color="#93C5FD", alpha=0.35)
    plt.title("Observed Default Rate (%) by FICO Credit Score Tier", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("FICO Score Bracket", fontsize=11, fontweight="bold")
    plt.ylabel("Default Rate (%)", fontsize=11, fontweight="bold")
    plt.xticks(rotation=30)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_dir, "eda_fico_vs_default.png"), dpi=200)
    plt.close()

    # 3. DTI Distribution by Loan Status (Violin / KDE)
    plt.figure(figsize=(8, 5), dpi=200)
    sns.kdeplot(
        data=df[df["dti"] <= 45],
        x="dti",
        hue="default",
        common_norm=False,
        palette=["#10B981", "#EF4444"],
        fill=True,
        alpha=0.4,
        linewidth=2,
    )
    plt.title("Debt-to-Income (DTI) Density Distribution by Loan Outcome", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Debt-to-Income Ratio (%)", fontsize=11, fontweight="bold")
    plt.ylabel("Density", fontsize=11, fontweight="bold")
    plt.legend(title="Loan Status", labels=["Default (1)", "Fully Paid (0)"])
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_dir, "eda_dti_distribution.png"), dpi=200)
    plt.close()

    # 4. Correlation Heatmap across Top Features
    top_corr_cols = [
        "default", "fico_score", "int_rate", "dti", "installment",
        "loan_amnt", "annual_inc", "revol_util", "inq_last_6mths",
        "open_acc", "delinq_2yrs", "pub_rec", "mort_acc"
    ]
    corr = df[top_corr_cols].corr()

    plt.figure(figsize=(10, 8), dpi=200)
    sns.heatmap(
        corr,
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 8.5},
    )
    plt.title("Correlation Heatmap: Loan Characteristics and Default Status", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_dir, "eda_correlation_matrix.png"), dpi=200)
    plt.close()

    # 5. Architecture diagram
    _generate_architecture_diagram(output_img_dir)

    # 6. Save EDA Summary Report
    eda_summary = {
        "dataset_rows": len(df),
        "dataset_columns": len(df.columns),
        "target_column": "default",
        "total_defaults": int(counts[1]),
        "total_non_defaults": int(counts[0]),
        "default_rate": float(counts[1] / len(df)),
        "class_imbalance_ratio": f"{counts[0] / counts[1]:.2f}:1",
        "mean_loan_amount": float(df["loan_amnt"].mean()),
        "median_loan_amount": float(df["loan_amnt"].median()),
        "mean_interest_rate": float(df["int_rate"].mean()),
        "mean_annual_income": float(df["annual_inc"].mean()),
        "median_annual_income": float(df["annual_inc"].median()),
        "mean_fico_score": float(df["fico_score"].mean()),
        "mean_dti": float(df["dti"].mean()),
        "numerical_features_count": len(df.select_dtypes(include=[np.number]).columns) - 1,
        "categorical_features_count": len(df.select_dtypes(include=["object"]).columns),
        "missing_values_total": int(df.isnull().sum().sum()),
    }

    with open(os.path.join(output_report_dir, "eda_summary.json"), "w") as f:
        json.dump(eda_summary, f, indent=2)

    print("EDA figures and summary report successfully generated.")


def _generate_architecture_diagram(output_img_dir: str):
    """Generates visual architecture diagram."""
    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
    ax.axis("off")

    boxes = [
        {"text": "Historical\nLendingClub Data\n(50k Loans, 25 Features)", "x": 0.08, "y": 0.65, "color": "#E0E7FF"},
        {"text": "Leakage-Free\nPreprocessing\n(Stratified Split,\nColumnTransformer)", "x": 0.32, "y": 0.65, "color": "#DBEAFE"},
        {"text": "Random Forest\nTuned Model\n(150 Trees,\nClass-Weighted)", "x": 0.58, "y": 0.65, "color": "#CFFAFE"},
        {"text": "Explainable AI\n(SHAP TreeExplainer\nGlobal & Local)", "x": 0.84, "y": 0.65, "color": "#D1FAE5"},
        {"text": "Baseline Models\n- Logistic Regression\n- Decision Tree", "x": 0.58, "y": 0.25, "color": "#FEE2E2"},
        {"text": "Streamlit Fintech\nDashboard &\nPrediction Workspace", "x": 0.84, "y": 0.25, "color": "#FEF3C7"},
    ]

    for b in boxes:
        ax.text(
            b["x"], b["y"], b["text"],
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.8", facecolor=b["color"], edgecolor="#1E293B", linewidth=1.5),
            fontsize=10, fontweight="bold"
        )

    # Arrows
    arrow_props = dict(arrowstyle="->", lw=2, color="#1E293B")
    ax.annotate("", xy=(0.20, 0.65), xytext=(0.17, 0.65), arrowprops=arrow_props)
    ax.annotate("", xy=(0.46, 0.65), xytext=(0.43, 0.65), arrowprops=arrow_props)
    ax.annotate("", xy=(0.72, 0.65), xytext=(0.69, 0.65), arrowprops=arrow_props)
    ax.annotate("", xy=(0.46, 0.35), xytext=(0.35, 0.55), arrowprops=arrow_props)
    ax.annotate("", xy=(0.72, 0.25), xytext=(0.69, 0.25), arrowprops=arrow_props)
    ax.annotate("", xy=(0.84, 0.40), xytext=(0.84, 0.50), arrowprops=arrow_props)

    plt.title("Credit Risk Intelligence Platform Architecture", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_dir, "architecture_diagram.png"), dpi=200)
    plt.close()


if __name__ == "__main__":
    base = "C:/Users/sanka/.gemini/antigravity/scratch/Credit-Risk-Intelligence"
    generate_eda_artifacts(
        raw_csv_path=os.path.join(base, "data/raw/lending_club_loans.csv"),
        output_img_dir=os.path.join(base, "images"),
        output_report_dir=os.path.join(base, "reports"),
    )
