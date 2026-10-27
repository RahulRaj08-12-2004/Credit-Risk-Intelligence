"""
Training Module for Credit Risk Intelligence platform.
Orchestrates baseline training, hyperparameter tuning for Random Forest,
cross-validation, model serialization, and evaluation artifact generation.
"""

import os
import json
import time
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_score

from src.data_processing import (
    load_raw_data,
    clean_and_prepare_data,
    split_data,
    build_preprocessor,
    get_feature_names_from_preprocessor,
    save_processed_data,
)
from src.features import (
    ALL_NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
)
from src.evaluate import (
    compute_metrics,
    compute_threshold_curve,
    generate_evaluation_plots,
)


def run_training_pipeline(
    raw_data_path: str,
    output_model_dir: str,
    output_reports_dir: str,
    output_images_dir: str,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Executes end-to-end model training, tuning, and evaluation workflow.
    """
    start_time = time.time()
    os.makedirs(output_model_dir, exist_ok=True)
    os.makedirs(output_reports_dir, exist_ok=True)
    os.makedirs(output_images_dir, exist_ok=True)

    print("=================================================================")
    print("STEP 1: Loading and Preparing Data")
    print("=================================================================")
    df = load_raw_data(raw_data_path)
    print(f"Loaded raw dataset with {len(df):,} rows and {len(df.columns)} columns.")

    X, y = clean_and_prepare_data(df)
    print(f"Cleaned feature matrix: {X.shape[1]} predictors. Target default rate: {y.mean():.4f}")

    print("\n=================================================================")
    print("STEP 2: Stratified Train-Test Splitting (80% Train, 20% Test)")
    print("=================================================================")
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.20, random_state=random_state)
    print(f"Train set: {len(X_train):,} samples (Defaults: {y_train.sum():,} / {y_train.mean():.2%})")
    print(f"Test set:  {len(X_test):,} samples (Defaults: {y_test.sum():,} / {y_test.mean():.2%})")

    # Save processed splits
    processed_dir = os.path.join(os.path.dirname(raw_data_path), "..", "processed")
    save_processed_data(X_train, X_test, y_train, y_test, processed_dir)

    print("\n=================================================================")
    print("STEP 3: Fitting ColumnTransformer (Leakage-Free on Train Only)")
    print("=================================================================")
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)

    feature_names = get_feature_names_from_preprocessor(
        preprocessor,
        ALL_NUMERICAL_FEATURES,
        CATEGORICAL_FEATURES
    )
    print(f"Total transformed feature dimension: {len(feature_names)} features.")

    X_train_proc = preprocessor.transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # 5-Fold Stratified CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    print("\n=================================================================")
    print("STEP 4: Training Baseline 1 - Logistic Regression")
    print("=================================================================")
    log_reg = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=random_state
    )
    cv_scores_lr = cross_val_score(log_reg, X_train_proc, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    log_reg.fit(X_train_proc, y_train)
    print(f"Logistic Regression 5-Fold Mean ROC-AUC: {cv_scores_lr.mean():.4f} (+/- {cv_scores_lr.std():.4f})")

    print("\n=================================================================")
    print("STEP 5: Training Baseline 2 - Decision Tree Classifier")
    print("=================================================================")
    dec_tree = DecisionTreeClassifier(
        max_depth=6,
        min_samples_leaf=25,
        class_weight="balanced",
        random_state=random_state
    )
    cv_scores_dt = cross_val_score(dec_tree, X_train_proc, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    dec_tree.fit(X_train_proc, y_train)
    print(f"Decision Tree 5-Fold Mean ROC-AUC: {cv_scores_dt.mean():.4f} (+/- {cv_scores_dt.std():.4f})")

    print("\n=================================================================")
    print("STEP 6: Tuning & Training Primary Model - Random Forest")
    print("=================================================================")
    rf_base = RandomForestClassifier(
        random_state=random_state,
        n_jobs=-1
    )

    param_distributions = {
        "n_estimators": [100, 150, 200],
        "max_depth": [12, 16, 20],
        "min_samples_split": [10, 20],
        "min_samples_leaf": [4, 8],
        "max_features": ["sqrt", 0.3],
        "class_weight": ["balanced", "balanced_subsample"],
    }

    random_search = RandomizedSearchCV(
        estimator=rf_base,
        param_distributions=param_distributions,
        n_iter=8,
        cv=cv,
        scoring="roc_auc",
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )
    random_search.fit(X_train_proc, y_train)

    best_rf = random_search.best_estimator_
    best_params = random_search.best_params_
    best_cv_score = float(random_search.best_score_)
    print(f"Best Random Forest 5-Fold ROC-AUC: {best_cv_score:.4f}")
    print(f"Optimized Hyperparameters: {best_params}")

    print("\n=================================================================")
    print("STEP 7: Evaluating Models on Untouched Holdout Test Set")
    print("=================================================================")
    models = {
        "Random Forest": best_rf,
        "Logistic Regression": log_reg,
        "Decision Tree": dec_tree,
    }
    X_test_dict = {
        "Random Forest": X_test_proc,
        "Logistic Regression": X_test_proc,
        "Decision Tree": X_test_proc,
    }

    metrics_summary: Dict[str, Any] = {}
    for name, model in models.items():
        y_prob = model.predict_proba(X_test_proc)[:, 1]
        m = compute_metrics(y_test.values, y_prob, threshold=0.50)
        metrics_summary[name] = m
        print(f"[{name}] Test ROC-AUC: {m['roc_auc']:.4f} | PR-AUC: {m['pr_auc']:.4f} | F1: {m['f1_score']:.4f} | Recall: {m['recall']:.4f} | Prec: {m['precision']:.4f}")

    print("\n=================================================================")
    print("STEP 8: Threshold Optimization & Financial Impact Analysis")
    print("=================================================================")
    rf_test_prob = best_rf.predict_proba(X_test_proc)[:, 1]
    threshold_curve = compute_threshold_curve(y_test.values, rf_test_prob, cost_fn=5000.0, cost_fp=1200.0)

    # Find minimum business loss threshold
    best_cost_entry = min(threshold_curve, key=lambda x: x["total_business_loss"])
    best_f1_entry = max(threshold_curve, key=lambda x: x["f1_score"])
    print(f"Default (0.50) Expected Portfolio Loss: ${metrics_summary['Random Forest']['confusion_matrix']['fn']*5000 + metrics_summary['Random Forest']['confusion_matrix']['fp']*1200:,.2f}")
    print(f"Optimal Loss Threshold ({best_cost_entry['threshold']:.2f}): ${best_cost_entry['total_business_loss']:,.2f} (Saves ${best_cost_entry['net_capital_saved']:,.2f})")
    print(f"Optimal F1 Threshold ({best_f1_entry['threshold']:.2f}): F1 = {best_f1_entry['f1_score']:.4f}")

    print("\n=================================================================")
    print("STEP 9: Generating Publication-Ready Figures")
    print("=================================================================")
    generate_evaluation_plots(models, X_test_dict, y_test.values, output_images_dir)
    print(f"Plots saved to: {output_images_dir}")

    print("\n=================================================================")
    print("STEP 10: Serializing Models, Metadata, and Artifacts")
    print("=================================================================")
    # 1. Models and preprocessor
    joblib.dump(best_rf, os.path.join(output_model_dir, "best_random_forest.joblib"))
    joblib.dump(log_reg, os.path.join(output_model_dir, "baseline_logistic_regression.joblib"))
    joblib.dump(dec_tree, os.path.join(output_model_dir, "baseline_decision_tree.joblib"))
    joblib.dump(preprocessor, os.path.join(output_model_dir, "preprocessor.joblib"))

    # 2. Transformed feature names
    with open(os.path.join(output_model_dir, "feature_names.json"), "w") as f:
        json.dump(feature_names, f, indent=2)

    # 3. Model metadata
    metadata = {
        "model_name": "RandomForestClassifier",
        "version": "1.0.0",
        "trained_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training_time_seconds": round(time.time() - start_time, 2),
        "dataset_rows_total": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "best_hyperparameters": best_params,
        "5_fold_cv_roc_auc": best_cv_score,
        "test_roc_auc": metrics_summary["Random Forest"]["roc_auc"],
        "test_pr_auc": metrics_summary["Random Forest"]["pr_auc"],
        "test_f1": metrics_summary["Random Forest"]["f1_score"],
        "optimal_cost_threshold": best_cost_entry["threshold"],
        "optimal_f1_threshold": best_f1_entry["threshold"],
        "feature_count_raw": len(X.columns),
        "feature_count_transformed": len(feature_names),
    }
    with open(os.path.join(output_model_dir, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # 4. Save background sample for fast SHAP explanations (~500 instances)
    bg_sample = X_train.sample(n=min(500, len(X_train)), random_state=random_state)
    bg_sample.to_parquet(os.path.join(output_model_dir, "background_sample.parquet"), index=False)

    # 5. Reports
    with open(os.path.join(output_reports_dir, "evaluation_metrics.json"), "w") as f:
        json.dump(metrics_summary, f, indent=2)

    with open(os.path.join(output_reports_dir, "threshold_analysis.json"), "w") as f:
        json.dump(threshold_curve, f, indent=2)

    cv_results = {
        "Random Forest (Tuned)": {"mean_cv_roc_auc": best_cv_score},
        "Logistic Regression": {"mean_cv_roc_auc": float(cv_scores_lr.mean()), "std": float(cv_scores_lr.std())},
        "Decision Tree": {"mean_cv_roc_auc": float(cv_scores_dt.mean()), "std": float(cv_scores_dt.std())},
    }
    with open(os.path.join(output_reports_dir, "cv_results.json"), "w") as f:
        json.dump(cv_results, f, indent=2)

    print(f"Training completed in {metadata['training_time_seconds']} seconds.")
    return metadata


if __name__ == "__main__":
    base_dir = "C:/Users/sanka/.gemini/antigravity/scratch/Credit-Risk-Intelligence"
    run_training_pipeline(
        raw_data_path=os.path.join(base_dir, "data/raw/lending_club_loans.csv"),
        output_model_dir=os.path.join(base_dir, "models"),
        output_reports_dir=os.path.join(base_dir, "reports"),
        output_images_dir=os.path.join(base_dir, "images"),
    )
