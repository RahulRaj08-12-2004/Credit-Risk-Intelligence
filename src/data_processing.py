"""
Data Processing Module for Credit Risk Intelligence platform.
Handles data loading, leakage-free stratified splitting, feature transformation,
and ColumnTransformer preprocessor construction.
"""

import os
from typing import Tuple, List, Dict, Any, Optional
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.features import (
    TARGET_COLUMN,
    BASE_NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    ALL_NUMERICAL_FEATURES,
    ALL_FEATURE_COLUMNS,
    add_engineered_features,
)


def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Loads raw loan dataset from disk (CSV or Parquet).
    Raises FileNotFoundError if filepath does not exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Raw data file not found at: {filepath}")
    
    if filepath.endswith(".parquet"):
        return pd.read_parquet(filepath)
    else:
        return pd.read_csv(filepath)


def clean_and_prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Cleans raw dataframe, verifies target column, excludes post-origination columns,
    and applies domain feature engineering.
    Returns:
        X: Feature dataframe with engineered features
        y: Target series (0 = non-default, 1 = default)
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset columns: {list(df.columns)}")

    # Ensure binary integer target
    y = df[TARGET_COLUMN].astype(int)

    # Keep only defined origination features
    available_cols = [c for c in ALL_FEATURE_COLUMNS if c in df.columns]
    X = df[available_cols].copy()

    # Apply domain feature engineering
    X = add_engineered_features(X)

    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Performs stratified train/test split to preserve class imbalance ratio.
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )


def build_preprocessor(
    numerical_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None
) -> ColumnTransformer:
    """
    Constructs a scikit-learn ColumnTransformer with robust imputers and encoders.
    - Numerical: Median imputation + StandardScaler
    - Categorical: Most frequent imputation + OneHotEncoder with handle_unknown='ignore'
    """
    num_cols = numerical_features or ALL_NUMERICAL_FEATURES
    cat_cols = categorical_features or CATEGORICAL_FEATURES

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def get_feature_names_from_preprocessor(
    preprocessor: ColumnTransformer,
    numerical_features: List[str],
    categorical_features: List[str]
) -> List[str]:
    """
    Extracts human-readable transformed feature names after ColumnTransformer fitting.
    """
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        # Fallback manual reconstruction if get_feature_names_out is unavailable
        names = list(numerical_features)
        cat_pipe = preprocessor.named_transformers_["cat"]
        onehot = cat_pipe.named_steps["onehot"]
        cat_names = onehot.get_feature_names_out(categorical_features)
        names.extend(list(cat_names))
        return names


def save_processed_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: str
) -> None:
    """
    Saves the train and test splits to the specified output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    X_train.to_parquet(os.path.join(output_dir, "X_train.parquet"), index=False)
    X_test.to_parquet(os.path.join(output_dir, "X_test.parquet"), index=False)
    y_train.to_frame("default").to_parquet(os.path.join(output_dir, "y_train.parquet"), index=False)
    y_test.to_frame("default").to_parquet(os.path.join(output_dir, "y_test.parquet"), index=False)
