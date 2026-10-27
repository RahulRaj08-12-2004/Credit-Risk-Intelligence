"""
Tests for Data Processing and Feature Engineering pipelines.
"""

import pytest
import pandas as pd
import numpy as np

from src.features import (
    add_engineered_features,
    validate_input_record,
    get_default_applicant_dict,
    BASE_NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_METADATA,
)
from src.data_processing import (
    build_preprocessor,
    clean_and_prepare_data,
    split_data,
)


def test_engineered_features():
    """Verify calculation of FICO score, installment-to-income, and credit utilization."""
    df = pd.DataFrame([{
        "fico_range_low": 680.0,
        "fico_range_high": 684.0,
        "installment": 500.0,
        "annual_inc": 60000.0,
        "revol_bal": 10000.0,
        "total_rev_hi_lim": 20000.0,
    }])
    df_feat = add_engineered_features(df)
    
    assert df_feat["fico_score"].iloc[0] == 682.0
    expected_iti = (500.0 * 12.0) / (60000.0 + 1.0)
    assert pytest.approx(df_feat["installment_to_income"].iloc[0], 0.0001) == expected_iti
    expected_util = 10000.0 / (20000.0 + 1.0)
    assert pytest.approx(df_feat["credit_limit_utilization"].iloc[0], 0.0001) == expected_util


def test_validate_input_record_valid():
    """Test standard valid record sanitization."""
    record = get_default_applicant_dict()
    sanitized = validate_input_record(record)
    assert isinstance(sanitized, dict)
    assert sanitized["loan_amnt"] == 15000.0
    assert sanitized["term"] == " 36 months"


def test_validate_input_record_clamping():
    """Test boundary clamping for extreme numerical outliers."""
    record = get_default_applicant_dict()
    record["loan_amnt"] = 100000.0  # Above max 40,000
    record["dti"] = -5.0           # Below min 0
    sanitized = validate_input_record(record)
    
    assert sanitized["loan_amnt"] == 40000.0
    assert sanitized["dti"] == 0.0


def test_validate_input_record_unseen_category():
    """Test fallback when category is unknown."""
    record = get_default_applicant_dict()
    record["home_ownership"] = "PENTHOUSE_MANSION"
    sanitized = validate_input_record(record)
    assert sanitized["home_ownership"] in FEATURE_METADATA["home_ownership"]["options"]


def test_preprocessor_handles_missing_and_unseen():
    """Test that ColumnTransformer handles nulls and unseen categories gracefully."""
    X_train = pd.DataFrame([get_default_applicant_dict() for _ in range(20)])
    X_train = add_engineered_features(X_train)
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)

    # Test with unseen category and NaN values
    test_dict = get_default_applicant_dict()
    test_dict["home_ownership"] = "MARS_COLONY"
    test_dict["annual_inc"] = np.nan
    test_df = pd.DataFrame([test_dict])
    test_df = add_engineered_features(test_df)

    transformed = preprocessor.transform(test_df)
    assert transformed.shape[0] == 1
    assert not np.isnan(transformed).any()
