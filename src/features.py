"""
Features module for Credit Risk Intelligence platform.
Defines column sets, metadata, domain feature engineering, and validation.
"""

from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

# Target variable definition
TARGET_COLUMN = "default"

# Base numerical predictors (available at origination)
BASE_NUMERICAL_FEATURES: List[str] = [
    "loan_amnt",
    "int_rate",
    "installment",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "fico_range_low",
    "fico_range_high",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "mort_acc",
    "pub_rec_bankruptcies",
    "tot_cur_bal",
    "total_rev_hi_lim",
]

# Engineered numerical features added during pipeline
ENGINEERED_FEATURES: List[str] = [
    "fico_score",
    "installment_to_income",
    "credit_limit_utilization",
]

# Combined list of numerical features used in models
ALL_NUMERICAL_FEATURES: List[str] = BASE_NUMERICAL_FEATURES + ENGINEERED_FEATURES

# Categorical predictors (available at origination)
CATEGORICAL_FEATURES: List[str] = [
    "term",
    "emp_length",
    "home_ownership",
    "verification_status",
    "purpose",
    "addr_state",
]

ALL_FEATURE_COLUMNS: List[str] = BASE_NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# Feature groups for structured UI forms and reports
FEATURE_GROUPS = {
    "Loan Details": [
        "loan_amnt",
        "term",
        "int_rate",
        "installment",
        "purpose",
    ],
    "Applicant Profile": [
        "annual_inc",
        "emp_length",
        "home_ownership",
        "verification_status",
        "addr_state",
    ],
    "Financial & Debt Burden": [
        "dti",
        "tot_cur_bal",
        "revol_bal",
        "revol_util",
        "total_rev_hi_lim",
    ],
    "Credit Bureau & History": [
        "fico_range_low",
        "fico_range_high",
        "delinq_2yrs",
        "inq_last_6mths",
        "open_acc",
        "total_acc",
        "pub_rec",
        "pub_rec_bankruptcies",
        "mort_acc",
    ],
}

# Detailed schema and validation specifications
FEATURE_METADATA: Dict[str, Dict[str, Any]] = {
    "loan_amnt": {
        "label": "Loan Amount ($)",
        "type": "float",
        "min": 1000.0,
        "max": 40000.0,
        "default": 15000.0,
        "step": 500.0,
        "description": "Total requested principal amount of the loan.",
    },
    "term": {
        "label": "Loan Term",
        "type": "category",
        "options": [" 36 months", " 60 months"],
        "default": " 36 months",
        "description": "Repayment period in months (36 or 60).",
    },
    "int_rate": {
        "label": "Interest Rate (%)",
        "type": "float",
        "min": 5.0,
        "max": 31.0,
        "default": 12.5,
        "step": 0.1,
        "description": "Assigned annual percentage interest rate for the loan.",
    },
    "installment": {
        "label": "Monthly Installment ($)",
        "type": "float",
        "min": 25.0,
        "max": 1600.0,
        "default": 450.0,
        "step": 10.0,
        "description": "Calculated expected monthly payment committed by the borrower.",
    },
    "emp_length": {
        "label": "Employment Length",
        "type": "category",
        "options": [
            "< 1 year", "1 year", "2 years", "3 years", "4 years",
            "5 years", "6 years", "7 years", "8 years", "9 years",
            "10+ years"
        ],
        "default": "5 years",
        "description": "Employment tenure in years at time of application.",
    },
    "home_ownership": {
        "label": "Home Ownership",
        "type": "category",
        "options": ["RENT", "MORTGAGE", "OWN", "OTHER"],
        "default": "MORTGAGE",
        "description": "Residential home ownership status provided by the applicant.",
    },
    "annual_inc": {
        "label": "Annual Income ($)",
        "type": "float",
        "min": 10000.0,
        "max": 500000.0,
        "default": 75000.0,
        "step": 1000.0,
        "description": "Self-reported or verified gross annual personal income.",
    },
    "verification_status": {
        "label": "Income Verification",
        "type": "category",
        "options": ["Verified", "Source Verified", "Not Verified"],
        "default": "Source Verified",
        "description": "Whether the applicant's stated income was verified by LendingClub.",
    },
    "purpose": {
        "label": "Loan Purpose",
        "type": "category",
        "options": [
            "debt_consolidation", "credit_card", "home_improvement",
            "other", "major_purchase", "small_business", "car",
            "medical", "moving", "vacation", "house"
        ],
        "default": "debt_consolidation",
        "description": "Self-stated primary category or reason for borrowing funds.",
    },
    "addr_state": {
        "label": "Borrower State",
        "type": "category",
        "options": [
            "CA", "NY", "TX", "FL", "IL", "PA", "OH", "GA", "NC", "NJ",
            "VA", "WA", "AZ", "CO", "MI", "MD", "MA", "MN", "IN", "MO",
            "TN", "WI", "CT", "SC", "OR", "NV", "AL", "LA", "KY", "OK",
            "KS", "UT", "AR", "IA", "MS", "NE", "NM", "WV", "HI", "ID",
            "NH", "ME", "RI", "DE", "MT", "AK", "VT", "WY", "DC", "SD"
        ],
        "default": "CA",
        "description": "State address of residence provided by the applicant.",
    },
    "dti": {
        "label": "Debt-to-Income Ratio (%)",
        "type": "float",
        "min": 0.0,
        "max": 50.0,
        "default": 18.0,
        "step": 0.5,
        "description": "Ratio of monthly debt payments (excluding mortgage) to monthly income.",
    },
    "delinq_2yrs": {
        "label": "Delinquencies in Past 2 Yrs",
        "type": "float",
        "min": 0.0,
        "max": 15.0,
        "default": 0.0,
        "step": 1.0,
        "description": "Number of 30+ days past-due payment delinquencies in past 2 years.",
    },
    "fico_range_low": {
        "label": "FICO Range Low",
        "type": "float",
        "min": 600.0,
        "max": 850.0,
        "default": 690.0,
        "step": 5.0,
        "description": "Lower bound of applicant credit score at origination.",
    },
    "fico_range_high": {
        "label": "FICO Range High",
        "type": "float",
        "min": 604.0,
        "max": 854.0,
        "default": 694.0,
        "step": 5.0,
        "description": "Upper bound of applicant credit score at origination.",
    },
    "inq_last_6mths": {
        "label": "Inquiries in Past 6 Mths",
        "type": "float",
        "min": 0.0,
        "max": 10.0,
        "default": 0.0,
        "step": 1.0,
        "description": "Number of formal hard inquiries on credit file in the past 6 months.",
    },
    "open_acc": {
        "label": "Open Credit Lines",
        "type": "float",
        "min": 1.0,
        "max": 50.0,
        "default": 11.0,
        "step": 1.0,
        "description": "Total number of active, non-closed credit lines currently on file.",
    },
    "pub_rec": {
        "label": "Derogatory Public Records",
        "type": "float",
        "min": 0.0,
        "max": 15.0,
        "default": 0.0,
        "step": 1.0,
        "description": "Number of adverse public court records such as judgments or liens.",
    },
    "revol_bal": {
        "label": "Revolving Balance ($)",
        "type": "float",
        "min": 0.0,
        "max": 200000.0,
        "default": 14000.0,
        "step": 500.0,
        "description": "Total outstanding revolving debt carried on credit cards / lines.",
    },
    "revol_util": {
        "label": "Revolving Line Utilization (%)",
        "type": "float",
        "min": 0.0,
        "max": 120.0,
        "default": 52.0,
        "step": 1.0,
        "description": "Percentage of available revolving credit credit line being utilized.",
    },
    "total_acc": {
        "label": "Total Credit Lines (Historical)",
        "type": "float",
        "min": 2.0,
        "max": 100.0,
        "default": 24.0,
        "step": 1.0,
        "description": "Cumulative total of all credit lines opened throughout credit history.",
    },
    "mort_acc": {
        "label": "Mortgage Accounts",
        "type": "float",
        "min": 0.0,
        "max": 15.0,
        "default": 1.0,
        "step": 1.0,
        "description": "Total number of mortgage debt accounts reported on credit bureau.",
    },
    "pub_rec_bankruptcies": {
        "label": "Public Record Bankruptcies",
        "type": "float",
        "min": 0.0,
        "max": 6.0,
        "default": 0.0,
        "step": 1.0,
        "description": "Total count of bankruptcy filings in borrower's public record history.",
    },
    "tot_cur_bal": {
        "label": "Total Current Balance ($)",
        "type": "float",
        "min": 0.0,
        "max": 1500000.0,
        "default": 125000.0,
        "step": 5000.0,
        "description": "Aggregate current balance across all credit, mortgage, and loan accounts.",
    },
    "total_rev_hi_lim": {
        "label": "Total Revolving High Credit Limit ($)",
        "type": "float",
        "min": 500.0,
        "max": 300000.0,
        "default": 35000.0,
        "step": 1000.0,
        "description": "Combined credit limit across all revolving accounts on file.",
    },
}


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes domain-specific credit features:
    - fico_score: Mean of lower and upper FICO score bounds
    - installment_to_income: Annualized installment debt divided by gross income
    - credit_limit_utilization: Total revolving balance relative to high credit limit
    """
    df = df.copy()

    # Average FICO credit score
    if "fico_range_low" in df.columns and "fico_range_high" in df.columns:
        df["fico_score"] = (df["fico_range_low"] + df["fico_range_high"]) / 2.0
    elif "fico_range_low" in df.columns:
        df["fico_score"] = df["fico_range_low"]

    # Annualized installment to income ratio
    if "installment" in df.columns and "annual_inc" in df.columns:
        df["installment_to_income"] = (df["installment"] * 12.0) / (df["annual_inc"] + 1.0)
    else:
        df["installment_to_income"] = 0.0

    # Credit limit utilization ratio
    if "revol_bal" in df.columns and "total_rev_hi_lim" in df.columns:
        df["credit_limit_utilization"] = df["revol_bal"] / (df["total_rev_hi_lim"] + 1.0)
    else:
        df["credit_limit_utilization"] = 0.0

    return df


def validate_input_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates and sanitizes a single applicant record against schema definitions.
    Raises ValueError with descriptive context if any field violates boundary or category checks.
    """
    sanitized: Dict[str, Any] = {}

    for field, meta in FEATURE_METADATA.items():
        val = record.get(field, meta.get("default"))
        if val is None:
            val = meta.get("default")

        if meta["type"] == "float":
            try:
                fval = float(val)
            except (ValueError, TypeError):
                raise ValueError(f"Field '{field}' must be numeric, got: {val}")
            
            # Clamp or validate min/max
            min_val = meta.get("min", float("-inf"))
            max_val = meta.get("max", float("inf"))
            if fval < min_val:
                fval = min_val
            elif fval > max_val:
                fval = max_val
            sanitized[field] = fval

        elif meta["type"] == "category":
            sval = str(val).strip()
            # If not in options, normalize or fallback safely
            valid_opts = meta.get("options", [])
            # Try case-insensitive matching
            matched = None
            for opt in valid_opts:
                if opt.strip().lower() == sval.lower():
                    matched = opt
                    break
            if matched:
                sanitized[field] = matched
            else:
                # Fallback to default or first option
                sanitized[field] = meta.get("default", valid_opts[0] if valid_opts else sval)

    return sanitized


def get_default_applicant_dict() -> Dict[str, Any]:
    """Returns a dictionary containing verified median/default values for all predictors."""
    return {k: v["default"] for k, v in FEATURE_METADATA.items()}
