"""
Credit Risk Intelligence Platform
Explainable Loan Default Prediction Using Random Forest
Enterprise Streamlit Dashboard
"""

import os
import json
import time
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.features import (
    FEATURE_METADATA,
    FEATURE_GROUPS,
    BASE_NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    get_default_applicant_dict,
    validate_input_record,
    add_engineered_features,
)
from src.predict import (
    ModelManager,
    predict_applicant_risk,
    get_risk_tier,
)
from src.explain import (
    ModelExplainer,
)

# Page configuration
st.set_page_config(
    page_title="Credit Risk Intelligence | Loan Default Prediction",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Base Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
IMAGES_DIR = os.path.join(BASE_DIR, "images")
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")


# Custom CSS for Modern Fintech Slate/Navy UI
st.markdown("""
<style>
    /* Main Background & Fonts */
    .main {
        background-color: #0F172A;
        color: #F8FAFC;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Top Banner */
    .platform-header {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    .platform-title {
        font-size: 26px;
        font-weight: 700;
        color: #38BDF8;
        margin: 0;
    }
    .platform-subtitle {
        font-size: 14px;
        color: #94A3B8;
        margin-top: 6px;
    }

    /* KPI Cards */
    .kpi-container {
        display: flex;
        flex-direction: column;
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .kpi-container:hover {
        border-color: #38BDF8;
        transform: translateY(-2px);
    }
    .kpi-title {
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94A3B8;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 24px;
        font-weight: 700;
        color: #F8FAFC;
    }
    .kpi-delta {
        font-size: 11px;
        color: #10B981;
        margin-top: 4px;
    }

    /* Containers */
    .card-box {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .card-title {
        font-size: 16px;
        font-weight: 600;
        color: #E2E8F0;
        margin-bottom: 14px;
        border-bottom: 1px solid #334155;
        padding-bottom: 8px;
    }

    /* Decision Badges */
    .badge-approved {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid #10B981;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 14px;
        display: inline-block;
    }
    .badge-declined {
        background-color: rgba(239, 68, 68, 0.15);
        color: #EF4444;
        border: 1px solid #EF4444;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 14px;
        display: inline-block;
    }

    /* Disclaimer box */
    .disclaimer-box {
        background-color: #1E293B;
        border-left: 4px solid #F59E0B;
        padding: 12px 16px;
        border-radius: 4px;
        color: #CBD5E1;
        font-size: 12px;
        margin-top: 24px;
    }
</style>
""", unsafe_allow_html=True)


# Caching Functions
@st.cache_resource
def get_cached_model_manager():
    return ModelManager.get_instance(MODELS_DIR)

@st.cache_resource
def get_cached_explainer():
    return ModelExplainer.get_instance(MODELS_DIR)

@st.cache_data
def load_cached_reports():
    eval_path = os.path.join(REPORTS_DIR, "evaluation_metrics.json")
    thresh_path = os.path.join(REPORTS_DIR, "threshold_analysis.json")
    eda_path = os.path.join(REPORTS_DIR, "eda_summary.json")
    shap_path = os.path.join(REPORTS_DIR, "shap_feature_importance.json")

    metrics = json.load(open(eval_path)) if os.path.exists(eval_path) else {}
    thresh = json.load(open(thresh_path)) if os.path.exists(thresh_path) else []
    eda = json.load(open(eda_path)) if os.path.exists(eda_path) else {}
    shap_imp = json.load(open(shap_path)) if os.path.exists(shap_path) else []
    return metrics, thresh, eda, shap_imp

@st.cache_data
def load_sample_raw_data():
    raw_path = os.path.join(DATA_DIR, "lending_club_loans.csv")
    if os.path.exists(raw_path):
        return pd.read_csv(raw_path, nrows=5000)
    return pd.DataFrame()


# Initialize Session State
if "history" not in st.session_state:
    st.session_state.history = []

if "applicant_form_data" not in st.session_state:
    st.session_state.applicant_form_data = get_default_applicant_dict()


# Sidebar Navigation
with st.sidebar:
    st.markdown("### 🛡️ Credit Risk Intelligence")
    st.markdown("*Explainable Underwriting Platform*")
    st.markdown("---")

    page = st.radio(
        "Navigation Menu",
        [
            "Executive Overview",
            "Loan Risk Prediction Workspace",
            "Model Performance Center",
            "Explainable AI Center",
            "Data Exploration Center",
            "Model Information & Governance",
            "Prediction History & Export",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("### ⚙️ Engine Status")
    st.markdown("**Model:** Random Forest (Tuned)")
    st.markdown("**Version:** v1.0.0")
    st.markdown("**Dataset:** LendingClub (50k)")
    st.markdown("**Status:** 🟢 Operational")

    st.markdown("---")
    st.caption("Antigravity ML Framework | Educational & Underwriting Support Demonstration")


# Header Banner
st.markdown("""
<div class="platform-header">
    <h1 class="platform-title">Credit Risk Intelligence Platform</h1>
    <div class="platform-subtitle">
        Explainable Machine Learning System for Loan Default Prediction & Underwriting Risk Assessment
    </div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# PAGE 1: EXECUTIVE OVERVIEW
# ==============================================================================
if page == "Executive Overview":
    metrics, thresh, eda, shap_imp = load_cached_reports()
    rf_m = metrics.get("Random Forest", {})

    st.markdown("### 📊 Executive Underwriting Overview")
    st.markdown(
        "A real-time overview of the production credit scoring engine, holdout evaluation benchmarks, "
        "and portfolio risk management dynamics."
    )

    # KPI Top Row
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-title">Test ROC-AUC</div>
            <div class="kpi-value">{rf_m.get('roc_auc', 0.7081):.4f}</div>
            <div class="kpi-delta">Tuned Random Forest</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-title">Test PR-AUC</div>
            <div class="kpi-value">{rf_m.get('pr_auc', 0.3693):.4f}</div>
            <div class="kpi-delta">vs 0.200 Baseline</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-title">Accuracy</div>
            <div class="kpi-value">{rf_m.get('accuracy', 0.7434)*100:.1f}%</div>
            <div class="kpi-delta">7,434 / 10,000 Correct</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-title">Default Specificity</div>
            <div class="kpi-value">{rf_m.get('specificity', 0.8224)*100:.1f}%</div>
            <div class="kpi-delta">82.2% Non-Defaults Approved</div>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-title">Optimal Threshold</div>
            <div class="kpi-value">0.370</div>
            <div class="kpi-delta">Saves $3.22M Losses</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 2-Column Section: Model Comparison & Target Distribution
    c_left, c_right = st.columns([3, 2])

    with c_left:
        st.markdown("#### 🏆 Baseline Model Benchmark Comparison (Test Set)")
        comp_df = pd.DataFrame([
            {"Model": "Random Forest (Tuned)", "ROC-AUC": 0.7081, "PR-AUC": 0.3693, "F1-Score": 0.3991, "Precision": 0.3748, "Recall": 0.4266, "Accuracy": 0.7434},
            {"Model": "Logistic Regression", "ROC-AUC": 0.7104, "PR-AUC": 0.3702, "F1-Score": 0.4312, "Precision": 0.3231, "Recall": 0.6480, "Accuracy": 0.6586},
            {"Model": "Decision Tree", "ROC-AUC": 0.6872, "PR-AUC": 0.3383, "F1-Score": 0.4122, "Precision": 0.2883, "Recall": 0.7231, "Accuracy": 0.5882},
        ])
        fig_bar = px.bar(
            comp_df,
            x="Model",
            y=["ROC-AUC", "PR-AUC", "Accuracy", "Precision", "Recall"],
            barmode="group",
            title="Algorithm Comparison Across Core Metrics",
            color_discrete_sequence=["#38BDF8", "#3B82F6", "#10B981", "#F59E0B", "#EC4899"],
            height=340,
        )
        fig_bar.update_layout(
            template="plotly_dark",
            paper_bgcolor="#1E293B",
            plot_bgcolor="#1E293B",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with c_right:
        st.markdown("#### 🎯 LendingClub Class Imbalance")
        fig_pie = go.Figure(data=[go.Pie(
            labels=["Fully Paid (Non-Default)", "Default / Charged Off"],
            values=[40017, 9983],
            hole=0.55,
            marker_colors=["#10B981", "#EF4444"],
            textinfo="label+percent",
        )])
        fig_pie.update_layout(
            title="Portfolio Target Outcome Ratio (4:1)",
            template="plotly_dark",
            paper_bgcolor="#1E293B",
            plot_bgcolor="#1E293B",
            showlegend=False,
            height=340,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # Key Architectural Insights
    st.markdown("#### 💡 Key Engineering & Underwriting Insights")
    st.markdown("""
    - **No Data Leakage:** All post-origination features (e.g., payments received, recoveries, collection fees) have been strictly purged from training and inference pipelines. Only applicant and bureau data known at origination is utilized.
    - **Realistic Performance:** On real consumer credit data, loan default is inherently stochastic. An ROC-AUC of **0.7081** and PR-AUC of **0.3693** on holdout test data reflects genuine predictive capability without over-optimistic data leakage.
    - **Threshold Optimization vs Default 0.50:** The default classification threshold of 0.50 optimizes raw 0-1 error. However, in retail lending, a defaulted loan carries far greater loss ($5,000) than a rejected creditworthy borrower's foregone margin ($1,200). Operating at an underwriting threshold of **0.370** minimizes portfolio capital loss, saving **$3,219,800** on the test cohort.
    """)

    st.markdown("""
    <div class="disclaimer-box">
        <strong>⚠️ Regulatory & Educational Disclaimer:</strong> This platform is designed as an interactive decision-support demonstration and educational tool. Actual lending decisions require compliance with the Equal Credit Opportunity Act (ECOA), Fair Credit Reporting Act (FCRA), formal model validation, disparity testing, and human underwriter review.
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# PAGE 2: LOAN RISK PREDICTION WORKSPACE
# ==============================================================================
elif page == "Loan Risk Prediction Workspace":
    st.markdown("### 📝 Loan Risk Prediction & Real-Time Scoring")
    st.markdown("Evaluate individual loan applications with real-time risk tiering and local SHAP explainability.")

    # Preset applicant profiles for convenient demonstration
    st.markdown("#### ⚡ Quick Preset Applicants")
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)

    if p_col1.button("🟢 Load Prime Applicant (Low Risk)"):
        preset = get_default_applicant_dict()
        preset.update({
            "loan_amnt": 12000.0, "int_rate": 7.5, "annual_inc": 115000.0,
            "dti": 9.5, "fico_range_low": 760.0, "fico_range_high": 764.0,
            "inq_last_6mths": 0.0, "delinq_2yrs": 0.0, "home_ownership": "MORTGAGE",
            "revol_util": 18.0, "tot_cur_bal": 280000.0, "total_rev_hi_lim": 65000.0
        })
        st.session_state.applicant_form_data = preset
        st.rerun()

    if p_col2.button("🟡 Load Moderate Applicant (Near-Prime)"):
        preset = get_default_applicant_dict()
        preset.update({
            "loan_amnt": 16000.0, "int_rate": 13.0, "annual_inc": 65000.0,
            "dti": 19.5, "fico_range_low": 685.0, "fico_range_high": 689.0,
            "inq_last_6mths": 1.0, "delinq_2yrs": 0.0, "home_ownership": "RENT",
            "revol_util": 55.0, "tot_cur_bal": 62000.0, "total_rev_hi_lim": 30000.0
        })
        st.session_state.applicant_form_data = preset
        st.rerun()

    if p_col3.button("🔴 Load Subprime Applicant (High Risk)"):
        preset = get_default_applicant_dict()
        preset.update({
            "loan_amnt": 28000.0, "int_rate": 22.5, "annual_inc": 42000.0,
            "dti": 34.0, "fico_range_low": 630.0, "fico_range_high": 634.0,
            "inq_last_6mths": 3.0, "delinq_2yrs": 2.0, "home_ownership": "RENT",
            "revol_util": 92.0, "tot_cur_bal": 35000.0, "total_rev_hi_lim": 12000.0,
            "pub_rec_bankruptcies": 1.0
        })
        st.session_state.applicant_form_data = preset
        st.rerun()

    if p_col4.button("🔄 Reset to Portfolio Median"):
        st.session_state.applicant_form_data = get_default_applicant_dict()
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Input Form
    current_data = st.session_state.applicant_form_data
    form_inputs = {}

    with st.form("loan_prediction_form"):
        st.markdown("#### 📋 Application Parameters")

        tab1, tab2, tab3, tab4 = st.tabs([
            "💰 Loan Details",
            "👤 Applicant Profile",
            "⚖️ Financial & Debt Burden",
            "📜 Credit Bureau & History"
        ])

        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                meta = FEATURE_METADATA["loan_amnt"]
                form_inputs["loan_amnt"] = st.number_input(
                    meta["label"], min_value=meta["min"], max_value=meta["max"],
                    value=float(current_data["loan_amnt"]), step=meta["step"], help=meta["description"]
                )
                meta = FEATURE_METADATA["int_rate"]
                form_inputs["int_rate"] = st.number_input(
                    meta["label"], min_value=meta["min"], max_value=meta["max"],
                    value=float(current_data["int_rate"]), step=meta["step"], help=meta["description"]
                )
                meta = FEATURE_METADATA["purpose"]
                cur_idx = meta["options"].index(current_data["purpose"]) if current_data["purpose"] in meta["options"] else 0
                form_inputs["purpose"] = st.selectbox(
                    meta["label"], options=meta["options"], index=cur_idx, help=meta["description"]
                )
            with c2:
                meta = FEATURE_METADATA["term"]
                t_idx = meta["options"].index(current_data["term"]) if current_data["term"] in meta["options"] else 0
                form_inputs["term"] = st.selectbox(
                    meta["label"], options=meta["options"], index=t_idx, help=meta["description"]
                )
                meta = FEATURE_METADATA["installment"]
                form_inputs["installment"] = st.number_input(
                    meta["label"], min_value=meta["min"], max_value=meta["max"],
                    value=float(current_data["installment"]), step=meta["step"], help=meta["description"]
                )

        with tab2:
            c1, c2 = st.columns(2)
            with c1:
                meta = FEATURE_METADATA["annual_inc"]
                form_inputs["annual_inc"] = st.number_input(
                    meta["label"], min_value=meta["min"], max_value=meta["max"],
                    value=float(current_data["annual_inc"]), step=meta["step"], help=meta["description"]
                )
                meta = FEATURE_METADATA["home_ownership"]
                h_idx = meta["options"].index(current_data["home_ownership"]) if current_data["home_ownership"] in meta["options"] else 0
                form_inputs["home_ownership"] = st.selectbox(
                    meta["label"], options=meta["options"], index=h_idx, help=meta["description"]
                )
                meta = FEATURE_METADATA["addr_state"]
                s_idx = meta["options"].index(current_data["addr_state"]) if current_data["addr_state"] in meta["options"] else 0
                form_inputs["addr_state"] = st.selectbox(
                    meta["label"], options=meta["options"], index=s_idx, help=meta["description"]
                )
            with c2:
                meta = FEATURE_METADATA["emp_length"]
                e_idx = meta["options"].index(current_data["emp_length"]) if current_data["emp_length"] in meta["options"] else 0
                form_inputs["emp_length"] = st.selectbox(
                    meta["label"], options=meta["options"], index=e_idx, help=meta["description"]
                )
                meta = FEATURE_METADATA["verification_status"]
                v_idx = meta["options"].index(current_data["verification_status"]) if current_data["verification_status"] in meta["options"] else 0
                form_inputs["verification_status"] = st.selectbox(
                    meta["label"], options=meta["options"], index=v_idx, help=meta["description"]
                )

        with tab3:
            c1, c2 = st.columns(2)
            with c1:
                meta = FEATURE_METADATA["dti"]
                form_inputs["dti"] = st.number_input(
                    meta["label"], min_value=meta["min"], max_value=meta["max"],
                    value=float(current_data["dti"]), step=meta["step"], help=meta["description"]
                )
                meta = FEATURE_METADATA["tot_cur_bal"]
                form_inputs["tot_cur_bal"] = st.number_input(
                    meta["label"], min_value=meta["min"], max_value=meta["max"],
                    value=float(current_data["tot_cur_bal"]), step=meta["step"], help=meta["description"]
                )
                meta = FEATURE_METADATA["total_rev_hi_lim"]
                form_inputs["total_rev_hi_lim"] = st.number_input(
                    meta["label"], min_value=meta["min"], max_value=meta["max"],
                    value=float(current_data["total_rev_hi_lim"]), step=meta["step"], help=meta["description"]
                )
            with c2:
                meta = FEATURE_METADATA["revol_bal"]
                form_inputs["revol_bal"] = st.number_input(
                    meta["label"], min_value=meta["min"], max_value=meta["max"],
                    value=float(current_data["revol_bal"]), step=meta["step"], help=meta["description"]
                )
                meta = FEATURE_METADATA["revol_util"]
                form_inputs["revol_util"] = st.number_input(
                    meta["label"], min_value=meta["min"], max_value=meta["max"],
                    value=float(current_data["revol_util"]), step=meta["step"], help=meta["description"]
                )

        with tab4:
            c1, c2, c3 = st.columns(3)
            with c1:
                meta = FEATURE_METADATA["fico_range_low"]
                form_inputs["fico_range_low"] = st.number_input(
                    meta["label"], min_value=meta["min"], max_value=meta["max"],
                    value=float(current_data["fico_range_low"]), step=meta["step"], help=meta["description"]
                )
                meta = FEATURE_METADATA["fico_range_high"]
                form_inputs["fico_range_high"] = st.number_input(
                    meta["label"], min_value=meta["min"], max_value=meta["max"],
                    value=float(current_data["fico_range_high"]), step=meta["step"], help=meta["description"]
                )
                meta = FEATURE_METADATA["delinq_2yrs"]
                form_inputs["delinq_2yrs"] = st.number_input(
                    meta["label"], min_value=meta["min"], max_value=meta["max"],
                    value=float(current_data["delinq_2yrs"]), step=meta["step"], help=meta["description"]
                )
            with c2:
                meta = FEATURE_METADATA["inq_last_6mths"]
                form_inputs["inq_last_6mths"] = st.number_input(
                    meta["label"], min_value=meta["min"], max_value=meta["max"],
                    value=float(current_data["inq_last_6mths"]), step=meta["step"], help=meta["description"]
                )
                meta = FEATURE_METADATA["open_acc"]
                form_inputs["open_acc"] = st.number_input(
                    meta["label"], min_value=meta["min"], max_value=meta["max"],
                    value=float(current_data["open_acc"]), step=meta["step"], help=meta["description"]
                )
                meta = FEATURE_METADATA["total_acc"]
                form_inputs["total_acc"] = st.number_input(
                    meta["label"], min_value=meta["min"], max_value=meta["max"],
                    value=float(current_data["total_acc"]), step=meta["step"], help=meta["description"]
                )
            with c3:
                meta = FEATURE_METADATA["pub_rec"]
                form_inputs["pub_rec"] = st.number_input(
                    meta["label"], min_value=meta["min"], max_value=meta["max"],
                    value=float(current_data["pub_rec"]), step=meta["step"], help=meta["description"]
                )
                meta = FEATURE_METADATA["pub_rec_bankruptcies"]
                form_inputs["pub_rec_bankruptcies"] = st.number_input(
                    meta["label"], min_value=meta["min"], max_value=meta["max"],
                    value=float(current_data["pub_rec_bankruptcies"]), step=meta["step"], help=meta["description"]
                )
                meta = FEATURE_METADATA["mort_acc"]
                form_inputs["mort_acc"] = st.number_input(
                    meta["label"], min_value=meta["min"], max_value=meta["max"],
                    value=float(current_data["mort_acc"]), step=meta["step"], help=meta["description"]
                )

        st.markdown("<br>", unsafe_allow_html=True)
        col_submit, col_thresh = st.columns([1, 2])
        with col_thresh:
            threshold_val = st.slider(
                "🎯 Underwriting Decision Cutoff (Default Threshold)",
                min_value=0.10,
                max_value=0.80,
                value=0.37,
                step=0.01,
                help="Loans with default probability >= cutoff are flagged for Decline. Default recommended: 0.370."
            )
        with col_submit:
            submit_btn = st.form_submit_button("⚡ Evaluate Credit Risk", use_container_width=True)

    # Perform Prediction
    if submit_btn:
        with st.spinner("Analyzing credit file through Random Forest ensemble..."):
            result = predict_applicant_risk(form_inputs, MODELS_DIR, threshold=threshold_val)
            explainer = get_cached_explainer()
            local_shap = explainer.explain_instance(form_inputs, top_n=8)

            # Record session history
            history_entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "loan_amnt": form_inputs["loan_amnt"],
                "annual_inc": form_inputs["annual_inc"],
                "fico": (form_inputs["fico_range_low"] + form_inputs["fico_range_high"]) / 2,
                "dti": form_inputs["dti"],
                "default_prob": result["default_probability"],
                "threshold": threshold_val,
                "risk_tier": result["risk_tier"],
                "decision": result["decision"],
            }
            st.session_state.history.append(history_entry)

        st.markdown("---")
        st.markdown("### 📋 Credit Decision & Risk Analysis")

        # Top Results Cards
        r_col1, r_col2, r_col3, r_col4 = st.columns([2, 2, 2, 3])
        with r_col1:
            badge_class = "badge-approved" if result["decision"] == "Approve Loan" else "badge-declined"
            st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-title">Underwriting Decision</div>
                <div style="margin-top: 6px;"><span class="{badge_class}">{result['decision']}</span></div>
                <div class="kpi-delta">Cutoff: {threshold_val:.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        with r_col2:
            st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-title">Default Probability</div>
                <div class="kpi-value" style="color: {result['risk_color']};">{result['default_probability']*100:.1f}%</div>
                <div class="kpi-delta">Repayment: {result['repayment_probability']*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        with r_col3:
            st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-title">Risk Rating</div>
                <div class="kpi-value" style="font-size: 18px; color: {result['risk_color']};">{result['risk_tier'].split(' - ')[0]}</div>
                <div class="kpi-delta">{result['risk_tier'].split(' - ')[1]}</div>
            </div>
            """, unsafe_allow_html=True)

        with r_col4:
            st.markdown(f"""
            <div class="kpi-container" style="text-align: left;">
                <div class="kpi-title">Policy Assessment</div>
                <div style="font-size: 13px; color: #CBD5E1; margin-top: 4px;">{result['risk_description']}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Risk Gauge Chart & Key Drivers
        g_left, g_right = st.columns([2, 3])

        with g_left:
            st.markdown("#### 🧭 Risk Gauge")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result["default_probability"] * 100,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Default Probability (%)", "font": {"size": 15, "color": "#E2E8F0"}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94A3B8"},
                    "bar": {"color": result["risk_color"]},
                    "bgcolor": "#1E293B",
                    "steps": [
                        {"range": [0, 15], "color": "rgba(16, 185, 129, 0.25)"},
                        {"range": [15, 30], "color": "rgba(245, 158, 11, 0.25)"},
                        {"range": [30, 50], "color": "rgba(249, 115, 22, 0.25)"},
                        {"range": [50, 100], "color": "rgba(239, 68, 68, 0.25)"},
                    ],
                    "threshold": {
                        "line": {"color": "#38BDF8", "width": 4},
                        "thickness": 0.8,
                        "value": threshold_val * 100,
                    },
                },
            ))
            fig_gauge.update_layout(
                paper_bgcolor="#1E293B",
                height=260,
                margin=dict(l=20, r=20, t=30, b=10),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with g_right:
            st.markdown("#### 🔍 Primary Credit Risk Drivers")
            st.markdown(f"**Underwriter Summary:** {local_shap['natural_language_summary']}")
            st.markdown("<br>", unsafe_allow_html=True)

            d_col1, d_col2 = st.columns(2)
            with d_col1:
                st.markdown("**⚠️ Adverse Factors (Pushing Risk Up):**")
                if result["risk_factors"]:
                    for rf in result["risk_factors"]:
                        st.markdown(f"- 🔴 {rf}")
                else:
                    st.markdown("*- No significant adverse flags detected.*")

            with d_col2:
                st.markdown("**🛡️ Protective Factors (Offsetting Risk):**")
                if result["protective_factors"]:
                    for pf in result["protective_factors"]:
                        st.markdown(f"- 🟢 {pf}")
                else:
                    st.markdown("*- Limited protective credit factors.*")

        # Local SHAP Contribution Waterfall Bar Chart
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 🔬 Local SHAP Attribution Breakdown")
        st.markdown("Each bar indicates how strongly a specific applicant characteristic shifted the default probability from the portfolio baseline expectation.")

        shap_df = pd.DataFrame(local_shap["top_factors"])
        shap_df["clean_feature"] = shap_df["feature"].str.replace("num__", "").str.replace("cat__", "")
        shap_df["abs_shap"] = shap_df["shap_value"].abs()
        shap_df = shap_df.sort_values("shap_value", ascending=True)

        fig_shap_local = px.bar(
            shap_df,
            x="shap_value",
            y="clean_feature",
            orientation="h",
            color="direction",
            color_discrete_map={"Risk-Increasing": "#EF4444", "Risk-Reducing": "#10B981"},
            title="Local SHAP Feature Attribution (Shapley Contribution)",
            labels={"shap_value": "SHAP Contribution to Default Probability", "clean_feature": "Feature"},
            height=320,
        )
        fig_shap_local.update_layout(
            template="plotly_dark",
            paper_bgcolor="#1E293B",
            plot_bgcolor="#1E293B",
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_shap_local, use_container_width=True)


# ==============================================================================
# PAGE 3: MODEL PERFORMANCE CENTER
# ==============================================================================
elif page == "Model Performance Center":
    metrics, thresh, eda, shap_imp = load_cached_reports()
    rf_m = metrics.get("Random Forest", {})

    st.markdown("### 📈 Model Performance & Advanced Analytics")
    st.markdown("Detailed evaluation on the 10,000-sample holdout test cohort.")

    # Threshold Simulator Interactive Slider
    st.markdown("#### 🎚️ Dynamic Decision Threshold Simulator")
    selected_t = st.slider(
        "Adjust Operating Threshold to Inspect Trade-offs:",
        min_value=0.10,
        max_value=0.80,
        value=0.37,
        step=0.01,
    )

    # Find closest entry in threshold curve
    closest_t_entry = min(thresh, key=lambda x: abs(x["threshold"] - selected_t)) if thresh else {}

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Expected Precision", f"{closest_t_entry.get('precision', 0.3748)*100:.1f}%")
    with c2:
        st.metric("Expected Recall", f"{closest_t_entry.get('recall', 0.4266)*100:.1f}%")
    with c3:
        st.metric("Expected F1-Score", f"{closest_t_entry.get('f1_score', 0.3991):.4f}")
    with c4:
        st.metric("Expected Portfolio Loss", f"${closest_t_entry.get('total_business_loss', 6765200):,.0f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Interactive ROC & PR Curves
    col_roc, col_pr = st.columns(2)
    with col_roc:
        st.markdown("#### ⚡ ROC Curve Analysis")
        if os.path.exists(os.path.join(IMAGES_DIR, "roc_curve.png")):
            st.image(os.path.join(IMAGES_DIR, "roc_curve.png"), caption="ROC Curve Benchmark Comparison (AUC: 0.7081)")
        else:
            st.info("ROC plot loading...")

    with col_pr:
        st.markdown("#### 🎯 Precision-Recall Curve (Imbalance Focus)")
        if os.path.exists(os.path.join(IMAGES_DIR, "pr_curve.png")):
            st.image(os.path.join(IMAGES_DIR, "pr_curve.png"), caption="PR Curve Benchmark (PR-AUC: 0.3693 vs 0.200 Baseline)")
        else:
            st.info("PR plot loading...")

    st.markdown("<br>", unsafe_allow_html=True)

    # Confusion Matrix & Financial Cost Curve
    col_cm, col_cost = st.columns(2)
    with col_cm:
        st.markdown("#### 🔲 Test Cohort Confusion Matrix")
        if os.path.exists(os.path.join(IMAGES_DIR, "confusion_matrix.png")):
            st.image(os.path.join(IMAGES_DIR, "confusion_matrix.png"), caption="Confusion Matrix on 10,000 Unseen Loans")

    with col_cost:
        st.markdown("#### 📉 Portfolio Financial Loss Curve")
        if thresh:
            thresh_df = pd.DataFrame(thresh)
            fig_loss = px.line(
                thresh_df,
                x="threshold",
                y="total_business_loss",
                title="Expected Portfolio Capital Loss ($) vs Decision Threshold",
                labels={"threshold": "Classification Cutoff", "total_business_loss": "Net Financial Loss ($)"},
            )
            fig_loss.add_vline(x=0.37, line_dash="dash", line_color="#10B981", annotation_text="Cost-Optimal: 0.370")
            fig_loss.add_vline(x=0.50, line_dash="dot", line_color="#EF4444", annotation_text="Default: 0.500")
            fig_loss.update_layout(
                template="plotly_dark",
                paper_bgcolor="#1E293B",
                plot_bgcolor="#1E293B",
                height=320,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_loss, use_container_width=True)

    # Detailed Metrics Comparison Table
    st.markdown("#### 📑 Verified Multi-Model Evaluation Benchmark")
    eval_table = pd.DataFrame([
        {
            "Algorithm": "Random Forest (Tuned)",
            "Holdout ROC-AUC": 0.7081,
            "Holdout PR-AUC": 0.3693,
            "Accuracy": 0.7434,
            "F1-Score": 0.3991,
            "Precision": 0.3748,
            "Recall": 0.4266,
            "Specificity": 0.8224,
            "Brier Score": 0.1788,
        },
        {
            "Algorithm": "Logistic Regression (Baseline 1)",
            "Holdout ROC-AUC": 0.7104,
            "Holdout PR-AUC": 0.3702,
            "Accuracy": 0.6586,
            "F1-Score": 0.4312,
            "Precision": 0.3231,
            "Recall": 0.6480,
            "Specificity": 0.6613,
            "Brier Score": 0.2174,
        },
        {
            "Algorithm": "Decision Tree (Baseline 2)",
            "Holdout ROC-AUC": 0.6872,
            "Holdout PR-AUC": 0.3383,
            "Accuracy": 0.5882,
            "F1-Score": 0.4122,
            "Precision": 0.2883,
            "Recall": 0.7231,
            "Specificity": 0.5545,
            "Brier Score": 0.2233,
        },
    ])
    st.dataframe(eval_table.set_index("Algorithm"), use_container_width=True)


# ==============================================================================
# PAGE 4: EXPLAINABLE AI CENTER
# ==============================================================================
elif page == "Explainable AI Center":
    metrics, thresh, eda, shap_imp = load_cached_reports()

    st.markdown("### 🧠 Explainable AI Center (SHAP)")
    st.markdown("Global and local model interpretability powered by Shapley Additive Explanations.")

    # Global SHAP Summary
    col_bee, col_bar = st.columns(2)
    with col_bee:
        st.markdown("#### 🐝 SHAP Beeswarm Summary Plot")
        if os.path.exists(os.path.join(IMAGES_DIR, "shap_summary.png")):
            st.image(os.path.join(IMAGES_DIR, "shap_summary.png"), caption="Global Feature Impact on Default Probability")

    with col_bar:
        st.markdown("#### 📊 Global Feature Importance Ranking")
        if os.path.exists(os.path.join(IMAGES_DIR, "shap_bar_importance.png")):
            st.image(os.path.join(IMAGES_DIR, "shap_bar_importance.png"), caption="Top 15 Features by Mean Absolute SHAP Value")

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature Importance Ranking Table
    st.markdown("#### 📑 SHAP Importance Ranking")
    if shap_imp:
        imp_df = pd.DataFrame(shap_imp[:15])
        imp_df["clean_feature"] = imp_df["feature"].str.replace("num__", "").str.replace("cat__", "")
        st.dataframe(imp_df[["rank", "clean_feature", "mean_abs_shap"]].set_index("rank"), use_container_width=True)

    # Explainability Interpretation Guide
    st.markdown("#### 📘 How to Interpret SHAP Values in Lending")
    st.markdown("""
    1. **Additive Baseline:** The base value (expected value) is the average model output across the training population (~19.9% default risk).
    2. **Positive SHAP Values ($>0$):** Push the predicted probability of default **higher** (increasing credit risk). For example, higher interest rate or higher debt-to-income ratio.
    3. **Negative SHAP Values ($<0$):** Push the predicted probability **lower** (mitigating risk). For example, high FICO score, extensive credit history, or homeownership.
    4. **Non-Causal Distinction:** SHAP identifies associative feature attributions in the model's learned decision surface. It does **not** establish direct causal relationships.
    5. **Fair Lending & Adverse Action:** Under FCRA/ECOA regulations, when a loan is declined, lenders must provide principal reasons. Local SHAP rank ordering provides an auditable basis for selecting statutory adverse action notice drivers.
    """)


# ==============================================================================
# PAGE 5: DATA EXPLORATION CENTER
# ==============================================================================
elif page == "Data Exploration Center":
    metrics, thresh, eda, shap_imp = load_cached_reports()

    st.markdown("### 🔍 Data Exploration Center (LendingClub)")
    st.markdown("Comprehensive statistical exploration of 50,000 historical consumer loan applications.")

    # Summary Metrics Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Dataset Sample Size", f"{eda.get('dataset_rows', 50000):,} Loans")
    with c2:
        st.metric("Total Predictors", f"{eda.get('dataset_columns', 25)} Variables")
    with c3:
        st.metric("Historical Default Rate", f"{eda.get('default_rate', 0.1997)*100:.2f}%")
    with c4:
        st.metric("Missing Values", "0 (Imputed in Pipeline)")

    st.markdown("<br>", unsafe_allow_html=True)

    # EDA Visualizations
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown("#### 🎯 FICO Credit Score vs Default Rate")
        if os.path.exists(os.path.join(IMAGES_DIR, "eda_fico_vs_default.png")):
            st.image(os.path.join(IMAGES_DIR, "eda_fico_vs_default.png"), caption="Observed Default Probability by FICO Tier")

    with col_d2:
        st.markdown("#### ⚖️ Debt-to-Income (DTI) Density Distribution")
        if os.path.exists(os.path.join(IMAGES_DIR, "eda_dti_distribution.png")):
            st.image(os.path.join(IMAGES_DIR, "eda_dti_distribution.png"), caption="DTI Density Separated by Loan Outcome")

    st.markdown("<br>", unsafe_allow_html=True)

    # Correlation Matrix
    st.markdown("#### 🌡️ Feature Correlation Heatmap")
    if os.path.exists(os.path.join(IMAGES_DIR, "eda_correlation_matrix.png")):
        st.image(os.path.join(IMAGES_DIR, "eda_correlation_matrix.png"), caption="Pearson Correlation Heatmap Across Core Credit Metrics")

    # Raw Sample Explorer
    st.markdown("#### 📄 Data Table Preview (First 100 Records)")
    sample_raw = load_sample_raw_data()
    if not sample_raw.empty:
        st.dataframe(sample_raw.head(100), use_container_width=True)


# ==============================================================================
# PAGE 6: MODEL INFORMATION & GOVERNANCE
# ==============================================================================
elif page == "Model Information & Governance":
    st.markdown("### 🏛️ Model Governance, Architecture & Ethics")
    st.markdown("Standardized documentation conforming to enterprise Model Risk Management (MRM) practices.")

    tab_arch, tab_leak, tab_ethics = st.tabs([
        "🏗️ System Architecture",
        "🛡️ Leakage Prevention Audit",
        "⚖️ Fairness, Ethics & Limitations"
    ])

    with tab_arch:
        st.markdown("#### Platform System Architecture")
        if os.path.exists(os.path.join(IMAGES_DIR, "architecture_diagram.png")):
            st.image(os.path.join(IMAGES_DIR, "architecture_diagram.png"), caption="End-to-End System Architecture")

        st.markdown("""
        **Pipeline Specification:**
        - **Data Processing:** Stratified 80/20 partitioning, keeping target default rate equal across splits (~19.97%).
        - **Numerical Preprocessing:** Median imputation + standard z-score normalization.
        - **Categorical Preprocessing:** Most frequent imputation + OneHotEncoding (`handle_unknown='ignore'`).
        - **Ensemble Model:** Scikit-Learn `RandomForestClassifier` with 200 estimators, max depth 20, class weighting.
        - **Explainability:** Lundberg & Lee `TreeExplainer` providing local Shapley attributions in $<50$ ms.
        """)

    with tab_leak:
        st.markdown("#### Strict Data Leakage Prevention")
        st.markdown("""
        A common failure mode in credit risk modeling is inadvertently including post-origination variables that reveal the outcome.
        In this platform, the following variables have been **strictly audited and permanently excluded**:
        - `last_pymnt_d`, `last_pymnt_amnt`: Revealed actual payment status and timing.
        - `total_pymnt`, `total_pymnt_inv`: Revealed total amount returned to investors.
        - `total_rec_prncp`, `total_rec_int`: Disclosed whether principal was fully amortized.
        - `recoveries`, `collection_recovery_fee`: Disclosed defaulted collections.
        - `hardship_status`, `settlement_status`: Post-origination distress events.

        **Result:** The feature space consists exclusively of variables verifiable on the loan application date.
        """)

    with tab_ethics:
        st.markdown("#### Fairness, Ethics & Limitations")
        st.markdown("""
        - **Historical Bias:** Historical LendingClub loan performance reflects historical economic environments (2007-2018). Macroeconomic cycles and interest rate shifts affect default rates.
        - **Disparate Impact Risks:** Credit predictors such as zip code, state, or certain debt proxies can correlate with protected classes. Production deployment requires Fair Lending Disparate Impact testing (e.g., 80% Four-Fifths rule analysis).
        - **Human in the Loop:** Automated scoring should serve as decision-support for credit underwriters, rather than fully autonomous adverse action.
        - **Probability Calibration:** The Brier score of 0.1788 indicates good calibration, but non-linear probability recalibration (Isotonic or Platt scaling) is recommended for extreme tails.
        """)


# ==============================================================================
# PAGE 7: PREDICTION HISTORY & EXPORT
# ==============================================================================
elif page == "Prediction History & Export":
    st.markdown("### 📜 Session Prediction History & Audit Log")
    st.markdown("Audit log of applicant assessments conducted during the active session.")

    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True)

        col_dl, col_clear = st.columns([1, 4])
        with col_dl:
            csv_data = hist_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Export Audit Log (CSV)",
                data=csv_data,
                file_name=f"credit_risk_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col_clear:
            if st.button("🗑️ Clear Session History"):
                st.session_state.history = []
                st.rerun()
    else:
        st.info("No loan predictions performed in this session yet. Navigate to the **Loan Risk Prediction Workspace** to evaluate applicants.")
