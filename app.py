import streamlit as st
import pandas as pd
import os

from utils.data_utils import load_data
from utils.model_utils import (
    load_classification_model,
    load_regression_model,
    load_feature_info
)


# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="EMIPredict AI",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================
# LOAD DATA / MODELS
# ============================================

@st.cache_resource
def get_models():

    classification_model = (
        load_classification_model()
    )

    regression_model = (
        load_regression_model()
    )

    feature_info = load_feature_info()

    return (
        classification_model,
        regression_model,
        feature_info
    )


@st.cache_data
def get_data():

    return load_data()


try:

    (
        classification_model,
        regression_model,
        feature_info
    ) = get_models()

    df = get_data()

except Exception as e:

    st.error(
        f"Unable to load project files: {e}"
    )

    st.stop()


# ============================================
# SIDEBAR
# ============================================

with st.sidebar:

    st.title("💰 EMIPredict AI")

    st.markdown(
        """
        **Intelligent Financial Risk
        Assessment Platform**
        """
    )

    st.divider()

    st.markdown("### Navigation")

    st.info(
        """
        Use the pages on the left to:

        🔮 Predict EMI eligibility

        📊 Explore financial data

        🤖 Compare ML models

        🗄️ Manage financial records
        """
    )

    st.divider()

    st.caption(
        "FinTech & Banking | Machine Learning"
    )


# ============================================
# HEADER
# ============================================

st.title(
    "💰 EMIPredict AI"
)

st.subheader(
    "Intelligent Financial Risk Assessment Platform"
)

st.markdown(
    """
    EMIPredict AI uses machine learning to assess
    EMI eligibility and estimate the maximum
    affordable monthly EMI for customers.
    """
)


# ============================================
# KEY METRICS
# ============================================

st.divider()

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Total Records",
        f"{len(df):,}"
    )

with col2:

    st.metric(
        "Input Features",
        "22"
    )

with col3:

    st.metric(
        "EMI Scenarios",
        df["emi_scenario"].nunique()
    )

with col4:

    st.metric(
        "Eligibility Classes",
        df["emi_eligibility"].nunique()
    )


# ============================================
# MODEL INFORMATION
# ============================================

st.divider()

st.subheader(
    "🤖 Production Models"
)

col1, col2 = st.columns(2)

with col1:

    st.markdown(
        "### Classification"
    )

    st.success(
        feature_info[
            "best_classification_model"
        ]
    )

    st.write(
        "Predicts EMI eligibility:"
    )

    st.write(
        ", ".join(
            feature_info[
                "classification_classes"
            ]
        )
    )


with col2:

    st.markdown(
        "### Regression"
    )

    st.success(
        feature_info[
            "best_regression_model"
        ]
    )

    st.write(
        "Predicts maximum safe monthly EMI."
    )


# ============================================
# DATASET PREVIEW
# ============================================

st.divider()

st.subheader(
    "📋 Dataset Preview"
)

st.dataframe(
    df.head(10),
    use_container_width=True
)


# ============================================
# PROJECT PIPELINE
# ============================================

st.divider()

st.subheader(
    "🔄 Project Pipeline"
)

st.markdown(
    """
    **Dataset**
    ↓
    **Data Cleaning**
    ↓
    **EDA & Feature Engineering**
    ↓
    **Classification + Regression**
    ↓
    **MLflow Experiment Tracking**
    ↓
    **Best Model Selection**
    ↓
    **Real-Time Prediction**
    """
)


st.divider()

st.caption(
    "EMIPredict AI — Financial Risk Assessment Platform"
)