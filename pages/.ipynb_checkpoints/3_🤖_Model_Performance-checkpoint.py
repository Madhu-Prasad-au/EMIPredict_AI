import streamlit as st
import pandas as pd
import os


st.set_page_config(
    page_title="Model Performance",
    page_icon="🤖",
    layout="wide"
)


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

REPORT_DIR = os.path.join(
    BASE_DIR,
    "reports"
)


# ============================================
# LOAD RESULTS
# ============================================

classification_path = os.path.join(
    REPORT_DIR,
    "classification_model_comparison.csv"
)

regression_path = os.path.join(
    REPORT_DIR,
    "regression_model_comparison.csv"
)


classification_df = pd.read_csv(
    classification_path
)

regression_df = pd.read_csv(
    regression_path
)


# ============================================
# PAGE
# ============================================

st.title(
    "🤖 Model Performance"
)

st.write(
    """
    Comparison of the machine learning models
    trained for EMI eligibility classification
    and maximum EMI regression.
    """
)


# ============================================
# CLASSIFICATION
# ============================================

st.header(
    "Classification Models"
)

st.dataframe(
    classification_df,
    use_container_width=True
)


st.subheader(
    "Classification F1 Score"
)

classification_chart = (
    classification_df[
        ["Model", "F1 Score"]
    ]
    .set_index("Model")
)

st.bar_chart(
    classification_chart
)


st.subheader(
    "Classification ROC-AUC"
)

roc_chart = (
    classification_df[
        ["Model", "ROC-AUC"]
    ]
    .set_index("Model")
)

st.bar_chart(
    roc_chart
)


# ============================================
# REGRESSION
# ============================================

st.divider()

st.header(
    "Regression Models"
)

st.dataframe(
    regression_df,
    use_container_width=True
)


st.subheader(
    "Regression RMSE"
)

rmse_chart = (
    regression_df[
        ["Model", "RMSE"]
    ]
    .set_index("Model")
)

st.bar_chart(
    rmse_chart
)


st.subheader(
    "Regression R²"
)

r2_chart = (
    regression_df[
        ["Model", "R2 Score"]
    ]
    .set_index("Model")
)

st.bar_chart(
    r2_chart
)


# ============================================
# BEST MODELS
# ============================================

st.divider()

best_classification = (
    classification_df
    .sort_values(
        "F1 Score",
        ascending=False
    )
    .iloc[0]
)

best_regression = (
    regression_df
    .sort_values(
        "RMSE"
    )
    .iloc[0]
)


col1, col2 = st.columns(2)

with col1:

    st.success(
        f"""
        ### Best Classification Model

        **{best_classification['Model']}**

        F1 Score:
        **{best_classification['F1 Score']:.4f}**

        ROC-AUC:
        **{best_classification['ROC-AUC']:.4f}**
        """
    )


with col2:

    st.success(
        f"""
        ### Best Regression Model

        **{best_regression['Model']}**

        RMSE:
        **₹{best_regression['RMSE']:,.2f}**

        R²:
        **{best_regression['R2 Score']:.4f}**
        """
    )