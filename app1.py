
import streamlit as st
import pandas as pd
import html
import traceback

from utils.data_utils import load_data
from utils.model_utils import (
    load_classification_model,
    load_regression_model,
    load_feature_info,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EMIPredict AI | Financial Intelligence",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM DESIGN SYSTEM
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --navy: #101a2c;
        --navy-light: #192940;
        --teal: #16b8a6;
        --teal-light: #dff8f3;
        --muted: #718096;
        --border: #e5eaf1;
        --surface: #ffffff;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background: #f5f7fb;
    }

    [data-testid="stHeader"] {
        background: rgba(245, 247, 251, 0.92);
    }

    [data-testid="stSidebar"] {
        background: #101a2c;
        border-right: 1px solid #25334a;
    }

    [data-testid="stSidebar"] * {
        color: #e9f0fa;
    }

    [data-testid="stSidebar"] hr {
        border-color: #334157;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        line-height: 1.7;
    }

    h1, h2, h3 {
        font-family: 'Manrope', sans-serif !important;
        color: #15243b;
        letter-spacing: -0.6px;
    }

    .hero {
        position: relative;
        overflow: hidden;
        padding: 34px 36px;
        border-radius: 22px;
        background: linear-gradient(120deg, #101a2c 0%, #1b3450 72%, #145b63 100%);
        color: white;
        margin-bottom: 26px;
        border: 1px solid rgba(255,255,255,0.08);
    }

    .hero::after {
        content: "";
        position: absolute;
        width: 260px;
        height: 260px;
        right: -70px;
        top: -120px;
        border-radius: 50%;
        border: 38px solid rgba(65, 221, 196, 0.10);
    }

    .hero-kicker {
        color: #66e4d1;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .hero-title {
        font-family: 'Manrope', sans-serif;
        font-size: clamp(28px, 3.2vw, 43px);
        font-weight: 800;
        line-height: 1.2;
        letter-spacing: -1.5px;
        color: #ffffff;
        margin-bottom: 12px;
    }

    .hero-copy {
        color: #d0dbea;
        font-size: 15px;
        max-width: 690px;
        line-height: 1.8;
        margin-bottom: 20px;
    }

    .hero-tag {
        display: inline-block;
        padding: 7px 12px;
        margin: 0 7px 5px 0;
        border: 1px solid rgba(255,255,255,0.19);
        border-radius: 20px;
        color: #e5f5f4;
        font-size: 11px;
        font-weight: 600;
    }

    .section-label {
        font-size: 11px;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #73849a;
        font-weight: 700;
        margin-bottom: 6px;
    }

    .section-heading {
        font-family: 'Manrope', sans-serif;
        color: #17263c;
        font-size: 22px;
        font-weight: 800;
        letter-spacing: -0.7px;
        margin-bottom: 5px;
    }

    .section-copy {
        color: #78869a;
        font-size: 13px;
        line-height: 1.7;
        margin-bottom: 16px;
    }

    .metric-card {
        background: #ffffff;
        border: 1px solid #e5eaf1;
        border-radius: 17px;
        padding: 20px;
        min-height: 135px;
        box-shadow: 0 3px 14px rgba(23, 42, 69, 0.025);
    }

    .metric-label {
        font-size: 12px;
        font-weight: 600;
        color: #7b899c;
        margin-bottom: 13px;
    }

    .metric-value {
        color: #15243b;
        font-family: 'Manrope', sans-serif;
        font-size: 28px;
        font-weight: 800;
        line-height: 1.2;
        overflow-wrap: anywhere;
    }

    .metric-note {
        color: #8a97a8;
        font-size: 11px;
        margin-top: 9px;
    }

    .model-card {
        background: #ffffff;
        border: 1px solid #e5eaf1;
        border-radius: 18px;
        padding: 24px;
        min-height: 215px;
        box-shadow: 0 3px 14px rgba(23, 42, 69, 0.025);
    }

    .model-icon {
        display: inline-block;
        background: #e1f8f3;
        color: #078f81;
        border-radius: 11px;
        padding: 11px;
        font-size: 21px;
        margin-bottom: 14px;
    }

    .model-name {
        font-family: 'Manrope', sans-serif;
        font-size: 19px;
        font-weight: 800;
        color: #17263c;
        margin-bottom: 8px;
        overflow-wrap: anywhere;
    }

    .model-description {
        color: #748399;
        font-size: 13px;
        line-height: 1.8;
    }

    .status-pill {
        display: inline-block;
        background: #e1f8ee;
        color: #16825a;
        padding: 5px 10px;
        border-radius: 20px;
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 0.7px;
        margin-bottom: 15px;
    }

    .pipeline-card {
        background: white;
        border: 1px solid #e5eaf1;
        border-radius: 15px;
        padding: 18px;
        min-height: 145px;
        margin-bottom: 8px;
    }

    .pipeline-number {
        font-size: 11px;
        color: #0a9c8d;
        font-weight: 800;
        letter-spacing: 1px;
        margin-bottom: 13px;
    }

    .pipeline-title {
        font-family: 'Manrope', sans-serif;
        font-size: 14px;
        color: #1a2a41;
        font-weight: 800;
        margin-bottom: 7px;
    }

    .pipeline-copy {
        font-size: 12px;
        color: #7c899a;
        line-height: 1.7;
    }

    .footer {
        color: #8995a5;
        font-size: 11px;
        text-align: center;
        padding: 22px 0 12px 0;
    }

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5eaf1;
        border-radius: 15px;
        padding: 16px;
    }

    div[data-testid="stMetricLabel"] {
        color: #78869a;
    }

    div[data-testid="stMetricValue"] {
        color: #15243b;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #e5eaf1;
        border-radius: 12px;
        overflow: hidden;
    }

    .stButton button {
        border-radius: 10px;
        font-weight: 600;
    }

    @media (max-width: 768px) {
        .hero {
            padding: 25px 22px;
        }
        .metric-card {
            padding: 15px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div style="padding: 12px 0 20px 0;">
            <div style="font-size: 12px; color: #5ee6d0;
                        font-weight: 800; letter-spacing: 2px;">
                EMIPREDICT
            </div>
            <div style="font-family: Manrope, sans-serif;
                        font-size: 25px; font-weight: 800;
                        color: white; margin-top: 4px;">
                AI Platform<span style="color:#42d7c0;">.</span>
            </div>
            <div style="font-size: 11px; color: #9eafc5;
                        margin-top: 8px;">
                FINANCIAL INTELLIGENCE
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("### Navigation")
    st.caption(
        "Use the pages in the Streamlit navigation menu "
        "to access prediction, analytics, and data management."
    )

    st.markdown(
        """
        <div style="background:#192940; padding:15px;
                    border-radius:12px; margin-top:20px;">
            <div style="color:#5ee6d0; font-size:11px;
                        font-weight:800; letter-spacing:1px;">
                PLATFORM MODULES
            </div>
            <div style="font-size:12px; line-height:2.2;
                        margin-top:8px;">
                ◈ EMI Eligibility Prediction<br>
                ◈ EMI Amount Estimation<br>
                ◈ Financial Data Analytics<br>
                ◈ Model Experiment Tracking<br>
                ◈ Financial Record Management
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.caption("FinTech • Machine Learning • Decision Support")


# ============================================================
# ERROR REPORTING
# ============================================================

def show_error(title, exc):
    """Display a readable error and its Python traceback."""
    st.error(title)
    st.code(f"{type(exc).__name__}: {exc}", language="text")

    with st.expander("View full technical traceback"):
        st.code(traceback.format_exc(), language="text")

    st.info(
        "Copy the exception type, message, and traceback when "
        "troubleshooting this problem. Avoid sharing credentials "
        "or private customer data."
    )


# ============================================================
# LOAD MODELS AND DATA
# ============================================================

try:
    with st.spinner("Loading financial intelligence platform..."):
        classification_model = load_classification_model()
        regression_model = load_regression_model()
        feature_info = load_feature_info()
        df = load_data()

    if df is None:
        raise ValueError("load_data() returned None.")

    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    if df.empty:
        st.warning("The dataset loaded successfully but contains no records.")

except Exception as exc:
    st.title("EMIPredict AI")
    st.error("The application could not load its required data or models.")
    show_error("Startup failure details", exc)
    st.stop()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_unique_count(data, column):
    if column in data.columns:
        return int(data[column].nunique(dropna=True))
    return None


def format_number(value):
    if value is None:
        return "—"
    return f"{value:,}"


def metric_card(label, value, note):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{html.escape(str(label))}</div>
            <div class="metric-value">{html.escape(str(value))}</div>
            <div class="metric-note">{html.escape(str(note))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# MAIN DASHBOARD
# ============================================================

try:
    # Hero section
    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">FINANCIAL INTELLIGENCE PLATFORM</div>
            <div class="hero-title">
                Smarter EMI decisions.<br>
                Better financial insights.
            </div>
            <div class="hero-copy">
                An integrated machine learning platform for assessing
                EMI eligibility, estimating affordable monthly payments,
                and exploring financial data through predictive analytics.
            </div>
            <span class="hero-tag">Machine Learning</span>
            <span class="hero-tag">Eligibility Classification</span>
            <span class="hero-tag">EMI Regression</span>
            <span class="hero-tag">Financial Analytics</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Overview
    st.markdown(
        """
        <div class="section-label">OVERVIEW</div>
        <div class="section-heading">Your financial intelligence overview</div>
        <div class="section-copy">
            A snapshot of the dataset powering the current application.
            These are dataset statistics, not live banking metrics.
        </div>
        """,
        unsafe_allow_html=True,
    )

    total_records = len(df)
    total_columns = len(df.columns)
    scenario_count = safe_unique_count(df, "emi_scenario")
    eligibility_count = safe_unique_count(df, "emi_eligibility")
    missing_cells = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "TOTAL RECORDS",
            format_number(total_records),
            "Rows in the loaded dataset",
        )

    with c2:
        metric_card(
            "DATASET FEATURES",
            format_number(total_columns),
            "Columns available for analysis",
        )

    with c3:
        metric_card(
            "EMI SCENARIOS",
            format_number(scenario_count),
            "Unique values, when available",
        )

    with c4:
        metric_card(
            "ELIGIBILITY CLASSES",
            format_number(eligibility_count),
            "Unique labels, when available",
        )

    st.write("")

    # Data health
    st.markdown(
        """
        <div class="section-label">DATA QUALITY</div>
        <div class="section-heading">Dataset health check</div>
        <div class="section-copy">
            Basic quality indicators calculated directly from the loaded dataset.
        </div>
        """,
        unsafe_allow_html=True,
    )

    q1, q2, q3 = st.columns(3)

    with q1:
        st.metric("Missing cells", f"{missing_cells:,}")

    with q2:
        st.metric("Duplicate rows", f"{duplicate_rows:,}")

    with q3:
        completeness = (
            100 * (1 - missing_cells / (total_records * total_columns))
            if total_records > 0 and total_columns > 0
            else 0
        )
        st.metric("Cell completeness", f"{completeness:.1f}%")

    st.divider()

    # Model cards
    st.markdown(
        """
        <div class="section-label">PREDICTIVE ENGINE</div>
        <div class="section-heading">Machine learning models</div>
        <div class="section-copy">
            The model names below come from your existing model metadata.
            Successful loading does not by itself guarantee prediction accuracy.
        </div>
        """,
        unsafe_allow_html=True,
    )

    classification_name = feature_info.get(
        "best_classification_model", "Classification model"
    )
    regression_name = feature_info.get(
        "best_regression_model", "Regression model"
    )

    classification_classes = feature_info.get(
        "classification_classes", []
    )

    if isinstance(classification_classes, (list, tuple)):
        classes_text = ", ".join(map(str, classification_classes))
    else:
        classes_text = str(classification_classes)

    mc1, mc2 = st.columns(2)

    with mc1:
        st.markdown(
            f"""
            <div class="model-card">
                <div class="model-icon">◈</div><br>
                <span class="status-pill">LOADED</span>
                <div class="model-name">
                    {html.escape(str(classification_name))}
                </div>
                <div class="model-description">
                    <b>Classification engine</b><br>
                    Estimates the EMI eligibility category using the
                    features expected by your trained classification model.
                    <br><br>
                    <b>Configured classes:</b>
                    {html.escape(classes_text or "Not specified in metadata")}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with mc2:
        st.markdown(
            f"""
            <div class="model-card">
                <div class="model-icon">⌁</div><br>
                <span class="status-pill">LOADED</span>
                <div class="model-name">
                    {html.escape(str(regression_name))}
                </div>
                <div class="model-description">
                    <b>Regression engine</b><br>
                    Estimates a numerical monthly EMI amount from the
                    features expected by your trained regression model.
                    <br><br>
                    <b>Output:</b> Estimated maximum affordable monthly EMI,
                    subject to your model's training and target definition.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    st.info(
        "Use the dedicated prediction pages to generate predictions. "
        "This homepage does not independently validate loan eligibility "
        "or guarantee that a customer can afford a particular EMI."
    )

    st.divider()

    # Dataset explorer
    st.markdown(
        """
        <div class="section-label">DATA EXPLORER</div>
        <div class="section-heading">Explore the financial dataset</div>
        <div class="section-copy">
            Search records and inspect the data available to the platform.
        </div>
        """,
        unsafe_allow_html=True,
    )

    search_term = st.text_input(
        "Search records",
        placeholder="Enter a name, category, scenario, or value...",
        key="homepage_dataset_search",
    )

    row_limit = st.selectbox(
        "Rows to display",
        options=[10, 25, 50, 100],
        index=1,
        key="homepage_dataset_row_limit",
    )

    filtered_df = df

    if search_term.strip():
        try:
            search_mask = (
                df.astype(str)
                .apply(
                    lambda col: col.str.contains(
                        search_term.strip(),
                        case=False,
                        regex=False,
                        na=False,
                    )
                )
                .any(axis=1)
            )
            filtered_df = df.loc[search_mask]
        except Exception as exc:
            st.warning(f"Dataset search could not be completed: {exc}")

    st.caption(
        f"Showing {min(len(filtered_df), row_limit):,} of "
        f"{len(filtered_df):,} matching records."
    )

    st.dataframe(
        filtered_df.head(row_limit),
        width="stretch",
        hide_index=True,
    )

    with st.expander("View column information"):
        column_info = pd.DataFrame(
            {
                "Column": df.columns.astype(str),
                "Data type": df.dtypes.astype(str).values,
                "Missing values": df.isna().sum().values,
                "Unique values": df.nunique(dropna=True).values,
            }
        )

        st.dataframe(
            column_info,
            width="stretch",
            hide_index=True,
        )

    # Eligibility distribution
    if "emi_eligibility" in df.columns:
        st.write("")
        st.markdown(
            """
            <div class="section-label">ELIGIBILITY DISTRIBUTION</div>
            <div class="section-heading">Labels in the dataset</div>
            <div class="section-copy">
                This chart describes the loaded dataset's labels. It is
                not a prediction result or a measure of model accuracy.
            </div>
            """,
            unsafe_allow_html=True,
        )

        eligibility_counts = (
            df["emi_eligibility"]
            .fillna("Missing")
            .astype(str)
            .value_counts()
        )

        if not eligibility_counts.empty:
            chart_df = eligibility_counts.rename_axis(
                "Eligibility category"
            ).reset_index(name="Record count")

            st.bar_chart(
                chart_df.set_index("Eligibility category"),
                width="stretch",
            )

    st.divider()

    # Project workflow
    st.markdown(
        """
        <div class="section-label">HOW IT WORKS</div>
        <div class="section-heading">The machine learning workflow</div>
        <div class="section-copy">
            A high-level representation of the project's intended workflow.
        </div>
        """,
        unsafe_allow_html=True,
    )

    stages = [
        (
            "01",
            "Data ingestion",
            "Load the financial dataset and supporting project assets.",
        ),
        (
            "02",
            "Data preparation",
            "Clean records and prepare the features used by the models.",
        ),
        (
            "03",
            "Feature engineering",
            "Prepare input variables in the format expected by each model.",
        ),
        (
            "04",
            "Model inference",
            "Use classification and regression for their respective tasks.",
        ),
        (
            "05",
            "Model tracking",
            "Track experiments and model information through the project's MLflow setup.",
        ),
        (
            "06",
            "Prediction interface",
            "Present predictions through the application's dedicated pages.",
        ),
    ]

    for start in range(0, len(stages), 3):
        stage_cols = st.columns(3)

        for col, stage in zip(stage_cols, stages[start:start + 3]):
            number, title, description = stage

            with col:
                st.markdown(
                    f"""
                    <div class="pipeline-card">
                        <div class="pipeline-number">STAGE {number}</div>
                        <div class="pipeline-title">
                            {html.escape(title)}
                        </div>
                        <div class="pipeline-copy">
                            {html.escape(description)}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.write("")
    st.markdown(
        """
        <div class="footer">
            EMIPredict AI · Financial Risk Assessment & Decision Support<br>
            Machine learning outputs are estimates and should be reviewed
            alongside appropriate financial checks.
        </div>
        """,
        unsafe_allow_html=True,
    )

except Exception as exc:
    # This catches ordinary exceptions raised while rendering the dashboard.
    st.error("An unexpected error occurred while rendering the dashboard.")
    st.code(f"{type(exc).__name__}: {exc}", language="text")

    with st.expander("Full Python traceback", expanded=True):
        st.code(traceback.format_exc(), language="text")

    st.warning(
        "Copy the exception and traceback above. They should help identify "
        "the failing operation. This diagnostic block cannot catch every "
        "process-level failure."
    )
