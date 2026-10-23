
# ============================================================
# EMIPredict AI | Premium Financial Intelligence Dashboard
# Streamlit homepage | Hugging Face fallback | Safe initialization
# ============================================================

from pathlib import Path
import html
import traceback
import warnings

import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

HF_REPO_ID = "M-Prasad/EMIPredict_AI"

# Dataset fallback. Update this if your CSV is in another HF repository.
HF_DATASET_URL = (
    f"https://huggingface.co/{HF_REPO_ID}/resolve/main/"
    "emi_prediction_cleaned.csv?download=true"
)

LOCAL_DATA_CANDIDATES = [
    DATA_DIR / "emi_prediction_cleaned.csv",
    ROOT / "emi_prediction_cleaned.csv",
]

st.set_page_config(
    page_title="EMIPredict AI | Financial Intelligence",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. VISUAL DESIGN SYSTEM
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --bg: #080d19;
        --panel: #10192a;
        --border: rgba(148,163,184,.15);
        --text: #edf3ff;
        --muted: #94a3b8;
        --cyan: #5ee7f5;
        --blue: #7299ff;
        --green: #57e3a2;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background:
          radial-gradient(ellipse at 12% 0%,
            rgba(45,91,170,.18), transparent 36%),
          radial-gradient(ellipse at 100% 12%,
            rgba(20,184,166,.09), transparent 30%),
          var(--bg);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(8,13,25,.8);
    }

    [data-testid="stSidebar"] {
        background: #0a1120;
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.4rem;
    }

    [data-testid="stSidebar"] * {
        color: #dce7f8;
    }

    .block-container {
        max-width: 1550px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, h4 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: var(--text);
        letter-spacing: -.5px;
    }

    p, li, label {
        color: #c6d2e4;
    }

    .brand {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.4rem;
        font-weight: 700;
        letter-spacing: -.8px;
    }

    .brand-accent {
        color: var(--cyan);
    }

    .sidebar-tag {
        display: inline-block;
        margin-top: 9px;
        padding: 5px 10px;
        border-radius: 20px;
        border: 1px solid rgba(94,231,245,.24);
        background: rgba(94,231,245,.07);
        color: var(--cyan);
        font-size: .68rem;
        letter-spacing: 1.4px;
    }

    .hero {
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(117,157,255,.25);
        border-radius: 25px;
        padding: 36px;
        margin-bottom: 22px;
        background:
          radial-gradient(circle at 86% 12%,
            rgba(94,231,245,.18), transparent 28%),
          radial-gradient(circle at 70% 110%,
            rgba(114,153,255,.22), transparent 38%),
          linear-gradient(125deg,#101a2d 0%,#111d35 54%,#102b39 100%);
        box-shadow: 0 18px 55px rgba(0,0,0,.18);
    }

    .hero-eyebrow {
        color: var(--cyan);
        text-transform: uppercase;
        font-size: .72rem;
        font-weight: 700;
        letter-spacing: 1.8px;
        margin-bottom: 16px;
    }

    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(2rem,4vw,3.35rem);
        line-height: 1.08;
        letter-spacing: -1.7px;
        font-weight: 700;
        color: #f3f7ff;
        margin-bottom: 17px;
    }

    .hero-title span {
        color: var(--cyan);
    }

    .hero-description {
        color: #b4c3d9;
        font-size: .98rem;
        line-height: 1.8;
        max-width: 750px;
        margin-bottom: 23px;
    }

    .hero-pills {
        display: flex;
        flex-wrap: wrap;
        gap: 9px;
    }

    .hero-pill {
        padding: 8px 12px;
        border-radius: 9px;
        background: rgba(255,255,255,.055);
        border: 1px solid rgba(255,255,255,.1);
        color: #dce8fa;
        font-size: .78rem;
    }

    .section-heading {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.28rem;
        font-weight: 700;
        color: #edf4ff;
        margin: 28px 0 4px;
    }

    .section-subtitle {
        color: #899bb4;
        font-size: .85rem;
        margin-bottom: 17px;
        line-height: 1.6;
    }

    .metric-card {
        background: linear-gradient(145deg,#142034,#0d1626);
        border: 1px solid var(--border);
        border-radius: 17px;
        padding: 19px;
        min-height: 143px;
        box-shadow: 0 8px 25px rgba(0,0,0,.1);
    }

    .metric-top {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 8px;
        margin-bottom: 13px;
    }

    .metric-label {
        color: #98a9c1;
        font-size: .79rem;
        font-weight: 600;
    }

    .metric-icon {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 35px;
        height: 35px;
        border-radius: 11px;
        background: rgba(94,231,245,.09);
        border: 1px solid rgba(94,231,245,.13);
        font-size: 1rem;
    }

    .metric-value {
        font-family: 'Space Grotesk', sans-serif;
        color: #f4f8ff;
        font-weight: 700;
        font-size: clamp(1.4rem,2.3vw,2rem);
        letter-spacing: -.8px;
        overflow-wrap: anywhere;
    }

    .metric-caption {
        color: #8194ad;
        font-size: .74rem;
        margin-top: 7px;
    }

    .panel {
        background: rgba(16,25,42,.86);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 20px;
        min-height: 100%;
    }

    .panel-title {
        font-family: 'Space Grotesk', sans-serif;
        color: #edf4ff;
        font-weight: 700;
        font-size: 1rem;
        margin-bottom: 5px;
    }

    .panel-description {
        color: #899bb4;
        font-size: .79rem;
        line-height: 1.6;
        margin-bottom: 14px;
    }

    .model-card {
        background: linear-gradient(145deg,#132034,#0d1626);
        border: 1px solid var(--border);
        border-radius: 17px;
        padding: 21px;
        min-height: 220px;
    }

    .model-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 42px;
        height: 42px;
        border-radius: 13px;
        background: rgba(114,153,255,.12);
        font-size: 1.25rem;
        margin-bottom: 14px;
    }

    .model-name {
        color: #f0f5ff;
        font-size: 1.03rem;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .model-description {
        color: #8fa1ba;
        font-size: .82rem;
        line-height: 1.7;
    }

    .status-chip {
        display: inline-block;
        border-radius: 20px;
        padding: 5px 9px;
        color: #8ff0c0;
        background: rgba(87,227,162,.08);
        border: 1px solid rgba(87,227,162,.18);
        font-size: .7rem;
        margin-top: 12px;
    }

    .insight {
        border-left: 3px solid var(--cyan);
        border-radius: 0 12px 12px 0;
        padding: 14px 16px;
        background: rgba(94,231,245,.055);
        color: #c9d7e9;
        font-size: .84rem;
        line-height: 1.7;
        margin: 9px 0;
    }

    .footer {
        border-top: 1px solid var(--border);
        margin-top: 42px;
        padding-top: 19px;
        color: #71839d;
        font-size: .77rem;
        line-height: 1.8;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
    }

    div[data-testid="stPlotlyChart"] {
        border-radius: 13px;
    }

    div.stButton > button {
        border-radius: 10px;
        min-height: 42px;
        font-weight: 600;
    }

    div.stButton > button[kind="primary"] {
        background: linear-gradient(110deg,#5ee7f5,#7299ff);
        color: #07111e;
        border: none;
    }

    @media(max-width:768px) {
        .hero { padding: 24px 20px; }
        .block-container { padding-left: 1rem; padding-right: 1rem; }
        .hero-title { letter-spacing: -.8px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. SAFE DATA LOADING
# ============================================================

@st.cache_data(show_spinner=False, ttl=3600)
def load_dashboard_data():
    errors = []

    # First use the project's existing data utility.
    try:
        from utils.data_utils import load_data

        result = load_data()

        if isinstance(result, pd.DataFrame) and not result.empty:
            return result, "Project data utility"

        errors.append("Project data utility returned an empty dataset.")

    except Exception as exc:
        errors.append(f"Project data utility: {exc}")

    # Try local CSV files.
    for path in LOCAL_DATA_CANDIDATES:
        try:
            if path.is_file():
                result = pd.read_csv(path)

                if not result.empty:
                    return result, f"Local file: {path.name}"

                errors.append(f"{path.name} is empty.")

        except Exception as exc:
            errors.append(f"{path.name}: {exc}")

    # Fall back to Hugging Face.
    try:
        result = pd.read_csv(HF_DATASET_URL)

        if not result.empty:
            return result, "Hugging Face"

        errors.append("Hugging Face CSV is empty.")

    except Exception as exc:
        errors.append(f"Hugging Face dataset: {exc}")

    raise RuntimeError(
        "The EMI dataset could not be loaded.\n\n"
        + "\n".join(errors[-5:])
    )


@st.cache_data(show_spinner=False, ttl=3600)
def load_model_metadata():
    # Only load lightweight metadata on the homepage.
    # Do not load large model objects here.
    try:
        from utils.model_utils import load_feature_info

        info = load_feature_info()
        if isinstance(info, dict):
            return info

    except Exception:
        pass

    return {}


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def safe_string(value, default="Not available"):
    if value is None:
        return default

    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass

    value = str(value).strip()

    if not value or value.lower() in {"nan", "none", "null"}:
        return default

    return value


def find_column(df, candidates):
    mapping = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in candidates:
        if candidate.lower() in mapping:
            return mapping[candidate.lower()]

    return None


def numeric_series(df, column):
    if column is None or column not in df.columns:
        return pd.Series(dtype="float64")

    return pd.to_numeric(df[column], errors="coerce").dropna()


def rupees(value):
    if pd.isna(value):
        return "N/A"

    return "₹" + f"{float(value):,.0f}"


def compact_number(value):
    value = float(value)

    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"

    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"

    return f"{value:,.0f}"


def metric_card(label, value, caption, icon):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-top">
                <span class="metric-label">{html.escape(str(label))}</span>
                <span class="metric-icon">{html.escape(str(icon))}</span>
            </div>
            <div class="metric-value">{html.escape(str(value))}</div>
            <div class="metric-caption">{html.escape(str(caption))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_heading(title, subtitle=""):
    st.markdown(
        f"""
        <div class="section-heading">{html.escape(title)}</div>
        <div class="section-subtitle">{html.escape(subtitle)}</div>
        """,
        unsafe_allow_html=True,
    )


def plotly_theme(fig, height=350):
    fig.update_layout(
        height=height,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "DM Sans, sans-serif", "color": "#c7d5e8"},
        margin={"l": 12, "r": 12, "t": 30, "b": 12},
        hoverlabel={
            "bgcolor": "#10192a",
            "font_color": "#edf3ff",
            "bordercolor": "#334155",
        },
    )

    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor="rgba(148,163,184,.15)",
    )
    fig.update_yaxes(
        gridcolor="rgba(148,163,184,.10)",
        zeroline=False,
        linecolor="rgba(148,163,184,.15)",
    )

    return fig


def build_data_types(df):
    """
    Avoid select_dtypes(include=['object']) so Pandas does not emit
    the warning about future string-dtype selection behaviour.
    """
    numeric = []
    categorical = []

    for column in df.columns:
        dtype = df[column].dtype

        if pd.api.types.is_numeric_dtype(dtype):
            numeric.append(column)

        is_categorical = (
            pd.api.types.is_object_dtype(dtype)
            or pd.api.types.is_string_dtype(dtype)
            or isinstance(dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(dtype)
        )

        if is_categorical:
            categorical.append(column)

    return numeric, categorical


# ============================================================
# 5. SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div class="brand">EMI<span class="brand-accent">Predict</span></div>
        <div style="color:#8fa1ba;font-size:.78rem;margin-top:5px;">
            Intelligent Financial Risk Platform
        </div>
        <div class="sidebar-tag">AI · ML · MLFLOW</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("#### Workspace")

    st.caption(
        "Use the Streamlit page navigation to open the prediction, "
        "analytics, MLflow, and financial data management modules."
    )

    st.markdown("---")
    st.markdown("#### Platform modules")

    st.markdown(
        """
        <div style="line-height:2.1;font-size:.84rem;color:#b5c5db;">
        ◈ &nbsp; Financial overview<br>
        ◈ &nbsp; EMI eligibility classification<br>
        ◈ &nbsp; Maximum EMI regression<br>
        ◈ &nbsp; Model performance & tracking<br>
        ◈ &nbsp; Financial data management
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("#### Data source")
    st.caption("Local files with Hugging Face fallback")

    if st.button(
        "↻ Refresh dashboard data",
        width="stretch",
        key="refresh_dashboard_data",
    ):
        load_dashboard_data.clear()
        load_model_metadata.clear()
        st.rerun()

    st.markdown(
        """
        <div style="margin-top:25px;color:#647892;font-size:.72rem;
                    line-height:1.7;">
        EMIPredict AI<br>
        Financial Intelligence Workspace
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 6. INITIALIZE APP WITH TRACEBACK REPORTING
# ============================================================

try:
    with st.spinner("Preparing your financial intelligence dashboard..."):
        df, data_source = load_dashboard_data()
        feature_info = load_model_metadata()

    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("The loaded dataset is empty or invalid.")

    # Prevent an empty or malformed dataframe from breaking analytics.
    df.columns = [str(c).strip() for c in df.columns]

    numeric_cols, categorical_cols = build_data_types(df)

except Exception:
    st.error("Dashboard initialization failed.")

    st.markdown(
        """
        Please check:
        - `utils/data_utils.py` and `utils/model_utils.py` exist.
        - The CSV filename and Hugging Face repository are correct.
        - The deployed app has network access to Hugging Face.
        - Your dependencies are compatible with the application code.
        """
    )

    with st.expander("Full technical traceback", expanded=True):
        st.code(traceback.format_exc(), language="text")

    st.stop()


# ============================================================
# 7. DETECT DATASET COLUMNS
# ============================================================

row_count = len(df)
column_count = len(df.columns)
missing_cells = int(df.isna().sum().sum())

scenario_col = find_column(
    df,
    ["emi_scenario", "loan_type", "loan_category", "scenario", "loan_scenario"],
)

eligibility_col = find_column(
    df,
    ["emi_eligibility", "eligibility", "emi_eligible", "loan_eligibility",
     "eligible", "target"],
)

emi_col = find_column(
    df,
    ["max_affordable_emi", "maximum_emi", "max_emi", "emi_amount",
     "monthly_emi", "emi", "predicted_emi"],
)

salary_col = find_column(
    df,
    ["monthly_salary", "monthly_income", "salary", "income",
     "monthly_income_inr"],
)

age_col = find_column(
    df,
    ["age", "applicant_age", "customer_age"],
)

affordability_col = find_column(
    df,
    ["affordability_ratio", "emi_affordability_ratio"],
)

unique_scenarios = (
    int(df[scenario_col].nunique(dropna=True))
    if scenario_col else 0
)

missing_percentage = (
    missing_cells / (row_count * column_count) * 100
    if row_count and column_count else 0
)


# ============================================================
# 8. HERO
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-eyebrow">
            ✦ FINANCIAL INTELLIGENCE · DASHBOARD
        </div>
        <div class="hero-title">
            Smarter lending.<br>
            <span>Clearer decisions.</span>
        </div>
        <div class="hero-description">
            An integrated machine-learning platform for EMI eligibility
            assessment, affordable installment estimation, financial
            dataset exploration, and model performance monitoring.
            Turn financial data into insights through one workspace.
        </div>
        <div class="hero-pills">
            <span class="hero-pill">◈ Classification</span>
            <span class="hero-pill">↗ Regression</span>
            <span class="hero-pill">◎ MLflow tracking</span>
            <span class="hero-pill">▦ Data analytics</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

meta_left, meta_right = st.columns([3, 1])

with meta_left:
    st.caption(
        f"DATA SOURCE  /  {data_source}  ·  "
        f"{pd.Timestamp.now().strftime('%d %b %Y')}"
    )

with meta_right:
    st.markdown(
        """
        <div style="text-align:right;color:#8fa1ba;font-size:.77rem;">
        <span style="color:#57e3a2;">●</span> Dashboard initialized
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 9. KPI CARDS
# ============================================================

section_heading(
    "Platform at a glance",
    "A high-level view of the financial dataset currently available.",
)

k1, k2, k3, k4 = st.columns(4, gap="medium")

with k1:
    metric_card(
        "Dataset records",
        compact_number(row_count),
        "Rows available for exploration",
        "▦",
    )

with k2:
    metric_card(
        "Data features",
        str(column_count),
        f"{len(numeric_cols)} numeric · {len(categorical_cols)} categorical",
        "⌘",
    )

with k3:
    metric_card(
        "Loan scenarios",
        str(unique_scenarios) if scenario_col else "N/A",
        "Distinct scenarios identified",
        "◈",
    )

with k4:
    metric_card(
        "Missing values",
        compact_number(missing_cells),
        "Missing cells across the dataset",
        "◎",
    )


# ============================================================
# 10. DISTRIBUTION CHARTS
# ============================================================

section_heading(
    "Financial overview",
    "Explore loan scenarios and eligibility outcomes in the loaded dataset.",
)

left_chart, right_chart = st.columns([1.12, 1], gap="large")

with left_chart:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-title">Loan scenario composition</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="panel-description">Most frequently occurring '
        'financing categories.</div>',
        unsafe_allow_html=True,
    )

    if scenario_col:
        scenario_counts = (
            df[scenario_col]
            .fillna("Unknown")
            .astype(str)
            .value_counts()
            .head(12)
            .rename_axis("Scenario")
            .reset_index(name="Records")
        )

        if PLOTLY_AVAILABLE:
            fig = px.bar(
                scenario_counts.sort_values("Records"),
                x="Records",
                y="Scenario",
                orientation="h",
                text="Records",
                color="Records",
                color_continuous_scale=["#24416b", "#5ee7f5"],
            )
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_layout(
                coloraxis_showscale=False,
                xaxis_title="Number of records",
                yaxis_title="",
            )
            st.plotly_chart(
                plotly_theme(fig, 350),
                width="stretch",
                config={"displayModeBar": False},
            )
        else:
            st.bar_chart(
                scenario_counts.set_index("Scenario"),
                horizontal=True,
                width="stretch",
            )
    else:
        st.info(
            "No recognized loan scenario column was found. "
            "Inspect the dataset explorer to verify column names."
        )

    st.markdown("</div>", unsafe_allow_html=True)


with right_chart:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-title">Eligibility distribution</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="panel-description">Distribution of target labels '
        'in the existing dataset, not live predictions.</div>',
        unsafe_allow_html=True,
    )

    if eligibility_col:
        eligibility_counts = (
            df[eligibility_col]
            .fillna("Unknown")
            .astype(str)
            .value_counts()
            .rename_axis("Eligibility")
            .reset_index(name="Records")
        )

        if PLOTLY_AVAILABLE:
            fig = px.pie(
                eligibility_counts,
                names="Eligibility",
                values="Records",
                hole=0.67,
                color_discrete_sequence=[
                    "#5ee7f5", "#7299ff", "#57e3a2",
                    "#b69aff", "#f6bd60", "#64748b",
                ],
            )
            fig.update_traces(
                textposition="inside",
                textinfo="percent",
                marker={"line": {"color": "#10192a", "width": 3}},
            )
            fig.update_layout(
                showlegend=True,
                legend={"orientation": "h", "y": -0.15},
            )
            st.plotly_chart(
                plotly_theme(fig, 350),
                width="stretch",
                config={"displayModeBar": False},
            )
        else:
            st.bar_chart(
                eligibility_counts.set_index("Eligibility"),
                width="stretch",
            )

        st.caption(
            f"{df[eligibility_col].nunique(dropna=True)} distinct "
            "eligibility labels detected."
        )
    else:
        st.info(
            "No recognized eligibility target column was found."
        )

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# 11. FINANCIAL INDICATORS
# ============================================================

section_heading(
    "Financial indicators",
    "Descriptive statistics from the detected financial columns.",
)

median_salary = (
    numeric_series(df, salary_col).median()
    if salary_col else float("nan")
)

median_emi = (
    numeric_series(df, emi_col).median()
    if emi_col else float("nan")
)

a, b, c = st.columns(3, gap="medium")

with a:
    metric_card(
        "Median monthly salary",
        rupees(median_salary),
        "Median of the detected income column",
        "₹",
    )

with b:
    metric_card(
        "Median EMI-related value",
        rupees(median_emi),
        "Median of the detected EMI column",
        "↗",
    )

with c:
    metric_card(
        "Data completeness",
        f"{100 - missing_percentage:.1f}%",
        "Percentage of non-missing cells",
        "✓",
    )


# ============================================================
# 12. AFFORDABILITY ANALYSIS
# ============================================================

section_heading(
    "Affordability analysis",
    "Inspect relationships between income and EMI-related values "
    "when the required fields are available.",
)

if salary_col and emi_col:
    chart_df = df[[salary_col, emi_col]].copy()

    chart_df[salary_col] = pd.to_numeric(
        chart_df[salary_col], errors="coerce"
    )
    chart_df[emi_col] = pd.to_numeric(
        chart_df[emi_col], errors="coerce"
    )

    chart_df = chart_df.dropna()
    chart_df = chart_df[
        (chart_df[salary_col] >= 0)
        & (chart_df[emi_col] >= 0)
    ]

    if len(chart_df) > 4000:
        chart_df = chart_df.sample(n=4000, random_state=42)

    if not chart_df.empty:
        if PLOTLY_AVAILABLE:
            fig = px.scatter(
                chart_df,
                x=salary_col,
                y=emi_col,
                opacity=0.55,
                render_mode="webgl",
                color_discrete_sequence=["#5ee7f5"],
                labels={
                    salary_col: "Monthly salary",
                    emi_col: "EMI-related value",
                },
            )
            fig.update_traces(marker={"size": 6})
            fig.update_layout(
                xaxis_title="Monthly salary",
                yaxis_title="EMI-related value",
            )
            st.plotly_chart(
                plotly_theme(fig, 400),
                width="stretch",
                config={"displayModeBar": False},
            )
        else:
            st.scatter_chart(
                chart_df,
                x=salary_col,
                y=emi_col,
                width="stretch",
            )

        st.caption(
            "This chart summarizes observed data. It does not establish "
            "causation or guarantee an individual's affordability."
        )
    else:
        st.info("No usable numeric rows are available for this chart.")

elif affordability_col:
    ratio = numeric_series(df, affordability_col)

    if not ratio.empty:
        if PLOTLY_AVAILABLE:
            fig = px.histogram(
                pd.DataFrame({"Affordability ratio": ratio}),
                x="Affordability ratio",
                nbins=35,
                color_discrete_sequence=["#7299ff"],
            )
            st.plotly_chart(
                plotly_theme(fig, 360),
                width="stretch",
                config={"displayModeBar": False},
            )
        else:
            st.bar_chart(
                ratio.value_counts(bins=20).sort_index(),
                width="stretch",
            )
    else:
        st.info("No usable affordability ratio values were found.")
else:
    st.info(
        "This chart requires recognized salary and EMI columns, "
        "or an affordability-ratio column. Review the dataset explorer "
        "to confirm the field names."
    )


# ============================================================
# 13. MACHINE LEARNING MODULES
# ============================================================

section_heading(
    "Machine learning architecture",
    "A unified interface for classification, regression, and monitoring.",
)

classification_name = safe_string(
    feature_info.get("best_classification_model")
    if isinstance(feature_info, dict) else None,
    "Configured in model utilities",
)

regression_name = safe_string(
    feature_info.get("best_regression_model")
    if isinstance(feature_info, dict) else None,
    "Configured in model utilities",
)

m1, m2, m3 = st.columns(3, gap="medium")

with m1:
    st.markdown(
        """
        <div class="model-card">
            <div class="model-icon">◈</div>
            <div class="model-name">Eligibility classification</div>
            <div class="model-description">
                Uses applicant and financial features to estimate the
                eligibility class from the trained classification model.
            </div>
            <div class="status-chip">Classification pipeline</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m2:
    st.markdown(
        """
        <div class="model-card">
            <div class="model-icon">↗</div>
            <div class="model-name">EMI regression</div>
            <div class="model-description">
                Estimates an EMI-related numerical outcome using the
                trained regression pipeline and its expected features.
            </div>
            <div class="status-chip">Regression pipeline</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m3:
    st.markdown(
        """
        <div class="model-card">
            <div class="model-icon">◎</div>
            <div class="model-name">MLflow monitoring</div>
            <div class="model-description">
                Experiment tracking, logged metrics, and model-related
                information are handled by the separate monitoring page.
            </div>
            <div class="status-chip">Tracking module</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with st.expander("Model configuration details"):
    st.markdown("**Classification model configured in metadata**")
    st.code(classification_name)

    st.markdown("**Regression model configured in metadata**")
    st.code(regression_name)

    st.caption(
        "The homepage intentionally does not load large model artifacts. "
        "Prediction pages should load their models when inference is needed."
    )

    if isinstance(feature_info, dict) and feature_info:
        st.markdown("**Available model metadata**")
        st.json(feature_info)
    else:
        st.caption(
            "No metadata was returned by load_feature_info(). "
            "Check the model utility if you expect metadata to be available."
        )


# ============================================================
# 14. DATASET EXPLORER
# ============================================================

section_heading(
    "Dataset explorer",
    "Search records, inspect feature statistics, and review data quality.",
)

with st.container(border=True):
    tab_preview, tab_profile, tab_quality = st.tabs(
        ["▦ Data preview", "◉ Statistical profile", "⌁ Data quality"]
    )

    with tab_preview:
        search_col, rows_col = st.columns([3, 1])

        with search_col:
            search_term = st.text_input(
                "Search records",
                placeholder="Search by scenario, eligibility, or any value...",
                key="dashboard_search",
            )

        with rows_col:
            preview_rows = st.selectbox(
                "Rows to display",
                [10, 25, 50, 100],
                index=1,
                key="dashboard_preview_rows",
            )

        filtered_df = df

        if search_term.strip():
            term = search_term.strip()

            # Use string conversion only for filtering, not for all analytics.
            mask = df.astype("string").apply(
                lambda column: column.str.contains(
                    term,
                    case=False,
                    na=False,
                    regex=False,
                )
            ).any(axis=1)

            filtered_df = df.loc[mask]

        st.caption(
            f"Showing {min(preview_rows, len(filtered_df)):,} of "
            f"{len(filtered_df):,} matching records."
        )

        st.dataframe(
            filtered_df.head(preview_rows),
            width="stretch",
            height=360,
            hide_index=True,
        )

        csv_data = filtered_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇ Download filtered dataset as CSV",
            data=csv_data,
            file_name="emipredict_filtered_data.csv",
            mime="text/csv",
            width="stretch",
        )

    with tab_profile:
        st.markdown("#### Numeric summary")

        if numeric_cols:
            profile = df[numeric_cols].describe().T
            profile.index.name = "Feature"

            st.dataframe(
                profile.round(3),
                width="stretch",
                height=350,
            )
        else:
            st.info("No numeric columns were detected.")

        st.markdown("#### Dataset structure")

        structure = pd.DataFrame(
            {
                "Column": [str(c) for c in df.columns],
                "Data type": [str(t) for t in df.dtypes],
                "Non-null values": [
                    int(df[c].notna().sum()) for c in df.columns
                ],
                "Unique values": [
                    int(df[c].nunique(dropna=True)) for c in df.columns
                ],
            }
        )

        st.dataframe(
            structure,
            width="stretch",
            hide_index=True,
        )

    with tab_quality:
        missing_by_column = df.isna().sum().sort_values(ascending=False)
        missing_by_column = missing_by_column[missing_by_column > 0]

        if missing_by_column.empty:
            st.success("No missing values were found in the loaded dataset.")
        else:
            missing_df = missing_by_column.rename(
                "Missing values"
            ).reset_index()

            missing_df.columns = ["Column", "Missing values"]

            missing_df["Missing percentage"] = (
                missing_df["Missing values"] / max(row_count, 1) * 100
            ).round(2)

            if PLOTLY_AVAILABLE:
                fig = px.bar(
                    missing_df.head(20).sort_values("Missing values"),
                    x="Missing values",
                    y="Column",
                    orientation="h",
                    color="Missing percentage",
                    color_continuous_scale=["#7299ff", "#f6bd60"],
                )
                fig.update_layout(coloraxis_showscale=False)

                st.plotly_chart(
                    plotly_theme(fig, 380),
                    width="stretch",
                    config={"displayModeBar": False},
                )

            st.dataframe(
                missing_df,
                width="stretch",
                hide_index=True,
            )


# ============================================================
# 15. AUTOMATIC INSIGHTS
# ============================================================

section_heading(
    "Quick insights",
    "Automatically generated observations from the detected columns.",
)

insight_left, insight_right = st.columns(2, gap="large")

with insight_left:
    if scenario_col:
        counts = df[scenario_col].value_counts(dropna=True)

        if not counts.empty:
            top_scenario = str(counts.index[0])
            top_count = int(counts.iloc[0])
            top_pct = top_count / max(row_count, 1) * 100

            st.markdown(
                f"""
                <div class="insight">
                    <b>Most frequent loan scenario</b><br>
                    {html.escape(top_scenario)} represents
                    <b>{top_pct:.1f}%</b> of the dataset
                    ({top_count:,} records).
                </div>
                """,
                unsafe_allow_html=True,
            )

    if salary_col:
        salaries = numeric_series(df, salary_col)

        if not salaries.empty:
            st.markdown(
                f"""
                <div class="insight">
                    <b>Income profile</b><br>
                    Median detected salary:
                    <b>{rupees(salaries.median())}</b>.
                    Mean detected salary:
                    <b>{rupees(salaries.mean())}</b>.
                </div>
                """,
                unsafe_allow_html=True,
            )

with insight_right:
    if eligibility_col:
        counts = df[eligibility_col].value_counts(dropna=True)

        if not counts.empty:
            dominant_label = str(counts.index[0])
            dominant_pct = counts.iloc[0] / max(row_count, 1) * 100

            st.markdown(
                f"""
                <div class="insight">
                    <b>Most frequent eligibility label</b><br>
                    <b>{html.escape(dominant_label)}</b> accounts for
                    {dominant_pct:.1f}% of non-missing labels.
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        f"""
        <div class="insight">
            <b>Data quality</b><br>
            {missing_cells:,} missing cells were detected across
            {row_count:,} rows and {column_count:,} columns.
            See the Data Quality tab for column-level details.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 16. FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        <div style="font-family:'Space Grotesk',sans-serif;
                    color:#d9e6fa;font-size:.92rem;font-weight:700;">
            EMIPredict <span style="color:#5ee7f5;">AI</span>
        </div>
        <div>
            Intelligent Financial Risk Assessment Platform
            · Machine Learning · Streamlit
        </div>
        <div style="margin-top:7px;">
            Dataset summaries are descriptive. Predictions depend on the
            trained model, its input schema, and data quality. Results
            should not be treated as guaranteed loan approvals or
            professional financial advice.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
