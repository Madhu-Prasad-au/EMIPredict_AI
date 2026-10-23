
# ============================================================
# EMIPredict AI | Intelligent Financial Risk Assessment Platform
# Premium Streamlit Dashboard
# ============================================================

from pathlib import Path
import html
import os
import warnings

import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore", category=FutureWarning)

# Optional visualization library
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# ------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

HF_REPO_ID = "M-Prasad/EMIPredict_AI"
HF_DATASET_URL = (
    "https://huggingface.co/datasets/"
    f"{HF_REPO_ID}/resolve/main/emi_prediction_cleaned.csv"
    "?download=true"
)

# Also try the model repository URL in case the dataset is hosted
# as a regular file in the same Hugging Face repository.
HF_MODEL_REPO_DATA_URL = (
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


# ------------------------------------------------------------
# 2. PREMIUM DESIGN SYSTEM
# ------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url(
      'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap'
    );

    :root {
        --bg: #080d19;
        --panel: #10192a;
        --panel-2: #121f33;
        --border: rgba(148, 163, 184, 0.15);
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
                rgba(45, 91, 170, 0.17), transparent 36%),
            radial-gradient(ellipse at 100% 12%,
                rgba(20, 184, 166, 0.09), transparent 30%),
            var(--bg);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(8, 13, 25, 0.78);
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
        padding-top: 1.6rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, h4 {
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: -0.5px;
        color: var(--text);
    }

    p, li, label {
        color: #c6d2e4;
    }

    .brand {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 1.35rem;
        letter-spacing: -0.7px;
    }

    .brand-accent {
        color: var(--cyan);
    }

    .sidebar-tag {
        display: inline-block;
        font-size: 0.68rem;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: var(--cyan);
        border: 1px solid rgba(94, 231, 245, 0.24);
        background: rgba(94, 231, 245, 0.07);
        border-radius: 20px;
        padding: 5px 10px;
        margin-top: 8px;
    }

    .hero {
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(117, 157, 255, 0.25);
        border-radius: 25px;
        padding: 35px 36px;
        background:
            radial-gradient(circle at 86% 12%,
                rgba(94, 231, 245, 0.18), transparent 28%),
            radial-gradient(circle at 70% 110%,
                rgba(114, 153, 255, 0.22), transparent 38%),
            linear-gradient(125deg, #101a2d 0%, #111d35 54%, #102b39 100%);
        box-shadow: 0 18px 55px rgba(0, 0, 0, 0.18);
        margin-bottom: 22px;
    }

    .hero-eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        color: var(--cyan);
        text-transform: uppercase;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 1.8px;
        margin-bottom: 15px;
    }

    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(2rem, 4vw, 3.3rem);
        line-height: 1.08;
        letter-spacing: -1.7px;
        font-weight: 700;
        max-width: 780px;
        color: #f3f7ff;
        margin: 0 0 16px 0;
    }

    .hero-title span {
        color: var(--cyan);
    }

    .hero-description {
        color: #b4c3d9;
        font-size: 0.98rem;
        line-height: 1.8;
        max-width: 710px;
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
        background: rgba(255, 255, 255, 0.055);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #dce8fa;
        font-size: 0.78rem;
    }

    .section-heading {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.25rem;
        font-weight: 700;
        color: #edf4ff;
        margin: 26px 0 4px 0;
    }

    .section-subtitle {
        color: #899bb4;
        font-size: 0.85rem;
        margin-bottom: 17px;
    }

    .metric-card {
        background: linear-gradient(
            145deg, rgba(20, 32, 52, 0.97), rgba(13, 22, 38, 0.97)
        );
        border: 1px solid var(--border);
        border-radius: 17px;
        padding: 19px 19px 17px 19px;
        min-height: 142px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.10);
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
        font-size: 0.79rem;
        font-weight: 600;
    }

    .metric-icon {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 35px;
        height: 35px;
        border-radius: 11px;
        background: rgba(94, 231, 245, 0.09);
        border: 1px solid rgba(94, 231, 245, 0.13);
        font-size: 1rem;
    }

    .metric-value {
        font-family: 'Space Grotesk', sans-serif;
        color: #f4f8ff;
        font-weight: 700;
        font-size: clamp(1.4rem, 2.3vw, 2rem);
        letter-spacing: -0.8px;
        overflow-wrap: anywhere;
    }

    .metric-caption {
        color: #8194ad;
        font-size: 0.74rem;
        margin-top: 7px;
    }

    .content-panel {
        background: rgba(16, 25, 42, 0.86);
        border: 1px solid var(--border);
        border-radius: 19px;
        padding: 21px;
        height: 100%;
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
        font-size: 0.79rem;
        line-height: 1.6;
        margin-bottom: 16px;
    }

    .model-card {
        background: linear-gradient(
            145deg, rgba(19, 32, 52, 0.98), rgba(13, 22, 38, 0.98)
        );
        border: 1px solid var(--border);
        border-radius: 17px;
        padding: 21px;
        height: 100%;
    }

    .model-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 42px;
        height: 42px;
        border-radius: 13px;
        background: rgba(114, 153, 255, 0.12);
        font-size: 1.25rem;
        margin-bottom: 14px;
    }

    .model-name {
        color: #f0f5ff;
        font-size: 1.03rem;
        font-weight: 700;
        margin-bottom: 8px;
        overflow-wrap: anywhere;
    }

    .model-description {
        color: #8fa1ba;
        font-size: 0.82rem;
        line-height: 1.7;
        min-height: 65px;
    }

    .status-chip {
        display: inline-block;
        border-radius: 20px;
        padding: 5px 9px;
        color: #8ff0c0;
        background: rgba(87, 227, 162, 0.08);
        border: 1px solid rgba(87, 227, 162, 0.18);
        font-size: 0.70rem;
        margin-top: 12px;
    }

    .insight {
        border-left: 3px solid var(--cyan);
        border-radius: 0 12px 12px 0;
        padding: 14px 16px;
        background: rgba(94, 231, 245, 0.055);
        color: #c9d7e9;
        font-size: 0.84rem;
        line-height: 1.7;
        margin: 9px 0;
    }

    .footer {
        border-top: 1px solid var(--border);
        margin-top: 42px;
        padding-top: 19px;
        color: #71839d;
        font-size: 0.77rem;
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
        background: linear-gradient(110deg, #5ee7f5, #7299ff);
        color: #07111e;
        border: none;
    }

    div.stButton > button[kind="primary"]:hover {
        border: none;
        filter: brightness(1.08);
        color: #07111e;
    }

    [data-testid="stMetric"] {
        background: rgba(17, 29, 48, 0.5);
        border: 1px solid var(--border);
        border-radius: 13px;
        padding: 14px;
    }

    @media (max-width: 768px) {
        .hero {
            padding: 25px 21px;
        }

        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .hero-title {
            letter-spacing: -0.8px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# 3. DATA LOADING
# ------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=3600)
def load_dashboard_data():
    """
    Load the dataset in this order:
    1. Existing project utility.
    2. Local CSV candidates.
    3. Hugging Face direct-download URLs.

    This function deliberately does not load the ML models.
    """

    errors = []

    # Use the project's existing utility first.
    try:
        from utils.data_utils import load_data

        df = load_data()

        if isinstance(df, pd.DataFrame) and not df.empty:
            return df, "Project data utility"
    except Exception as exc:
        errors.append(f"Project utility: {exc}")

    # Try local dataset copies.
    for path in LOCAL_DATA_CANDIDATES:
        try:
            if path.exists():
                df = pd.read_csv(path)
                if not df.empty:
                    return df, f"Local file: {path.name}"
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")

    # Try Hugging Face dataset and model repository URLs.
    for url in [HF_DATASET_URL, HF_MODEL_REPO_DATA_URL]:
        try:
            df = pd.read_csv(url)

            if not df.empty:
                return df, "Hugging Face"

        except Exception as exc:
            errors.append(f"Hugging Face source: {exc}")

    raise RuntimeError(
        "The EMI dataset could not be loaded from the project utility, "
        "local files, or Hugging Face.\n\n"
        + "\n".join(errors[-4:])
    )


@st.cache_data(show_spinner=False, ttl=3600)
def load_model_metadata():
    """
    Metadata is lightweight. Avoid loading the large classification and
    regression model objects on the homepage.
    """
    try:
        from utils.model_utils import load_feature_info

        info = load_feature_info()
        if isinstance(info, dict):
            return info
    except Exception:
        pass

    return {}


def safe_string(value, default="Not available"):
    if value is None:
        return default

    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass

    result = str(value).strip()

    if not result or result.lower() in {"nan", "none", "null"}:
        return default

    return result


def find_column(df, candidates):
    """Find a column without assuming exact capitalization."""
    mapping = {str(c).strip().lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in mapping:
            return mapping[candidate.lower()]

    return None


def numeric_series(df, column):
    if column is None or column not in df.columns:
        return pd.Series(dtype="float64")

    return pd.to_numeric(df[column], errors="coerce").dropna()


def rupees(value, decimals=0):
    if pd.isna(value):
        return "N/A"

    return "₹" + f"{float(value):,.{decimals}f}"


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
                <span class="metric-label">{html.escape(label)}</span>
                <span class="metric-icon">{icon}</span>
            </div>
            <div class="metric-value">{html.escape(str(value))}</div>
            <div class="metric-caption">{html.escape(caption)}</div>
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
        font=dict(
            family="DM Sans, sans-serif",
            color="#c7d5e8",
            size=12,
        ),
        margin=dict(l=12, r=12, t=30, b=12),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
        ),
        hoverlabel=dict(
            bgcolor="#10192a",
            font_color="#edf3ff",
            bordercolor="#334155",
        ),
    )

    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor="rgba(148,163,184,0.15)",
    )
    fig.update_yaxes(
        gridcolor="rgba(148,163,184,0.10)",
        zeroline=False,
        linecolor="rgba(148,163,184,0.15)",
    )

    return fig


# ------------------------------------------------------------
# 4. SIDEBAR BRANDING
# ------------------------------------------------------------

with st.sidebar:
    st.markdown(
        """
        <div class="brand">EMI<span class="brand-accent">Predict</span></div>
        <div style="color:#8fa1ba;font-size:0.78rem;margin-top:5px;">
            Intelligent Financial Risk Platform
        </div>
        <div class="sidebar-tag">AI · ML · MLFLOW</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown("#### Workspace")
    st.caption(
        "Use the application pages in Streamlit's navigation menu "
        "to open the prediction tools, analytics, model dashboard, "
        "and financial data management modules."
    )

    st.markdown("---")

    st.markdown("#### Platform modules")

    st.markdown(
        """
        <div style="line-height:2.15;font-size:0.85rem;color:#b5c5db;">
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
    st.caption("Local project files / Hugging Face fallback")

    if st.button("↻ Refresh dashboard data", use_container_width=True):
        load_dashboard_data.clear()
        load_model_metadata.clear()
        st.rerun()

    st.markdown(
        """
        <div style="margin-top:25px;color:#647892;font-size:0.72rem;
                    line-height:1.7;">
        EMIPredict AI<br>
        Financial Intelligence Workspace<br>
        <span style="color:#5ee7f5;">●</span> Dashboard interface
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------
# 5. LOAD DATA
# ------------------------------------------------------------

try:
    with st.spinner("Preparing your financial intelligence dashboard..."):
        df, data_source = load_dashboard_data()
        feature_info = load_model_metadata()

except Exception as exc:
    st.error("Unable to initialize the dashboard dataset.")

    st.markdown(
        """
        **Check the following:**
        - `utils/data_utils.py` is present in your repository.
        - The dataset is available locally or in the Hugging Face repository.
        - The Hugging Face repository and filename are correct.
        - Your deployment has access to the internet.
        """
    )

    with st.expander("Technical details"):
        st.code(str(exc))

    st.stop()


# ------------------------------------------------------------
# 6. COLUMN DETECTION
# ------------------------------------------------------------

row_count = len(df)
column_count = len(df.columns)
missing_cells = int(df.isna().sum().sum())

eligibility_col = find_column(
    df,
    [
        "emi_eligibility",
        "eligibility",
        "emi_eligible",
        "loan_eligibility",
        "eligible",
        "target",
    ],
)

scenario_col = find_column(
    df,
    [
        "emi_scenario",
        "loan_type",
        "loan_category",
        "scenario",
        "loan_scenario",
    ],
)

emi_col = find_column(
    df,
    [
        "max_affordable_emi",
        "maximum_emi",
        "max_emi",
        "emi_amount",
        "monthly_emi",
        "emi",
        "predicted_emi",
    ],
)

salary_col = find_column(
    df,
    [
        "monthly_salary",
        "monthly_income",
        "salary",
        "income",
        "monthly_income_inr",
    ],
)

age_col = find_column(
    df,
    [
        "age",
        "applicant_age",
        "customer_age",
    ],
)

affordability_col = find_column(
    df,
    [
        "affordability_ratio",
        "emi_affordability_ratio",
    ],
)

numeric_cols = df.select_dtypes(include="number").columns.tolist()
categorical_cols = df.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

median_salary = (
    numeric_series(df, salary_col).median()
    if salary_col else float("nan")
)

median_emi = (
    numeric_series(df, emi_col).median()
    if emi_col else float("nan")
)

unique_scenarios = (
    int(df[scenario_col].nunique(dropna=True))
    if scenario_col else 0
)

unique_eligibility = (
    int(df[eligibility_col].nunique(dropna=True))
    if eligibility_col else 0
)


# ------------------------------------------------------------
# 7. HERO SECTION
# ------------------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <div class="hero-eyebrow">
            <span>✦</span> FINANCIAL INTELLIGENCE · LIVE DASHBOARD
        </div>
        <div class="hero-title">
            Smarter lending.<br>
            <span>Clearer decisions.</span>
        </div>
        <div class="hero-description">
            An integrated machine-learning platform for EMI eligibility
            assessment, affordable installment estimation, financial
            dataset exploration, and model performance monitoring.
            Turn financial data into insights through a single workspace.
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

top_left, top_right = st.columns([3, 1])

with top_left:
    st.caption(
        f"DATA SOURCE  /  {data_source}  ·  "
        f"{pd.Timestamp.now().strftime('%d %b %Y')}"
    )

with top_right:
    st.markdown(
        """
        <div style="text-align:right;color:#8fa1ba;font-size:0.77rem;">
        <span style="color:#57e3a2;">●</span> Dashboard initialized
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------
# 8. KPI CARDS
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# 9. DATA-DRIVEN INSIGHTS
# ------------------------------------------------------------

section_heading(
    "Financial overview",
    "Explore the distribution of EMI scenarios, eligibility outcomes, "
    "and affordability indicators.",
)

chart_left, chart_right = st.columns([1.12, 1], gap="large")

with chart_left:
    st.markdown(
        '<div class="content-panel">',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="panel-title">Loan scenario composition</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="panel-description">'
        'Understand which financing categories appear most frequently '
        'in the available data.</div>',
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
            fig.update_traces(
                textposition="outside",
                cliponaxis=False,
                marker_line_width=0,
            )
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
            "A recognized loan scenario column was not found. "
            "The dataset preview below can help verify your column names."
        )

    st.markdown("</div>", unsafe_allow_html=True)


with chart_right:
    st.markdown(
        '<div class="content-panel">',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="panel-title">Eligibility distribution</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="panel-description">'
        'View the balance of eligibility labels in the dataset. '
        'This is a dataset distribution, not a live prediction.</div>',
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
                hole=0.68,
                color_discrete_sequence=[
                    "#5ee7f5",
                    "#7299ff",
                    "#57e3a2",
                    "#b69aff",
                    "#f6bd60",
                    "#64748b",
                ],
            )

            fig.update_traces(
                textposition="inside",
                textinfo="percent",
                marker=dict(line=dict(color="#10192a", width=3)),
                hovertemplate=(
                    "<b>%{label}</b><br>"
                    "Records: %{value:,}<br>"
                    "Share: %{percent}<extra></extra>"
                ),
            )

            fig.update_layout(
                showlegend=True,
                legend=dict(orientation="h", y=-0.15),
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
            f"{unique_eligibility} distinct eligibility label(s) detected."
        )
    else:
        st.info(
            "No recognized eligibility column was found. "
            "Verify the dataset's target column."
        )

    st.markdown("</div>", unsafe_allow_html=True)


# ------------------------------------------------------------
# 10. NUMERIC FINANCIAL INDICATORS
# ------------------------------------------------------------

section_heading(
    "Financial indicators",
    "Descriptive statistics from available numeric fields. "
    "Values below summarize the dataset rather than an individual applicant.",
)

f1, f2, f3 = st.columns(3, gap="medium")

with f1:
    salary_display = (
        rupees(median_salary)
        if pd.notna(median_salary) else "N/A"
    )
    metric_card(
        "Median monthly salary",
        salary_display,
        "Median of the detected income column",
        "₹",
    )

with f2:
    emi_display = (
        rupees(median_emi)
        if pd.notna(median_emi) else "N/A"
    )
    metric_card(
        "Median EMI-related value",
        emi_display,
        "Median of the detected EMI column",
        "↗",
    )

with f3:
    missing_pct = (
        (missing_cells / (row_count * column_count)) * 100
        if row_count and column_count else 0
    )
    metric_card(
        "Data completeness",
        f"{100 - missing_pct:.1f}%",
        "Percentage of cells that are not missing",
        "✓",
    )


# ------------------------------------------------------------
# 11. AFFORDABILITY ANALYSIS
# ------------------------------------------------------------

section_heading(
    "Affordability analysis",
    "Inspect the relationship between financial variables when "
    "the relevant columns are present.",
)

if salary_col and emi_col:
    affordability_df = df[[salary_col, emi_col]].copy()
    affordability_df[salary_col] = pd.to_numeric(
        affordability_df[salary_col], errors="coerce"
    )
    affordability_df[emi_col] = pd.to_numeric(
        affordability_df[emi_col], errors="coerce"
    )
    affordability_df = affordability_df.dropna()

    affordability_df = affordability_df[
        (affordability_df[salary_col] >= 0)
        & (affordability_df[emi_col] >= 0)
    ]

    if not affordability_df.empty:
        if len(affordability_df) > 4000:
            affordability_df = affordability_df.sample(
                n=4000, random_state=42
            )

        if PLOTLY_AVAILABLE:
            fig = px.scatter(
                affordability_df,
                x=salary_col,
                y=emi_col,
                opacity=0.55,
                render_mode="webgl",
                labels={
                    salary_col: "Monthly salary",
                    emi_col: "EMI-related value",
                },
                color_discrete_sequence=["#5ee7f5"],
            )

            fig.update_traces(
                marker=dict(size=6),
                hovertemplate=(
                    "Monthly salary: %{x:,.0f}<br>"
                    "EMI value: %{y:,.0f}<extra></extra>"
                ),
            )

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
                affordability_df,
                x=salary_col,
                y=emi_col,
                width="stretch",
            )

        st.caption(
            "Interpretation: this chart displays the observed relationship "
            "between the selected columns. It does not establish causation "
            "or guarantee an individual's affordability."
        )

    else:
        st.info("No usable numeric rows were available for this chart.")

elif affordability_col:
    ratio_data = numeric_series(df, affordability_col)

    if not ratio_data.empty:
        if PLOTLY_AVAILABLE:
            ratio_df = pd.DataFrame(
                {"Affordability ratio": ratio_data}
            )

            fig = px.histogram(
                ratio_df,
                x="Affordability ratio",
                nbins=35,
                color_discrete_sequence=["#7299ff"],
            )
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(
                plotly_theme(fig, 360),
                width="stretch",
                config={"displayModeBar": False},
            )
        else:
            st.bar_chart(
                ratio_data.value_counts(bins=20).sort_index(),
                width="stretch",
            )
    else:
        st.info("The affordability ratio column contains no usable values.")

else:
    st.info(
        "Affordability visualization is unavailable because the dataset "
        "does not contain a recognized salary, EMI, or affordability-ratio "
        "column. The Dataset Explorer below shows the actual column names."
    )


# ------------------------------------------------------------
# 12. MODEL INFORMATION
# ------------------------------------------------------------

section_heading(
    "Machine learning architecture",
    "A unified interface for classification and regression workflows.",
)

m1, m2, m3 = st.columns(3, gap="medium")

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

class_description = (
    "Estimates the EMI eligibility class from applicant and "
    "financial features using the trained classification pipeline."
)

reg_description = (
    "Estimates a maximum affordable EMI value using the trained "
    "regression pipeline and its required input features."
)

with m1:
    st.markdown(
        f"""
        <div class="model-card">
            <div class="model-icon">◈</div>
            <div class="model-name">Eligibility classification</div>
            <div class="model-description">
                {html.escape(class_description)}
            </div>
            <div class="status-chip">Classification pipeline</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m2:
    st.markdown(
        f"""
        <div class="model-card">
            <div class="model-icon">↗</div>
            <div class="model-name">EMI regression</div>
            <div class="model-description">
                {html.escape(reg_description)}
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
            <div class="model-name">MLflow model monitoring</div>
            <div class="model-description">
                Experiment tracking, logged metrics, and model-related
                information are handled by the dedicated monitoring page
                configured in this repository.
            </div>
            <div class="status-chip">Separate tracking module</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with st.expander("Model configuration details"):
    st.markdown("**Classification model configured in metadata**")
    st.code(classification_name)

    st.markdown("**Regression model configured in metadata**")
    st.code(regression_name)

    st.markdown(
        "The dashboard intentionally avoids loading the full model "
        "artifacts during homepage initialization. The prediction pages "
        "should load models when inference is requested."
    )

    if isinstance(feature_info, dict) and feature_info:
        st.markdown("**Available metadata fields**")
        st.json(
            {
                str(k): (
                    v if isinstance(v, (str, int, float, bool, list, dict))
                    else str(v)
                )
                for k, v in feature_info.items()
            }
        )
    else:
        st.caption(
            "No feature metadata was returned by load_feature_info(). "
            "The prediction pages can still use their configured models."
        )


# ------------------------------------------------------------
# 13. DATASET EXPLORER
# ------------------------------------------------------------

section_heading(
    "Dataset explorer",
    "Search, inspect, and profile the dataset without exposing "
    "the model-loading process to the homepage.",
)

with st.container(border=True):
    tab_preview, tab_profile, tab_missing = st.tabs(
        [
            "▦  Data preview",
            "◉  Statistical profile",
            "⌁  Data quality",
        ]
    )

    with tab_preview:
        search_col, row_col = st.columns([3, 1])

        with search_col:
            search_term = st.text_input(
                "Search records",
                placeholder="Search by scenario, eligibility, or any value...",
                key="dashboard_dataset_search",
            )

        with row_col:
            preview_rows = st.selectbox(
                "Rows to display",
                [10, 25, 50, 100],
                index=1,
                key="dashboard_preview_rows",
            )

        filtered_df = df

        if search_term.strip():
            term = search_term.strip()

            mask = df.astype("string").apply(
                lambda col: col.str.contains(
                    term, case=False, na=False, regex=False
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

        csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇ Download filtered dataset as CSV",
            data=csv_bytes,
            file_name="emipredict_filtered_data.csv",
            mime="text/csv",
            use_container_width=True,
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
                "Column": df.columns.astype(str),
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

    with tab_missing:
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


# ------------------------------------------------------------
# 14. DATASET INSIGHTS
# ------------------------------------------------------------

section_heading(
    "Quick insights",
    "Automatically generated observations based on available columns.",
)

insight_left, insight_right = st.columns(2, gap="large")

with insight_left:
    if scenario_col:
        scenario_counts = df[scenario_col].value_counts(dropna=True)

        if not scenario_counts.empty:
            top_scenario = scenario_counts.index[0]
            top_count = int(scenario_counts.iloc[0])
            top_pct = top_count / max(row_count, 1) * 100

            st.markdown(
                f"""
                <div class="insight">
                    <b>Most frequent loan scenario</b><br>
                    {html.escape(str(top_scenario))} represents
                    <b>{top_pct:.1f}%</b> of the dataset
                    ({top_count:,} records).
                </div>
                """,
                unsafe_allow_html=True,
            )

    if salary_col:
        salary_values = numeric_series(df, salary_col)

        if not salary_values.empty:
            st.markdown(
                f"""
                <div class="insight">
                    <b>Income profile</b><br>
                    The detected salary column has a median value of
                    <b>{rupees(salary_values.median())}</b> and a
                    mean value of <b>{rupees(salary_values.mean())}</b>.
                </div>
                """,
                unsafe_allow_html=True,
            )

with insight_right:
    if eligibility_col:
        label_counts = df[eligibility_col].value_counts(dropna=True)

        if not label_counts.empty:
            dominant_label = label_counts.index[0]
            dominant_pct = label_counts.iloc[0] / max(row_count, 1) * 100

            st.markdown(
                f"""
                <div class="insight">
                    <b>Most frequent eligibility label</b><br>
                    <b>{html.escape(str(dominant_label))}</b> is the most
                    frequent label, accounting for {dominant_pct:.1f}%
                    of non-missing labels.
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        f"""
        <div class="insight">
            <b>Dataset quality</b><br>
            {missing_cells:,} missing cells were detected across
            {row_count:,} rows and {column_count:,} columns.
            Review the Data Quality tab for a per-column breakdown.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------
# 15. FOOTER
# ------------------------------------------------------------

st.markdown(
    """
    <div class="footer">
        <div style="font-family:'Space Grotesk',sans-serif;
                    color:#d9e6fa;font-size:0.92rem;font-weight:700;">
            EMIPredict <span style="color:#5ee7f5;">AI</span>
        </div>
        <div>
            Intelligent Financial Risk Assessment Platform
            &nbsp;·&nbsp; Machine Learning &nbsp;·&nbsp; Streamlit
        </div>
        <div style="margin-top:7px;">
            Dataset summaries are descriptive. Predictions depend on
            the trained model, its input schema, and data quality.
            Results should not be treated as guaranteed loan approvals
            or professional financial advice.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
