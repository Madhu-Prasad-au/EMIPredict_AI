import html
import os

import pandas as pd
import streamlit as st

from utils.data_utils import load_data
from utils.model_utils import (
    load_classification_model,
    load_regression_model,
    load_feature_info,
)


# -----------------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="EMIPredict AI | Financial Intelligence",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# Visual system: Streamlit-native, deployment-friendly CSS
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
      --ink: #eaf1ff;
      --muted: #91a2bd;
      --panel: #111d31;
      --panel-2: #15243b;
      --line: rgba(153, 177, 211, .16);
      --mint: #5eead4;
      --blue: #7c9cff;
    }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp {
      background: radial-gradient(ellipse at 12% 0%, rgba(46, 82, 143, .20), transparent 38%),
                  radial-gradient(ellipse at 100% 8%, rgba(31, 166, 153, .10), transparent 30%),
                  #08111f;
      color: var(--ink);
    }
    [data-testid="stHeader"] { background: rgba(8,17,31,.72); }
    [data-testid="stSidebar"] {
      background: linear-gradient(180deg, #0d192b 0%, #0a1423 100%);
      border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] > div { padding-top: 1.5rem; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #aab8ce; }
    [data-testid="stSidebarNav"] { padding-top: .5rem; }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1500px; }
    h1, h2, h3 { font-family: 'Manrope', sans-serif !important; letter-spacing: -.035em; }
    h1 { color: #f4f7ff; font-weight: 800 !important; }
    h2, h3 { color: #e8efff; }
    p, label, .stCaption { color: #a9b8cf; }
    hr { border-color: var(--line); }
    [data-testid="stMetric"] {
      background: linear-gradient(145deg, rgba(21,36,59,.96), rgba(14,27,45,.96));
      border: 1px solid var(--line); border-radius: 18px;
      padding: 18px 20px; min-height: 120px;
      box-shadow: 0 12px 30px rgba(0,0,0,.12);
    }
    [data-testid="stMetricLabel"] { color: #9eafc8 !important; font-size: .82rem; }
    [data-testid="stMetricValue"] { color: #f2f6ff !important; font-family: 'Manrope', sans-serif; }
    [data-testid="stMetricDelta"] { font-size: .78rem; }
    [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 14px; overflow: hidden; }
    .hero {
      position: relative; overflow: hidden; border-radius: 26px;
      padding: 34px 36px; margin: 0 0 25px 0;
      background: linear-gradient(118deg, #142943 0%, #13233a 54%, #123c42 100%);
      border: 1px solid rgba(134, 171, 214, .22);
      box-shadow: 0 24px 65px rgba(0,0,0,.20);
    }
    .hero:after { content: ''; position: absolute; width: 340px; height: 340px; border-radius: 50%;
      right: -95px; top: -190px; background: rgba(94,234,212,.10); border: 1px solid rgba(94,234,212,.12); }
    .eyebrow { text-transform: uppercase; letter-spacing: .16em; font-size: .72rem; font-weight: 700; color: #73e7d4; margin-bottom: 12px; }
    .hero-title { font-family: 'Manrope', sans-serif; font-size: clamp(1.8rem, 3vw, 2.65rem); line-height: 1.15; font-weight: 800; color: #f5f8ff; max-width: 760px; margin-bottom: 13px; }
    .hero-copy { color: #b6c6dc; max-width: 710px; font-size: 1rem; line-height: 1.7; }
    .hero-pill { display: inline-flex; align-items: center; gap: 8px; margin-top: 22px; padding: 8px 12px; border-radius: 999px;
      border: 1px solid rgba(94,234,212,.28); background: rgba(94,234,212,.08); color: #aaf6e9; font-size: .78rem; font-weight: 600; }
    .section-kicker { color: #6ee7d4; text-transform: uppercase; letter-spacing: .15em; font-size: .7rem; font-weight: 800; margin-bottom: 5px; }
    .section-title { font-family: 'Manrope', sans-serif; color: #eff4ff; font-weight: 800; font-size: 1.35rem; letter-spacing: -.035em; margin-bottom: 4px; }
    .section-copy { color: #93a5bf; font-size: .9rem; margin-bottom: 18px; }
    .panel { height: 100%; background: linear-gradient(145deg, rgba(20,35,57,.95), rgba(13,25,42,.95));
      border: 1px solid var(--line); border-radius: 20px; padding: 22px 23px; }
    .panel-label { color: #94a8c5; font-size: .78rem; font-weight: 700; text-transform: uppercase; letter-spacing: .09em; margin-bottom: 13px; }
    .model-name { font-family: 'Manrope', sans-serif; color: #f1f5ff; font-weight: 800; font-size: 1.15rem; line-height: 1.35; margin-bottom: 10px; overflow-wrap: anywhere; }
    .model-desc { color: #a6b6cc; font-size: .88rem; line-height: 1.65; }
    .tag { display: inline-block; border: 1px solid rgba(94,234,212,.23); color: #a1f1e4; background: rgba(94,234,212,.07); border-radius: 8px; padding: 5px 9px; font-size: .72rem; margin: 3px 4px 0 0; }
    .step-card { background: rgba(18,32,52,.92); border: 1px solid var(--line); border-radius: 15px; padding: 17px 15px; min-height: 130px; }
    .step-num { color: #65e5d1; font-size: .73rem; font-weight: 800; letter-spacing: .12em; margin-bottom: 11px; }
    .step-title { color: #e9f0ff; font-weight: 700; font-size: .91rem; margin-bottom: 6px; }
    .step-copy { color: #91a4bf; font-size: .78rem; line-height: 1.5; }
    .footnote { color: #71839d; font-size: .78rem; }
    .stButton > button { border-radius: 11px; border: 1px solid rgba(94,234,212,.35); background: #5eead4; color: #082126; font-weight: 800; }
    .stButton > button:hover { background: #8bf3e3; border-color: #8bf3e3; color: #082126; }
    div[data-baseweb="tab-list"] { gap: 8px; }
    button[data-baseweb="tab"] { border-radius: 10px; padding: 10px 15px; }
    @media (max-width: 800px) { .hero { padding: 25px 22px; } .panel { padding: 18px; } }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Load application assets
# -----------------------------------------------------------------------------
@st.cache_resource
def get_models():
    classification_model = load_classification_model()
    regression_model = load_regression_model()
    feature_info = load_feature_info()
    return classification_model, regression_model, feature_info


@st.cache_data
def get_data():
    return load_data()


try:
    classification_model, regression_model, feature_info = get_models()
    df = get_data()
    if df is None or df.empty:
        st.error("The financial dataset loaded successfully, but it contains no records.")
        st.stop()
except Exception as exc:
    st.error(f"Unable to load project files: {exc}")
    st.info("Check that the dataset, saved models, and files inside the `utils` folder are included in the deployment.")
    st.stop()


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def safe_text(value, fallback="Not available"):
    if value is None:
        return fallback
    text = str(value).strip()
    return text if text and text.lower() != "nan" else fallback


def section_header(kicker, title, copy):
    st.markdown(
        f'<div class="section-kicker">{html.escape(kicker)}</div>'
        f'<div class="section-title">{html.escape(title)}</div>'
        f'<div class="section-copy">{html.escape(copy)}</div>',
        unsafe_allow_html=True,
    )


def panel_open(label):
    st.markdown(f'<div class="panel"><div class="panel-label">{html.escape(label)}</div>', unsafe_allow_html=True)


def panel_close():
    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:12px;margin:0 0 8px 0;">
          <div style="width:43px;height:43px;border-radius:14px;background:linear-gradient(135deg,#5eead4,#7c9cff);display:flex;align-items:center;justify-content:center;color:#071827;font-size:22px;font-weight:900;">E</div>
          <div><div style="font-family:Manrope,sans-serif;color:#f1f5ff;font-size:1.05rem;font-weight:800;">EMIPredict <span style="color:#5eead4;">AI</span></div>
          <div style="font-size:.68rem;color:#8499b6;letter-spacing:.1em;text-transform:uppercase;">Financial intelligence</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("**YOUR WORKSPACE**")
    st.caption("Use the page navigation to move between prediction, analytics, model comparison, and record management.")
    st.markdown("---")
    st.markdown("**PLATFORM MODULES**")
    st.markdown("🔮  EMI eligibility & prediction")
    st.markdown("📊  Financial data explorer")
    st.markdown("🧠  Model performance")
    st.markdown("🗂️  Financial record management")
    st.markdown("---")
    st.markdown("<span style='color:#5eead4;font-weight:700;'>●</span> <span style='color:#9eafc8;'>Dashboard initialized</span>", unsafe_allow_html=True)
    st.caption("FinTech • Machine Learning • Decision Support")


# -----------------------------------------------------------------------------
# Hero
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">FINANCIAL DECISION INTELLIGENCE</div>
      <div class="hero-title">Make smarter EMI decisions with machine learning.</div>
      <div class="hero-copy">A unified workspace for assessing EMI eligibility, estimating a customer's affordable monthly commitment, and exploring the data behind your predictions.</div>
      <div class="hero-pill"><span>✦</span> Classification + Regression <span style="opacity:.5;">|</span> Data-led assessment</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Metrics are based on the loaded dataset; use fallbacks for optional columns.
record_count = len(df)
feature_count = int(df.shape[1])
scenario_count = int(df["emi_scenario"].nunique()) if "emi_scenario" in df.columns else None
class_count = int(df["emi_eligibility"].nunique()) if "emi_eligibility" in df.columns else None

section_header("OVERVIEW", "Your portfolio at a glance", "A live snapshot of the dataset currently connected to EMIPredict AI.")
metric_cols = st.columns(4, gap="medium")
with metric_cols[0]:
    st.metric("Customer records", f"{record_count:,}")
with metric_cols[1]:
    st.metric("Dataset columns", f"{feature_count:,}")
with metric_cols[2]:
    st.metric("EMI scenarios", f"{scenario_count:,}" if scenario_count is not None else "—")
with metric_cols[3]:
    st.metric("Eligibility classes", f"{class_count:,}" if class_count is not None else "—")

st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

# Model cards
section_header("MODEL INTELLIGENCE", "Two models. One decision workflow.", "Classification estimates eligibility, while regression estimates an affordable monthly EMI amount.")
model_left, model_right = st.columns(2, gap="large")
classes = feature_info.get("classification_classes", []) if isinstance(feature_info, dict) else []
if isinstance(classes, (str, int, float)):
    classes = [classes]
classes_html = "".join(f'<span class="tag">{html.escape(str(item))}</span>' for item in classes) or '<span class="tag">Classes not listed</span>'
classification_name = safe_text(feature_info.get("best_classification_model") if isinstance(feature_info, dict) else None)
regression_name = safe_text(feature_info.get("best_regression_model") if isinstance(feature_info, dict) else None)
with model_left:
    st.markdown(
        f"""
        <div class="panel">
          <div class="panel-label">01 / Classification engine</div>
          <div style="font-size:1.7rem;margin-bottom:12px;">◈</div>
          <div class="model-name">{html.escape(classification_name)}</div>
          <div class="model-desc">Evaluates the customer's profile and predicts the EMI eligibility category.</div>
          <div style="margin-top:17px;color:#91a5c1;font-size:.76rem;font-weight:700;">ELIGIBILITY LABELS</div>
          <div style="margin-top:6px;">{classes_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with model_right:
    st.markdown(
        f"""
        <div class="panel">
          <div class="panel-label">02 / Regression engine</div>
          <div style="font-size:1.7rem;margin-bottom:12px;">⌁</div>
          <div class="model-name">{html.escape(regression_name)}</div>
          <div class="model-desc">Estimates a maximum affordable monthly EMI amount to support more informed customer-level decisions.</div>
          <div style="margin-top:17px;color:#91a5c1;font-size:.76rem;font-weight:700;">MODEL OUTPUT</div>
          <div style="margin-top:6px;"><span class="tag">Estimated monthly EMI</span><span class="tag">Numeric prediction</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:25px'></div>", unsafe_allow_html=True)

# Dataset analysis area
section_header("DATA EXPLORER", "Understand the data behind the models", "Review sample records and inspect the available fields before running customer-level predictions.")
tab_preview, tab_profile, tab_distribution = st.tabs(["▤  Records", "◫  Data profile", "▥  Eligibility distribution"])
with tab_preview:
    top_controls = st.columns([1, 1, 2])
    with top_controls[0]:
        preview_rows = st.selectbox("Rows to display", [10, 25, 50, 100], index=0)
    with top_controls[1]:
        show_all_columns = st.toggle("All columns", value=False)
    with top_controls[2]:
        search_text = st.text_input("Filter records", placeholder="Search visible values…", label_visibility="collapsed")
    preview_df = df.copy()
    if search_text.strip():
        search_mask = preview_df.astype(str).apply(lambda col: col.str.contains(search_text.strip(), case=False, na=False)).any(axis=1)
        preview_df = preview_df.loc[search_mask]
    if not show_all_columns:
        preview_df = preview_df.iloc[:, : min(8, len(preview_df.columns))]
    st.dataframe(preview_df.head(preview_rows), use_container_width=True, hide_index=True)
    st.caption(f"Showing {min(preview_rows, len(preview_df)):,} of {len(preview_df):,} matching records. Use the other tabs to inspect dataset quality and class balance.")
with tab_profile:
    profile_a, profile_b = st.columns(2, gap="large")
    with profile_a:
        st.markdown("**Dataset dimensions**")
        profile_df = pd.DataFrame({"Measure": ["Rows", "Columns", "Duplicate rows", "Missing values"], "Value": [len(df), len(df.columns), int(df.duplicated().sum()), int(df.isna().sum().sum())]})
        st.dataframe(profile_df, use_container_width=True, hide_index=True)
    with profile_b:
        st.markdown("**Column completeness**")
        missing = df.isna().sum().rename("Missing values").to_frame()
        missing["Completeness %"] = ((len(df) - missing["Missing values"]) / max(len(df), 1) * 100).round(1)
        missing = missing.sort_values("Completeness %")
        st.dataframe(missing, use_container_width=True)
with tab_distribution:
    if "emi_eligibility" in df.columns:
        counts = df["emi_eligibility"].fillna("Missing").astype(str).value_counts().rename_axis("Eligibility class").reset_index(name="Records")
        chart_col, table_col = st.columns([1.4, 1], gap="large")
        with chart_col:
            st.markdown("**Records by eligibility class**")
            st.bar_chart(counts.set_index("Eligibility class"), use_container_width=True)
        with table_col:
            st.markdown("**Class breakdown**")
            counts["Share"] = (counts["Records"] / max(len(df), 1) * 100).round(1).astype(str) + "%"
            st.dataframe(counts, use_container_width=True, hide_index=True)
    else:
        st.info("The current dataset does not contain an `emi_eligibility` column, so a class distribution cannot be shown.")

st.markdown("<div style='height:25px'></div>", unsafe_allow_html=True)

# Visual pipeline
section_header("HOW IT WORKS", "From customer data to an informed estimate", "A high-level view of the workflow implemented across the application.")
steps = [
    ("01", "Customer data", "Load and prepare the financial dataset."),
    ("02", "Data preparation", "Clean fields and engineer useful inputs."),
    ("03", "Model inference", "Run eligibility classification and EMI regression."),
    ("04", "Model tracking", "Use MLflow experiments to compare candidates."),
    ("05", "Best model", "Select the saved model configured for the app."),
    ("06", "Prediction", "Present estimates to support a review."),
]
step_cols = st.columns(3, gap="medium")
for idx, (number, title, description) in enumerate(steps):
    with step_cols[idx % 3]:
        st.markdown(
            f"""
            <div class="step-card" style="margin-bottom:14px;">
              <div class="step-num">STEP {number}</div>
              <div class="step-title">{html.escape(title)}</div>
              <div class="step-copy">{html.escape(description)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Footer
st.markdown("<div style='height:15px'></div><hr>", unsafe_allow_html=True)
footer_left, footer_right = st.columns([2, 1])
with footer_left:
    st.markdown("<div style='color:#a7b7ce;font-weight:700;'>EMIPredict <span style='color:#5eead4;'>AI</span></div><div class='footnote'>Financial Risk Assessment Platform · Machine Learning Decision Support</div>", unsafe_allow_html=True)
with footer_right:
    st.markdown("<div class='footnote' style='text-align:right;padding-top:5px;'>Predictions are estimates and should be reviewed alongside applicable lending policies.</div>", unsafe_allow_html=True)
