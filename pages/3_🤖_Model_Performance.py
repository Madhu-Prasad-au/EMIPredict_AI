import os
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Model Performance | EMIPredict AI", page_icon="🤖", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root{--ink:#eaf1ff;--muted:#91a2bd;--panel:#111d31;--panel2:#15243b;--line:rgba(153,177,211,.16);--mint:#5eead4;--blue:#7c9cff;}
    html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
    .stApp{background:radial-gradient(ellipse at 12% 0%,rgba(46,82,143,.20),transparent 38%),radial-gradient(ellipse at 100% 8%,rgba(31,166,153,.10),transparent 30%),#08111f;color:var(--ink);}
    [data-testid="stHeader"]{background:rgba(8,17,31,.72);}
    [data-testid="stSidebar"]{background:linear-gradient(180deg,#0d192b 0%,#0a1423 100%);border-right:1px solid var(--line);}
    [data-testid="stSidebar"]>div{padding-top:1.5rem;}
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p{color:#aab8ce;}
    .block-container{padding-top:2rem;padding-bottom:3rem;max-width:1500px;}
    h1,h2,h3{font-family:'Manrope',sans-serif!important;letter-spacing:-.035em;} h1{color:#f4f7ff;font-weight:800!important;} h2,h3{color:#e8efff;}
    p,label,.stCaption{color:#a9b8cf;} hr{border-color:var(--line);}
    [data-testid="stMetric"]{background:linear-gradient(145deg,rgba(21,36,59,.96),rgba(14,27,45,.96));border:1px solid var(--line);border-radius:18px;padding:18px 20px;min-height:112px;box-shadow:0 12px 30px rgba(0,0,0,.12);}
    [data-testid="stMetricLabel"]{color:#9eafc8!important;font-size:.82rem;} [data-testid="stMetricValue"]{color:#f2f6ff!important;font-family:'Manrope',sans-serif;}
    [data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:14px;overflow:hidden;}
    .hero{position:relative;overflow:hidden;border-radius:26px;padding:32px 34px;margin:0 0 25px;background:linear-gradient(118deg,#142943 0%,#13233a 54%,#123c42 100%);border:1px solid rgba(134,171,214,.22);box-shadow:0 24px 65px rgba(0,0,0,.20);}
    .hero:after{content:'';position:absolute;width:320px;height:320px;border-radius:50%;right:-95px;top:-190px;background:rgba(94,234,212,.10);border:1px solid rgba(94,234,212,.12);}
    .eyebrow{position:relative;z-index:1;text-transform:uppercase;letter-spacing:.16em;font-size:.72rem;font-weight:700;color:#73e7d4;margin-bottom:12px;}
    .hero-title{position:relative;z-index:1;font-family:'Manrope',sans-serif;font-size:clamp(1.8rem,3vw,2.55rem);line-height:1.15;font-weight:800;color:#f5f8ff;margin-bottom:12px;}
    .hero-copy{position:relative;z-index:1;color:#b6c6dc;max-width:780px;font-size:1rem;line-height:1.7;}
    .section-kicker{color:#6ee7d4;text-transform:uppercase;letter-spacing:.15em;font-size:.7rem;font-weight:800;margin-bottom:5px;}
    .section-copy{color:#93a5bf;font-size:.9rem;margin-bottom:18px;}
    .panel{background:linear-gradient(145deg,rgba(20,35,57,.95),rgba(13,25,42,.95));border:1px solid var(--line);border-radius:20px;padding:22px 23px;}
    .stButton>button{border-radius:11px;border:1px solid rgba(94,234,212,.35);background:#5eead4;color:#082126;font-weight:800;} .stButton>button:hover{background:#8bf3e3;border-color:#8bf3e3;color:#082126;}
    [data-testid="stForm"]{background:rgba(17,29,49,.55);border:1px solid var(--line);border-radius:18px;padding:20px;}
    div[data-baseweb="tab-list"]{gap:8px;} button[data-baseweb="tab"]{border-radius:10px;padding:10px 15px;}
    input,textarea,[data-baseweb="select"]>div{border-radius:10px!important;}
    .footnote{color:#71839d;font-size:.78rem;}
    @media(max-width:800px){.hero{padding:24px 21px;}.block-container{padding-top:1rem;}}
    </style>
    """, unsafe_allow_html=True,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_DIR = os.path.join(BASE_DIR, "reports")
CLASS_PATH = os.path.join(REPORT_DIR, "classification_model_comparison.csv")
REG_PATH = os.path.join(REPORT_DIR, "regression_model_comparison.csv")

st.markdown(
    """<div class="hero"><div class="eyebrow">Model observability · Evaluation lab</div>
    <div class="hero-title">Model performance & comparison</div>
    <div class="hero-copy">Review the reported evaluation metrics for eligibility classification and maximum-affordable-EMI regression. Lower error is better for RMSE; higher scores are generally better for F1, ROC-AUC, and R².</div></div>""", unsafe_allow_html=True)

def load_report(path, label):
    if not os.path.exists(path):
        st.warning(f"{label} report was not found: `{os.path.relpath(path, BASE_DIR)}`")
        return pd.DataFrame()
    try:
        frame = pd.read_csv(path)
        if frame.empty:
            st.info(f"{label} report exists but contains no rows.")
        return frame
    except Exception as exc:
        st.error(f"Could not read the {label.lower()} report: {exc}")
        return pd.DataFrame()

classification_df = load_report(CLASS_PATH, "Classification")
regression_df = load_report(REG_PATH, "Regression")

if classification_df.empty and regression_df.empty:
    st.info("Add the two model comparison CSV files to the project's `reports/` folder to populate this dashboard.")
    st.stop()

# Normalize numeric report columns without changing source labels.
for frame, columns in [(classification_df, ["F1 Score", "ROC-AUC"]), (regression_df, ["RMSE", "R2 Score"])]:
    for col in columns:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

if not classification_df.empty:
    st.markdown('<div class="section-kicker">Classification</div><h2>Eligibility model comparison</h2>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    c1.metric("Models evaluated", len(classification_df))
    if "F1 Score" in classification_df and classification_df["F1 Score"].notna().any():
        best = classification_df.loc[classification_df["F1 Score"].idxmax()]
        c2.metric("Top F1 score", f"{best['F1 Score']:.4f}")
        c3.metric("Top model", str(best.get("Model", "Not specified")))
    st.dataframe(classification_df, use_container_width=True, hide_index=True)
    chart_cols = [c for c in ["F1 Score", "ROC-AUC"] if c in classification_df.columns]
    if "Model" in classification_df.columns and chart_cols:
        st.markdown("### Classification metrics")
        st.bar_chart(classification_df.set_index("Model")[chart_cols], height=330)

if not regression_df.empty:
    st.divider()
    st.markdown('<div class="section-kicker">Regression</div><h2>Affordable EMI model comparison</h2>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    c1.metric("Models evaluated", len(regression_df))
    if "RMSE" in regression_df and regression_df["RMSE"].notna().any():
        best = regression_df.loc[regression_df["RMSE"].idxmin()]
        c2.metric("Lowest RMSE", f"₹{best['RMSE']:,.2f}")
        c3.metric("Lowest-error model", str(best.get("Model", "Not specified")))
    st.dataframe(regression_df, use_container_width=True, hide_index=True)
    chart_cols = [c for c in ["RMSE", "R2 Score"] if c in regression_df.columns]
    if "Model" in regression_df.columns and chart_cols:
        st.markdown("### Regression metrics")
        st.bar_chart(regression_df.set_index("Model")[chart_cols], height=330)

st.divider()
st.markdown('<p class="footnote">These values are read from the saved comparison reports; this page does not retrain models. Confirm evaluation methodology and test-set provenance before comparing models.</p>', unsafe_allow_html=True)
