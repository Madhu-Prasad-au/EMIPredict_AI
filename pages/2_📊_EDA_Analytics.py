import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.data_utils import load_data

st.set_page_config(page_title="EDA & Analytics | EMIPredict AI", page_icon="📊", layout="wide")

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

# Cache the dataframe as a shared resource instead of serializing/copying it on
# every rerun. Do not mutate `df` anywhere below.
@st.cache_resource(show_spinner="Loading financial dataset…")
def get_data():
    return load_data()

try:
    df = get_data()
    if df is None or df.empty:
        st.error("The dataset is empty. Check the dataset file or Hugging Face download settings.")
        st.stop()
except Exception as exc:
    st.error(f"Unable to load the financial dataset: {exc}")
    st.info("Confirm that `utils/data_utils.py` can access the CSV locally or download it from Hugging Face.")
    st.stop()

required = ["emi_scenario", "emi_eligibility", "monthly_salary", "credit_score", "max_monthly_emi"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error("The dataset is missing columns required for the analytics page: " + ", ".join(missing))
    st.stop()

def fmt_currency(value):
    return f"₹{value:,.0f}" if pd.notna(value) else "—"

st.markdown(
    """<div class="hero"><div class="eyebrow">Analytics workspace · Portfolio insights</div>
    <div class="hero-title">Explore the financial dataset</div>
    <div class="hero-copy">Filter EMI scenarios and eligibility groups, inspect customer-level patterns, and compare income, credit profile, and estimated EMI capacity.</div></div>""",
    unsafe_allow_html=True,
)

# Build filters from the two low-cardinality columns only.
with st.sidebar:
    st.markdown("### 🎛️ Analytics filters")
    scenarios = sorted(df["emi_scenario"].dropna().astype(str).unique().tolist())
    eligibilities = sorted(df["emi_eligibility"].dropna().astype(str).unique().tolist())
    scenario_filter = st.multiselect("EMI scenario", scenarios, default=scenarios)
    eligibility_filter = st.multiselect("Eligibility", eligibilities, default=eligibilities)
    chart_sample_size = st.select_slider(
        "Chart sample size", options=[500, 1000, 2000, 3000], value=2000
    )
    st.caption("Charts use a reproducible sample to reduce memory use on hosted deployments.")

# A single boolean mask is substantially lighter than repeatedly converting the
# whole dataset to strings or making several intermediate dataframe copies.
mask = (
    df["emi_scenario"].isin(scenario_filter)
    & df["emi_eligibility"].isin(eligibility_filter)
)
filtered_count = int(mask.sum())
if filtered_count == 0:
    st.warning("No records match the selected filters. Choose at least one scenario and eligibility class.")
    st.stop()

st.markdown('<div class="section-kicker">Snapshot</div><h2>Key portfolio indicators</h2>', unsafe_allow_html=True)
salary_mean = pd.to_numeric(df.loc[mask, "monthly_salary"], errors="coerce").mean()
credit_mean = pd.to_numeric(df.loc[mask, "credit_score"], errors="coerce").mean()
emi_mean = pd.to_numeric(df.loc[mask, "max_monthly_emi"], errors="coerce").mean()

a, b, c, d = st.columns(4)
a.metric("Filtered records", f"{filtered_count:,}", f"{filtered_count / max(len(df), 1):.1%} of dataset")
b.metric("Average monthly salary", fmt_currency(salary_mean))
c.metric("Average credit score", f"{credit_mean:,.0f}" if pd.notna(credit_mean) else "—")
d.metric("Average maximum EMI", fmt_currency(emi_mean))

# Extract only the columns needed for charts, then sample immediately.
chart_cols = ["emi_scenario", "emi_eligibility", "monthly_salary", "credit_score", "max_monthly_emi"]
chart_data = df.loc[mask, chart_cols]
if len(chart_data) > chart_sample_size:
    chart_data = chart_data.sample(n=chart_sample_size, random_state=42)
chart_data = chart_data.copy()
for col in ["monthly_salary", "credit_score", "max_monthly_emi"]:
    chart_data[col] = pd.to_numeric(chart_data[col], errors="coerce")

st.divider()
left, right = st.columns(2, gap="large")
with left:
    st.markdown("### Eligibility distribution")
    eligibility_counts = df.loc[mask, "emi_eligibility"].value_counts(dropna=False).rename_axis("Eligibility").reset_index(name="Records")
    st.bar_chart(eligibility_counts.set_index("Eligibility"), height=280)
    st.dataframe(eligibility_counts, use_container_width=True, hide_index=True)
with right:
    st.markdown("### EMI scenario mix")
    scenario_counts = df.loc[mask, "emi_scenario"].value_counts(dropna=False).rename_axis("Scenario").reset_index(name="Records")
    st.bar_chart(scenario_counts.set_index("Scenario"), height=280)
    st.dataframe(scenario_counts, use_container_width=True, hide_index=True)

st.divider()
left, right = st.columns(2, gap="large")
with left:
    st.markdown("### Credit score profile")
    credit_plot = chart_data.dropna(subset=["credit_score", "emi_eligibility"])
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for label, group in credit_plot.groupby("emi_eligibility", observed=True):
        ax.hist(group["credit_score"], bins=24, alpha=0.55, label=str(label))
    ax.set_xlabel("Credit score")
    ax.set_ylabel("Sampled records")
    ax.set_title("Credit score by eligibility")
    ax.set_facecolor("#111d31")
    fig.patch.set_alpha(0)
    ax.tick_params(colors="#a9b8cf")
    ax.xaxis.label.set_color("#a9b8cf")
    ax.yaxis.label.set_color("#a9b8cf")
    ax.title.set_color("#eaf1ff")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
with right:
    st.markdown("### Income vs maximum affordable EMI")
    scatter = chart_data.dropna(subset=["monthly_salary", "max_monthly_emi", "emi_eligibility"])
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for label, group in scatter.groupby("emi_eligibility", observed=True):
        ax.scatter(group["monthly_salary"], group["max_monthly_emi"], alpha=0.55, s=12, label=str(label))
    ax.set_xlabel("Monthly salary (₹)")
    ax.set_ylabel("Maximum monthly EMI (₹)")
    ax.set_title("Salary and estimated EMI capacity")
    ax.set_facecolor("#111d31")
    fig.patch.set_alpha(0)
    ax.tick_params(colors="#a9b8cf")
    ax.xaxis.label.set_color("#a9b8cf")
    ax.yaxis.label.set_color("#a9b8cf")
    ax.title.set_color("#eaf1ff")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

st.divider()
st.markdown('<div class="section-kicker">Data explorer</div><h2>Filtered customer records</h2>', unsafe_allow_html=True)
st.caption("To keep the app stable, search and preview operate on a limited record sample. Use the download option for a larger export.")
preview_limit = 5000
# Find matching row indices without making a string copy of every column in the full dataset.
matching_indices = df.index[mask]
if len(matching_indices) > preview_limit:
    preview_indices = matching_indices[:preview_limit]
else:
    preview_indices = matching_indices
preview = df.loc[preview_indices].copy()

search = st.text_input(
    "Search within preview",
    placeholder="Try a scenario, eligibility, employment type, or other text…",
    key="eda_record_search",
)
if search.strip():
    text_cols = preview.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    if text_cols:
        search_mask = preview[text_cols].apply(
            lambda col: col.astype("string").str.contains(search.strip(), case=False, na=False, regex=False)
        ).any(axis=1)
        preview = preview.loc[search_mask]

st.caption(f"Showing {len(preview):,} matching rows from a preview of up to {preview_limit:,} filtered records.")
st.dataframe(preview.head(1000), use_container_width=True, height=400, hide_index=True)

# Avoid building a potentially very large in-memory CSV on a page rerun.
export_limit = 50000
if filtered_count > export_limit:
    st.warning(f"The filtered dataset contains {filtered_count:,} rows. For stability, this page exports the first {export_limit:,} matching rows.")
export_indices = df.index[mask][:export_limit]
export_df = df.loc[export_indices]
st.download_button(
    "⬇️ Download filtered CSV (up to 50,000 rows)",
    data=export_df.to_csv(index=False).encode("utf-8"),
    file_name="emipredict_filtered_analytics.csv",
    mime="text/csv",
)
st.markdown('<p class="footnote">Analytics are descriptive and should not be treated as a standalone lending decision.</p>', unsafe_allow_html=True)
