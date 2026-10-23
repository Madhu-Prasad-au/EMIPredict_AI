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

@st.cache_data(show_spinner="Loading financial dataset…")
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

def fmt_currency(value):
    return f"₹{value:,.0f}" if pd.notna(value) else "—"

required = ["emi_scenario", "emi_eligibility", "monthly_salary", "credit_score", "max_monthly_emi"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error("The dataset is missing columns required for the analytics page: " + ", ".join(missing))
    st.stop()

st.markdown(
    """<div class="hero"><div class="eyebrow">Analytics workspace · Portfolio insights</div>
    <div class="hero-title">Explore the financial dataset</div>
    <div class="hero-copy">Filter EMI scenarios and eligibility groups, inspect customer-level patterns, and compare income, credit profile, and estimated EMI capacity.</div></div>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🎛️ Analytics filters")
    scenarios = sorted(df["emi_scenario"].dropna().astype(str).unique().tolist())
    eligibilities = sorted(df["emi_eligibility"].dropna().astype(str).unique().tolist())
    scenario_filter = st.multiselect("EMI scenario", scenarios, default=scenarios)
    eligibility_filter = st.multiselect("Eligibility", eligibilities, default=eligibilities)
    max_rows = st.select_slider("Chart sample size", options=[1000, 2500, 5000, 10000, 20000], value=10000)
    st.caption("Large charts use a reproducible sample to keep the dashboard responsive.")

filtered_df = df.loc[df["emi_scenario"].astype(str).isin(scenario_filter) & df["emi_eligibility"].astype(str).isin(eligibility_filter)]
if filtered_df.empty:
    st.warning("No records match the selected filters. Choose at least one scenario and eligibility class.")
    st.stop()

num = len(filtered_df)
sample = filtered_df.sample(n=min(max_rows, num), random_state=42) if num else filtered_df
mean_salary = pd.to_numeric(filtered_df["monthly_salary"], errors="coerce").mean()
mean_credit = pd.to_numeric(filtered_df["credit_score"], errors="coerce").mean()
mean_emi = pd.to_numeric(filtered_df["max_monthly_emi"], errors="coerce").mean()

st.markdown('<div class="section-kicker">Snapshot</div><h2>Key portfolio indicators</h2>', unsafe_allow_html=True)
a,b,c,d = st.columns(4)
a.metric("Filtered records", f"{num:,}", f"{num / max(len(df),1):.1%} of dataset")
b.metric("Average monthly salary", fmt_currency(mean_salary))
c.metric("Average credit score", f"{mean_credit:,.0f}" if pd.notna(mean_credit) else "—")
d.metric("Average maximum EMI", fmt_currency(mean_emi))

st.divider()
left,right = st.columns(2, gap="large")
with left:
    st.markdown("### Eligibility distribution")
    eligibility_counts = filtered_df["emi_eligibility"].value_counts().rename_axis("Eligibility").reset_index(name="Records")
    st.bar_chart(eligibility_counts.set_index("Eligibility"), height=300)
    st.dataframe(eligibility_counts, use_container_width=True, hide_index=True)
with right:
    st.markdown("### EMI scenario mix")
    scenario_counts = filtered_df["emi_scenario"].value_counts().rename_axis("Scenario").reset_index(name="Records")
    st.bar_chart(scenario_counts.set_index("Scenario"), height=300)
    st.dataframe(scenario_counts, use_container_width=True, hide_index=True)

st.divider()
left,right = st.columns(2, gap="large")
with left:
    st.markdown("### Credit score profile")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(data=sample, x="credit_score", hue="emi_eligibility", bins=28, element="step", stat="count", common_norm=False, ax=ax)
    ax.set_xlabel("Credit score"); ax.set_ylabel("Customer records"); ax.set_title("Credit score by eligibility")
    fig.patch.set_alpha(0); ax.set_facecolor("#111d31"); ax.tick_params(colors="#a9b8cf"); ax.xaxis.label.set_color("#a9b8cf"); ax.yaxis.label.set_color("#a9b8cf"); ax.title.set_color("#eaf1ff")
    st.pyplot(fig, use_container_width=True); plt.close(fig)
with right:
    st.markdown("### Income vs maximum affordable EMI")
    scatter_cols = ["monthly_salary", "max_monthly_emi", "emi_eligibility"]
    scatter = sample[scatter_cols].copy()
    scatter["monthly_salary"] = pd.to_numeric(scatter["monthly_salary"], errors="coerce")
    scatter["max_monthly_emi"] = pd.to_numeric(scatter["max_monthly_emi"], errors="coerce")
    scatter = scatter.dropna()
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.scatterplot(data=scatter, x="monthly_salary", y="max_monthly_emi", hue="emi_eligibility", alpha=.55, s=24, linewidth=0, ax=ax)
    ax.set_xlabel("Monthly salary (₹)"); ax.set_ylabel("Maximum monthly EMI (₹)"); ax.set_title("Salary and estimated EMI capacity")
    fig.patch.set_alpha(0); ax.set_facecolor("#111d31"); ax.tick_params(colors="#a9b8cf"); ax.xaxis.label.set_color("#a9b8cf"); ax.yaxis.label.set_color("#a9b8cf"); ax.title.set_color("#eaf1ff")
    st.pyplot(fig, use_container_width=True); plt.close(fig)

st.divider()
st.markdown('<div class="section-kicker">Data explorer</div><h2>Filtered customer records</h2>', unsafe_allow_html=True)
search = st.text_input("Search within visible columns", placeholder="Try a scenario, eligibility, or customer attribute…", key="eda_record_search")
view = filtered_df
if search.strip():
    text_cols = view.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    if text_cols:
        mask = view[text_cols].astype("string").apply(lambda col: col.str.contains(search.strip(), case=False, na=False, regex=False)).any(axis=1)
        view = view.loc[mask]
st.caption(f"Showing up to 1,000 of {len(view):,} matching records.")
st.dataframe(view.head(1000), use_container_width=True, height=430, hide_index=True)
st.download_button("⬇️ Download filtered CSV", data=view.to_csv(index=False).encode("utf-8"), file_name="emipredict_filtered_analytics.csv", mime="text/csv")
st.markdown('<p class="footnote">Analytics are descriptive and should not be treated as a standalone lending decision.</p>', unsafe_allow_html=True)
