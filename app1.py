from pathlib import Path
import pickle
import warnings

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# SHOPPER SPECTRUM | Retail intelligence workspace
# Keep this file alongside online_retail.csv, rfm_clusters.csv,
# and the existing models/ directory.
# ============================================================

st.set_page_config(
    page_title="Shopper Spectrum | Retail Intelligence",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"

# ---------- Visual system ----------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
      --ink: #172033;
      --muted: #738096;
      --line: #e7ebf2;
      --surface: #ffffff;
      --canvas: #f5f7fb;
      --violet: #6558d9;
      --teal: #18a999;
    }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: var(--canvas); color: var(--ink); }
    [data-testid="stHeader"] { background: rgba(245,247,251,.88); }
    .block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1600px; }
    [data-testid="stSidebar"] {
      background: #111827;
      border-right: 1px solid rgba(255,255,255,.08);
    }
    [data-testid="stSidebar"] * { color: #e7eaf3; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label { padding: .55rem .6rem; border-radius: 10px; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background: rgba(255,255,255,.08); }
    [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.13); }
    .brand-mark {
      width: 46px; height: 46px; display:flex; align-items:center; justify-content:center;
      border-radius: 15px; background: linear-gradient(135deg,#8b7cff,#5145cd);
      font-size: 24px; box-shadow: 0 8px 24px rgba(101,88,217,.32);
    }
    .brand-name { font-family: 'Manrope',sans-serif; font-size: 1.13rem; font-weight: 800; letter-spacing: -.4px; }
    .brand-sub { color:#9ca7bc; font-size:.74rem; margin-top:2px; }
    .eyebrow { text-transform:uppercase; letter-spacing:1.5px; font-size:.72rem; font-weight:700; color:var(--violet); margin-bottom:.4rem; }
    .page-title { font-family:'Manrope',sans-serif; font-size:2rem; line-height:1.15; font-weight:800; letter-spacing:-1px; color:#172033; margin:0; }
    .page-subtitle { color:var(--muted); font-size:.95rem; margin-top:.45rem; margin-bottom:1.2rem; }
    .surface-card {
      background: var(--surface); border:1px solid var(--line); border-radius:18px;
      padding:20px 22px; box-shadow:0 5px 18px rgba(24,39,75,.035);
    }
    .metric-label { color:#7b879b; font-size:.78rem; font-weight:600; letter-spacing:.2px; }
    .metric-value { font-family:'Manrope',sans-serif; color:#182238; font-size:1.75rem; font-weight:800; letter-spacing:-.8px; margin:.35rem 0 .15rem; }
    .metric-note { color:#8994a7; font-size:.76rem; }
    .metric-icon { float:right; width:38px;height:38px;border-radius:12px;display:flex;align-items:center;justify-content:center;background:#f0efff;font-size:18px; }
    .section-heading { font-family:'Manrope',sans-serif; font-size:1.08rem; font-weight:800; color:#1b263b; margin:.2rem 0 .2rem; }
    .section-caption { color:#8490a3; font-size:.82rem; margin-bottom:.8rem; }
    .hero {
      background:linear-gradient(115deg,#171d35 0%,#29245c 56%,#5146a8 100%);
      border-radius:24px; padding:30px 32px; color:white; position:relative; overflow:hidden;
      box-shadow:0 16px 36px rgba(36,31,91,.16); margin-bottom:22px;
    }
    .hero:after { content:""; position:absolute; right:-55px; top:-100px; width:290px; height:290px; border-radius:50%; border:1px solid rgba(255,255,255,.13); box-shadow:0 0 0 35px rgba(255,255,255,.035),0 0 0 75px rgba(255,255,255,.025); }
    .hero-kicker { color:#c5c1ff; text-transform:uppercase; letter-spacing:1.7px; font-size:.72rem; font-weight:700; }
    .hero-title { font-family:'Manrope',sans-serif; font-size:2.05rem; line-height:1.15; font-weight:800; letter-spacing:-1px; margin:.5rem 0 .65rem; max-width:720px; }
    .hero-copy { color:#d5d7e8; font-size:.94rem; max-width:680px; line-height:1.6; }
    .pill { display:inline-block; padding:5px 10px; border-radius:999px; font-size:.72rem; font-weight:700; background:#eeecff; color:#594ec4; }
    .insight { border-left:3px solid #7a6cf0; padding:10px 14px; background:#f7f6ff; border-radius:0 10px 10px 0; color:#4b5570; font-size:.86rem; margin:.6rem 0; }
    div[data-testid="stPlotlyChart"] { background:#fff; border:1px solid var(--line); border-radius:16px; padding:8px 8px 0; }
    div[data-testid="stMetric"] { background:white; border:1px solid var(--line); border-radius:16px; padding:15px 18px; box-shadow:0 5px 18px rgba(24,39,75,.035); }
    div[data-testid="stMetricLabel"] { color:#7b879b; }
    div[data-testid="stMetricValue"] { color:#182238; font-family:'Manrope',sans-serif; }
    .stButton > button { border-radius:10px; font-weight:700; min-height:2.7rem; }
    .stButton > button[kind="primary"] { background:#6558d9; border-color:#6558d9; }
    div[data-testid="stTabs"] button { font-weight:700; }
    .small-muted { color:#8792a5; font-size:.78rem; }
    .rec-card { border:1px solid var(--line); background:white; border-radius:15px; padding:16px; height:100%; }
    .rec-number { color:#6558d9; font-size:.75rem; font-weight:800; letter-spacing:1px; }
    .rec-name { color:#1d2940; font-weight:800; margin:.35rem 0; }
    footer { visibility:hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

PLOT_LAYOUT = dict(
    template="plotly_white",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color="#657188", size=12),
    title=dict(font=dict(family="Manrope, sans-serif", size=15, color="#202b42")),
    margin=dict(l=18, r=18, t=55, b=22),
    hoverlabel=dict(bgcolor="#172033", font_color="white"),
    legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
)
PALETTE = ["#6558d9", "#19a999", "#f3a64a", "#ed6a83", "#4d8ce8", "#9a70d6", "#7f8c9e"]


def style_fig(fig, height=330):
    fig.update_layout(**PLOT_LAYOUT, height=height)
    fig.update_xaxes(showgrid=False, linecolor="#e9edf4", zeroline=False)
    fig.update_yaxes(gridcolor="#edf0f5", zeroline=False)
    return fig


def money(value):
    if pd.isna(value):
        return "—"
    return f"£{value:,.0f}"


def page_header(kicker, title, subtitle):
    st.markdown(
        f'<div class="eyebrow">{kicker}</div><div class="page-title">{title}</div>'
        f'<div class="page-subtitle">{subtitle}</div>',
        unsafe_allow_html=True,
    )


def card_metric(label, value, note, emoji):
    st.markdown(
        f"""<div class="surface-card" style="padding:17px 18px;min-height:126px">
        <div class="metric-icon">{emoji}</div>
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-note">{note}</div></div>""",
        unsafe_allow_html=True,
    )


# ---------- Load files safely ----------
@st.cache_data(show_spinner=False)
def load_data():
    sales_path = ROOT / "online_retail.csv"
    rfm_path = ROOT / "rfm_clusters.csv"
    missing = [str(p.name) for p in (sales_path, rfm_path) if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required data file(s): " + ", ".join(missing))
    sales = pd.read_csv(sales_path)
    rfm_data = pd.read_csv(rfm_path)
    if "InvoiceDate" in sales.columns:
        sales["InvoiceDate"] = pd.to_datetime(sales["InvoiceDate"], errors="coerce")
    if {"Quantity", "UnitPrice"}.issubset(sales.columns):
        sales["Revenue"] = sales["Quantity"] * sales["UnitPrice"]
    return sales, rfm_data


@st.cache_resource(show_spinner=False)
def load_models():
    files = {
        "kmeans": MODEL_DIR / "kmeans_model.pkl",
        "scaler": MODEL_DIR / "scaler.pkl",
        "labels": MODEL_DIR / "cluster_labels.pkl",
        "recommendations": MODEL_DIR / "recommendations.pkl",
        "products": MODEL_DIR / "product_names.pkl",
    }
    missing = [str(path.relative_to(ROOT)) for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing model file(s): " + ", ".join(missing))
    loaded = {}
    for key, path in files.items():
        with path.open("rb") as handle:
            loaded[key] = pickle.load(handle)
    return loaded


try:
    df, rfm = load_data()
except Exception as exc:
    st.error("Shopper Spectrum could not load its data.")
    st.markdown(
        "Place `online_retail.csv` and `rfm_clusters.csv` in the same folder as `app.py`, "
        "then refresh the app."
    )
    st.exception(exc)
    st.stop()

try:
    models = load_models()
except Exception as exc:
    models = None
    model_error = str(exc)

# ---------- Normalize common data columns ----------
required_sales = {"Quantity", "UnitPrice"}
if not required_sales.issubset(df.columns):
    st.error("The retail CSV must contain `Quantity` and `UnitPrice` columns.")
    st.stop()
if "Revenue" not in df.columns:
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]

has_customer = "CustomerID" in df.columns
has_product = "Description" in df.columns
has_country = "Country" in df.columns
has_date = "InvoiceDate" in df.columns
total_revenue = float(df["Revenue"].fillna(0).sum())
customer_count = int(df["CustomerID"].nunique()) if has_customer else 0
product_count = int(df["Description"].nunique()) if has_product else 0
country_count = int(df["Country"].nunique()) if has_country else 0
transaction_count = len(df)
avg_order_value = total_revenue / max(df["InvoiceNo"].nunique(), 1) if "InvoiceNo" in df.columns else total_revenue / max(transaction_count, 1)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown(
        """<div style="display:flex;align-items:center;gap:12px;margin:8px 0 24px">
        <div class="brand-mark">🛍️</div><div>
        <div class="brand-name">Shopper Spectrum</div>
        <div class="brand-sub">RETAIL INTELLIGENCE</div></div></div>""",
        unsafe_allow_html=True,
    )
    st.markdown('<div style="font-size:.7rem;color:#8490a7;font-weight:800;letter-spacing:1.5px;margin-bottom:8px">WORKSPACE</div>', unsafe_allow_html=True)
    page = st.radio(
        "Navigate",
        [
            "Overview",
            "Analytics",
            "Customer Segments",
            "Product Recommendations",
            "About & Models",
        ],
        label_visibility="collapsed",
        format_func=lambda x: {
            "Overview": "◈   Executive Overview",
            "Analytics": "▦   Analytics Studio",
            "Customer Segments": "◎   Customer Intelligence",
            "Product Recommendations": "✧   Recommendation Studio",
            "About & Models": "ⓘ   About & Models",
        }[x],
    )
    st.markdown("---")
    st.markdown('<div style="font-size:.7rem;color:#8490a7;font-weight:800;letter-spacing:1.5px;margin-bottom:8px">DATA STATUS</div>', unsafe_allow_html=True)
    st.markdown(f"**{transaction_count:,}** rows loaded")
    st.markdown(f"<span style='color:#9ca7bc;font-size:.78rem'>{customer_count:,} unique customers · {product_count:,} products</span>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="small-muted">SHOPPER SPECTRUM · v2.0</div>', unsafe_allow_html=True)

# ---------- Overview ----------
if page == "Overview":
    st.markdown(
        """<div class="hero"><div class="hero-kicker">Retail performance workspace</div>
        <div class="hero-title">Know your customers.<br>Grow every basket.</div>
        <div class="hero-copy">A unified view of sales performance, customer value and product affinity.
        Explore the signals behind your retail data and turn them into better customer experiences.</div></div>""",
        unsafe_allow_html=True,
    )
    st.markdown("#### Your business at a glance")
    metric_cols = st.columns(4, gap="medium")
    with metric_cols[0]:
        card_metric("Total revenue", money(total_revenue), "Gross line-item revenue in dataset", "£")
    with metric_cols[1]:
        card_metric("Transactions", f"{transaction_count:,}", "Retail records loaded", "↗")
    with metric_cols[2]:
        card_metric("Unique customers", f"{customer_count:,}", "Identified customer accounts", "◎")
    with metric_cols[3]:
        card_metric("Products", f"{product_count:,}", f"{country_count:,} countries represented", "▧")
    st.write("")
    left, right = st.columns([1.45, 1], gap="large")
    with left:
        st.markdown('<div class="section-heading">Revenue momentum</div><div class="section-caption">Monthly sales performance over the available period</div>', unsafe_allow_html=True)
        if has_date and df["InvoiceDate"].notna().any():
            trend = df.dropna(subset=["InvoiceDate"]).copy()
            trend["Month"] = trend["InvoiceDate"].dt.to_period("M").astype(str)
            monthly = trend.groupby("Month", as_index=False)["Revenue"].sum().sort_values("Month")
            fig = px.area(monthly, x="Month", y="Revenue", markers=True, color_discrete_sequence=["#6558d9"])
            fig.update_traces(line_width=3, fillcolor="rgba(101,88,217,.12)")
            fig.update_yaxes(title="Revenue (£)", tickprefix="£", tickformat=",.0f")
            fig.update_xaxes(title="")
            st.plotly_chart(style_fig(fig, 350), use_container_width=True)
        else:
            st.info("Add a valid `InvoiceDate` column to view monthly revenue.")
    with right:
        st.markdown('<div class="section-heading">Revenue by market</div><div class="section-caption">Leading countries by total revenue</div>', unsafe_allow_html=True)
        if has_country:
            country = df.groupby("Country", as_index=False)["Revenue"].sum().nlargest(7, "Revenue").sort_values("Revenue")
            fig = px.bar(country, x="Revenue", y="Country", orientation="h", color="Revenue", color_continuous_scale=["#d9d5ff", "#6558d9"])
            fig.update_layout(coloraxis_showscale=False)
            fig.update_xaxes(title="Revenue (£)", tickprefix="£", tickformat=",.0f")
            fig.update_yaxes(title="")
            st.plotly_chart(style_fig(fig, 350), use_container_width=True)
        else:
            st.info("Add a `Country` column to view market performance.")
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.markdown('<div class="section-heading">Best-selling products</div><div class="section-caption">Products with the highest total units sold</div>', unsafe_allow_html=True)
        if has_product:
            best = df.groupby("Description", as_index=False)["Quantity"].sum().nlargest(6, "Quantity").sort_values("Quantity")
            fig = px.bar(best, x="Quantity", y="Description", orientation="h", color_discrete_sequence=["#19a999"])
            fig.update_layout(showlegend=False)
            fig.update_xaxes(title="Units sold")
            fig.update_yaxes(title="", tickfont_size=10)
            st.plotly_chart(style_fig(fig, 315), use_container_width=True)
        else:
            st.info("Add a `Description` column to view product rankings.")
    with right:
        st.markdown('<div class="section-heading">Customer segments</div><div class="section-caption">Composition of the precomputed RFM segments</div>', unsafe_allow_html=True)
        if "Segment" in rfm.columns:
            counts = rfm["Segment"].fillna("Unknown").value_counts().rename_axis("Segment").reset_index(name="Customers")
            fig = px.pie(counts, names="Segment", values="Customers", hole=.68, color_discrete_sequence=PALETTE)
            fig.update_traces(textposition="outside", textinfo="percent", hole=.68)
            fig.update_layout(legend=dict(orientation="h", y=-.12, x=0))
            st.plotly_chart(style_fig(fig, 315), use_container_width=True)
        else:
            st.info("The RFM file needs a `Segment` column for this view.")
    st.markdown('<div class="insight">Tip: open <b>Analytics Studio</b> to filter the dataset, compare customer groups and inspect the underlying distributions.</div>', unsafe_allow_html=True)

# ---------- Analytics ----------
elif page == "Analytics":
    page_header("Explore the numbers", "Analytics Studio", "Filter your retail data and investigate sales, product and customer behaviour.")
    with st.container(border=True):
        f1, f2, f3 = st.columns([1.2, 1, 1])
        filtered = df.copy()
        if has_date and df["InvoiceDate"].notna().any():
            min_date = df["InvoiceDate"].min().date()
            max_date = df["InvoiceDate"].max().date()
            with f1:
                date_range = st.date_input("Invoice date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
            if isinstance(date_range, tuple) and len(date_range) == 2:
                start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
                filtered = filtered[(filtered["InvoiceDate"] >= start) & (filtered["InvoiceDate"] < end)]
        else:
            with f1:
                st.caption("No valid date field found")
        if has_country:
            countries = sorted(df["Country"].dropna().astype(str).unique())
            with f2:
                chosen_countries = st.multiselect("Markets", countries, default=countries, placeholder="Select markets")
            filtered = filtered[filtered["Country"].astype(str).isin(chosen_countries)]
        else:
            with f2:
                st.caption("No country field found")
        if has_product:
            search = ""
            with f3:
                search = st.text_input("Product contains", placeholder="Search product name…")
            if search.strip():
                filtered = filtered[filtered["Description"].astype(str).str.contains(search.strip(), case=False, na=False)]
        else:
            with f3:
                st.caption("No product field found")
    rev = float(filtered["Revenue"].fillna(0).sum())
    orders = int(filtered["InvoiceNo"].nunique()) if "InvoiceNo" in filtered.columns else len(filtered)
    a, b, c, d = st.columns(4)
    a.metric("Filtered revenue", money(rev))
    b.metric("Records", f"{len(filtered):,}")
    c.metric("Orders / invoices", f"{orders:,}")
    d.metric("Average line value", money(rev / max(len(filtered), 1)))
    st.write("")
    tab_sales, tab_products, tab_customers, tab_rfm, tab_segments = st.tabs(["Sales", "Products", "Customers", "RFM analysis", "Segmentation"])
    with tab_sales:
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown("**Revenue by country**")
            if has_country and not filtered.empty:
                country = filtered.groupby("Country", as_index=False)["Revenue"].sum().nlargest(12, "Revenue").sort_values("Revenue")
                fig = px.bar(country, x="Revenue", y="Country", orientation="h", color_discrete_sequence=["#6558d9"])
                fig.update_xaxes(title="Revenue (£)", tickprefix="£", tickformat=",.0f")
                fig.update_yaxes(title="")
                st.plotly_chart(style_fig(fig, 400), use_container_width=True)
            else:
                st.info("No country data for the current filters.")
        with col2:
            st.markdown("**Revenue over time**")
            if has_date and not filtered.empty:
                temp = filtered.dropna(subset=["InvoiceDate"]).copy()
                temp["Month"] = temp["InvoiceDate"].dt.to_period("M").astype(str)
                monthly = temp.groupby("Month", as_index=False)["Revenue"].sum().sort_values("Month")
                fig = px.line(monthly, x="Month", y="Revenue", markers=True, color_discrete_sequence=["#19a999"])
                fig.update_traces(line_width=3, marker_size=6)
                fig.update_yaxes(title="Revenue (£)", tickprefix="£", tickformat=",.0f")
                fig.update_xaxes(title="")
                st.plotly_chart(style_fig(fig, 400), use_container_width=True)
            else:
                st.info("No date data for the current filters.")
        if has_country and not filtered.empty:
            st.markdown("**Revenue contribution**")
            pie_data = filtered.groupby("Country", as_index=False)["Revenue"].sum().nlargest(6, "Revenue")
            fig = px.pie(pie_data, names="Country", values="Revenue", hole=.55, color_discrete_sequence=PALETTE)
            fig.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(style_fig(fig, 360), use_container_width=True)
    with tab_products:
        if has_product and not filtered.empty:
            p1, p2 = st.columns(2, gap="large")
            with p1:
                st.markdown("**Top products by units**")
                top = filtered.groupby("Description", as_index=False)["Quantity"].sum().nlargest(12, "Quantity").sort_values("Quantity")
                fig = px.bar(top, x="Quantity", y="Description", orientation="h", color_discrete_sequence=["#19a999"])
                fig.update_yaxes(title="", tickfont_size=10)
                fig.update_xaxes(title="Units")
                st.plotly_chart(style_fig(fig, 440), use_container_width=True)
            with p2:
                st.markdown("**Unit price distribution**")
                prices = filtered[filtered["UnitPrice"].between(0, filtered["UnitPrice"].quantile(.99))]
                fig = px.histogram(prices, x="UnitPrice", nbins=35, color_discrete_sequence=["#8a7cf0"])
                fig.update_xaxes(title="Unit price (£)", tickprefix="£")
                fig.update_yaxes(title="Line items")
                st.plotly_chart(style_fig(fig, 440), use_container_width=True)
            st.markdown("**Product performance table**")
            product_table = filtered.groupby("Description", as_index=False).agg(Units=("Quantity", "sum"), Revenue=("Revenue", "sum"), Avg_Unit_Price=("UnitPrice", "mean")).sort_values("Revenue", ascending=False)
            st.dataframe(product_table, use_container_width=True, hide_index=True, column_config={"Revenue": st.column_config.NumberColumn(format="£%.2f"), "Avg_Unit_Price": st.column_config.NumberColumn(format="£%.2f")})
        else:
            st.info("No product records match the current filters.")
    with tab_customers:
        if has_customer and not filtered.empty:
            cust = filtered.groupby("CustomerID", as_index=False).agg(Revenue=("Revenue", "sum"), Orders=("InvoiceNo", "nunique") if "InvoiceNo" in filtered.columns else ("Revenue", "size"), Units=("Quantity", "sum"))
            left, right = st.columns(2, gap="large")
            with left:
                st.markdown("**Highest-value customers**")
                topc = cust.nlargest(12, "Revenue").copy()
                topc["CustomerID"] = topc["CustomerID"].astype(str)
                fig = px.bar(topc.sort_values("Revenue"), x="Revenue", y="CustomerID", orientation="h", color_discrete_sequence=["#6558d9"])
                fig.update_xaxes(title="Revenue (£)", tickprefix="£", tickformat=",.0f")
                fig.update_yaxes(title="")
                st.plotly_chart(style_fig(fig, 430), use_container_width=True)
            with right:
                st.markdown("**Customer revenue distribution**")
                dist = cust[cust["Revenue"] <= cust["Revenue"].quantile(.99)]
                fig = px.histogram(dist, x="Revenue", nbins=35, color_discrete_sequence=["#f3a64a"])
                fig.update_xaxes(title="Customer revenue (£)", tickprefix="£")
                fig.update_yaxes(title="Customers")
                st.plotly_chart(style_fig(fig, 430), use_container_width=True)
            st.markdown("**Customer leaderboard**")
            st.dataframe(cust.sort_values("Revenue", ascending=False), use_container_width=True, hide_index=True, column_config={"CustomerID": st.column_config.TextColumn(), "Revenue": st.column_config.NumberColumn(format="£%.2f")})
        else:
            st.info("Customer analysis requires a `CustomerID` column.")
    with tab_rfm:
        if {"Recency", "Frequency", "Monetary"}.issubset(rfm.columns):
            cols = st.columns(3)
            for col, metric, color in zip(cols, ["Recency", "Frequency", "Monetary"], ["#6558d9", "#19a999", "#f3a64a"]):
                with col:
                    fig = px.histogram(rfm, x=metric, nbins=30, color_discrete_sequence=[color])
                    fig.update_layout(title=metric)
                    st.plotly_chart(style_fig(fig, 290), use_container_width=True)
            st.markdown("**RFM data preview**")
            st.dataframe(rfm.head(100), use_container_width=True, hide_index=True)
        else:
            st.warning("The RFM file must contain Recency, Frequency and Monetary columns.")
    with tab_segments:
        if "Segment" in rfm.columns:
            counts = rfm["Segment"].fillna("Unknown").value_counts().rename_axis("Segment").reset_index(name="Customers")
            left, right = st.columns(2, gap="large")
            with left:
                fig = px.bar(counts, x="Segment", y="Customers", color="Segment", color_discrete_sequence=PALETTE)
                fig.update_layout(showlegend=False)
                fig.update_xaxes(title="")
                st.plotly_chart(style_fig(fig, 360), use_container_width=True)
            with right:
                if {"Frequency", "Monetary"}.issubset(rfm.columns):
                    fig = px.scatter(rfm, x="Frequency", y="Monetary", color="Segment", hover_data=["Recency"] if "Recency" in rfm.columns else None, color_discrete_sequence=PALETTE, opacity=.78)
                    fig.update_xaxes(title="Purchase frequency")
                    fig.update_yaxes(title="Monetary value (£)")
                    st.plotly_chart(style_fig(fig, 360), use_container_width=True)
                else:
                    st.info("Frequency and Monetary columns are needed for the scatter plot.")
            if {"Recency", "Frequency", "Monetary"}.issubset(rfm.columns):
                st.markdown("**Explore in 3D**")
                fig = px.scatter_3d(rfm, x="Recency", y="Frequency", z="Monetary", color="Segment", opacity=.75, color_discrete_sequence=PALETTE)
                fig.update_layout(height=550, margin=dict(l=0, r=0, t=40, b=0), paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("The RFM file needs a `Segment` column.")

# ---------- Customer intelligence ----------
elif page == "Customer Segments":
    page_header("Understand customer value", "Customer Intelligence", "Profile a customer with RFM signals and estimate the segment using your trained K-Means model.")
    if models is None:
        st.error("The prediction models could not be loaded.")
        st.code(model_error)
        st.stop()
    left, right = st.columns([1, 1.2], gap="large")
    with left:
        st.markdown('<div class="section-heading">Customer profile</div><div class="section-caption">Enter the customer's RFM behaviour</div>', unsafe_allow_html=True)
        with st.container(border=True):
            recency = st.number_input("Recency · days since last purchase", min_value=0, max_value=10000, value=30, step=1, help="Lower recency usually means a more recent purchase.")
            frequency = st.number_input("Frequency · number of purchases", min_value=1, max_value=100000, value=5, step=1)
            monetary = st.number_input("Monetary · total spend (£)", min_value=0.0, max_value=100000000.0, value=1000.0, step=50.0)
            st.caption("Use the same units and feature order used when training your scaler and clustering model.")
            predict = st.button("✦  Analyze customer segment", type="primary", use_container_width=True)
    with right:
        st.markdown('<div class="section-heading">Segment analysis</div><div class="section-caption">Your model prediction and suggested next steps</div>', unsafe_allow_html=True)
        if predict:
            try:
                sample = pd.DataFrame([[recency, frequency, monetary]], columns=["Recency", "Frequency", "Monetary"])
                scaled = models["scaler"].transform(sample)
                cluster = int(models["kmeans"].predict(scaled)[0])
                labels = models["labels"]
                segment = labels.get(cluster, labels.get(str(cluster), f"Cluster {cluster}")) if isinstance(labels, dict) else labels[cluster]
                st.session_state["last_segment_result"] = {"segment": str(segment), "cluster": cluster, "recency": recency, "frequency": frequency, "monetary": monetary}
            except Exception as exc:
                st.error("Prediction failed. Confirm that the model was trained on Recency, Frequency and Monetary in that order.")
                st.exception(exc)
        result = st.session_state.get("last_segment_result")
        if result:
            st.markdown(
                f"""<div class="hero" style="padding:25px 26px;margin-bottom:16px">
                <div class="hero-kicker">Model classification · cluster {result['cluster']}</div>
                <div class="hero-title" style="font-size:1.8rem">{result['segment']}</div>
                <div class="hero-copy">Prediction generated from the supplied recency, frequency and monetary values.</div></div>""",
                unsafe_allow_html=True,
            )
            x, y, z = st.columns(3)
            x.metric("Recency", f"{result['recency']:,.0f} days")
            y.metric("Frequency", f"{result['frequency']:,.0f}")
            z.metric("Monetary", money(result["monetary"]))
            segment_text = str(result["segment"]).lower()
            if "loyal" in segment_text or "champion" in segment_text:
                advice = "Reward repeat behaviour with loyalty benefits, early access and relevant cross-sell offers."
            elif "at risk" in segment_text or "risk" in segment_text or "hibernat" in segment_text:
                advice = "Consider a win-back campaign, a relevant reminder and a carefully targeted incentive."
            elif "new" in segment_text or "recent" in segment_text:
                advice = "Build the relationship with onboarding messages, product education and a second-purchase offer."
            elif "big" in segment_text or "value" in segment_text:
                advice = "Protect the relationship with personalised service and recommendations matched to prior purchases."
            else:
                advice = "Test a relevant offer or product recommendation, then monitor repeat purchase and spend."
            st.markdown(f'<div class="insight"><b>Suggested action</b><br>{advice}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="surface-card"><div class="metric-label">READY FOR ANALYSIS</div><div class="section-heading" style="margin-top:12px">Your customer profile will appear here</div><p class="small-muted">Enter RFM values and run the model to see the assigned cluster and a practical next step.</p></div>', unsafe_allow_html=True)
    st.markdown("")
    if {"Recency", "Frequency", "Monetary"}.issubset(rfm.columns):
        st.markdown('<div class="section-heading">Reference: your existing customer segments</div><div class="section-caption">Descriptive statistics calculated from rfm_clusters.csv</div>', unsafe_allow_html=True)
        if "Segment" in rfm.columns:
            summary = rfm.groupby("Segment", dropna=False).agg(Customers=("Segment", "size"), Avg_Recency=("Recency", "mean"), Avg_Frequency=("Frequency", "mean"), Avg_Monetary=("Monetary", "mean")).reset_index()
            st.dataframe(summary, use_container_width=True, hide_index=True, column_config={"Avg_Recency": st.column_config.NumberColumn("Avg. recency (days)", format="%.1f"), "Avg_Frequency": st.column_config.NumberColumn("Avg. frequency", format="%.1f"), "Avg_Monetary": st.column_config.NumberColumn("Avg. monetary (£)", format="£%.2f")})

# ---------- Recommendations ----------
elif page == "Product Recommendations":
    page_header("Personalise the next purchase", "Recommendation Studio", "Explore related products using the item-based recommendation mapping already prepared for your project.")
    if models is None:
        st.error("The recommendation files could not be loaded.")
        st.code(model_error)
        st.stop()
    product_names = models["products"]
    if isinstance(product_names, dict):
        product_options = sorted(map(str, product_names.keys()))
    else:
        product_options = sorted(map(str, list(product_names)))
    recommendation_dict = models["recommendations"]
    top_left, top_right = st.columns([1.2, 1], gap="large")
    with top_left:
        st.markdown('<div class="section-heading">Choose a product</div><div class="section-caption">Search the catalogue and generate a related-product shortlist</div>', unsafe_allow_html=True)
        selected = st.selectbox("Product catalogue", product_options, index=None, placeholder="Type a product name…", label_visibility="collapsed")
        count = st.slider("Number of recommendations", min_value=3, max_value=12, value=5)
        run_recs = st.button("✦  Generate recommendations", type="primary", use_container_width=True)
    with top_right:
        st.markdown('<div class="surface-card"><div class="eyebrow">HOW IT WORKS</div><div class="section-heading">Item-based recommendations</div><p class="small-muted">The app looks up the selected product in your saved recommendation dictionary and displays the related items in ranked order. Results depend on the mapping generated during model preparation.</p><div class="pill">Existing model · No retraining</div></div>', unsafe_allow_html=True)
    if run_recs:
        if not selected:
            st.warning("Choose a product first.")
        else:
            candidates = recommendation_dict.get(selected, recommendation_dict.get(str(selected), []))
            if candidates is None:
                candidates = []
            if isinstance(candidates, str):
                candidates = [candidates]
            st.session_state["last_recommendations"] = {"product": selected, "items": list(candidates)[:count]}
    result = st.session_state.get("last_recommendations")
    if result:
        st.markdown("")
        st.markdown(f'<div class="section-heading">Products related to your selection</div><div class="section-caption">Selected product: <b>{result["product"]}</b></div>', unsafe_allow_html=True)
        items = result["items"]
        if not items:
            st.info("No recommendations were found for this product in the saved mapping.")
        else:
            cols = st.columns(min(4, len(items)), gap="medium")
            for idx, item in enumerate(items):
                with cols[idx % len(cols)]:
                    st.markdown(
                        f"""<div class="rec-card"><div class="rec-number">RECOMMENDATION {idx + 1:02d}</div>
                        <div style="font-size:1.7rem;margin:.65rem 0">✧</div>
                        <div class="rec-name">{str(item)}</div>
                        <div class="small-muted">Related item from your saved recommendation model</div></div>""",
                        unsafe_allow_html=True,
                    )
            st.download_button(
                "Download recommendation list (CSV)",
                data=pd.DataFrame({"Rank": range(1, len(items) + 1), "Selected product": [result["product"]] * len(items), "Recommended product": items}).to_csv(index=False).encode("utf-8"),
                file_name="shopper_spectrum_recommendations.csv",
                mime="text/csv",
            )
    else:
        st.markdown('<div class="surface-card"><div style="font-size:2rem">✧</div><div class="section-heading">Your shortlist will appear here</div><div class="small-muted">Select an item from the catalogue and generate recommendations to view the ranked results.</div></div>', unsafe_allow_html=True)

# ---------- About ----------
elif page == "About & Models":
    page_header("Inside the solution", "About Shopper Spectrum", "A compact overview of the analytical methods and assets powering this application.")
    st.markdown(
        """<div class="hero"><div class="hero-kicker">Customer-centric retail analytics</div>
        <div class="hero-title">From transaction data<br>to useful decisions.</div>
        <div class="hero-copy">Shopper Spectrum combines descriptive analytics, RFM customer profiling, unsupervised clustering and item-based product recommendations in one workspace.</div></div>""",
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<div class="surface-card"><div class="eyebrow">CUSTOMER ANALYTICS</div><div class="section-heading">RFM + K-Means</div><p class="small-muted">Recency measures how recently a customer purchased. Frequency measures purchase activity. Monetary measures spend. The saved scaler transforms these features before the K-Means model assigns a cluster.</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="surface-card" style="margin-top:14px"><div class="eyebrow">BUSINESS VALUE</div><div class="section-heading">Where this can help</div><p class="small-muted">Customer retention · loyalty campaigns · product discovery · market comparisons · sales and inventory planning.</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="surface-card"><div class="eyebrow">RECOMMENDATION ENGINE</div><div class="section-heading">Item-based filtering</div><p class="small-muted">The application reads the precomputed recommendation dictionary. The recommendation method and quality depend on how that mapping was created during model development.</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="surface-card" style="margin-top:14px"><div class="eyebrow">PROJECT ASSETS</div><div class="section-heading">Data & model files</div><p class="small-muted"><b>online_retail.csv</b> — transaction data<br><b>rfm_clusters.csv</b> — customer RFM values and segment labels<br><b>models/kmeans_model.pkl</b> — clustering model<br><b>models/scaler.pkl</b> — feature scaler<br><b>models/cluster_labels.pkl</b> — cluster names<br><b>models/recommendations.pkl</b> — recommendation mapping<br><b>models/product_names.pkl</b> — product catalogue</p></div>', unsafe_allow_html=True)
    st.markdown("")
    with st.expander("Technical notes and assumptions"):
        st.markdown(
            "- Revenue is calculated as `Quantity × UnitPrice` at transaction-line level.\n"
            "- Segment predictions expect features in this order: `Recency`, `Frequency`, `Monetary`.\n"
            "- The prediction form assumes monetary values use the same currency and preprocessing conventions as model training.\n"
            "- Pickle files should only be loaded when they come from a trusted source.\n"
            "- This interface improves presentation and interaction; it does not retrain or independently validate the supplied models."
        )

st.markdown(
    '<div style="text-align:center;color:#9aa4b5;font-size:.75rem;padding:26px 0 0">SHOPPER SPECTRUM · RETAIL INTELLIGENCE WORKSPACE</div>',
    unsafe_allow_html=True,
)
