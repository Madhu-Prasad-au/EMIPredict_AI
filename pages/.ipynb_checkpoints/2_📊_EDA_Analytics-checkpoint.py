import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.data_utils import load_data


st.set_page_config(
    page_title="EDA & Analytics",
    page_icon="📊",
    layout="wide"
)


@st.cache_data
def get_data():

    return load_data()


df = get_data()


st.title(
    "📊 EDA & Financial Analytics"
)

st.write(
    "Interactive exploration of the EMI financial dataset."
)


# ============================================
# FILTERS
# ============================================

st.sidebar.header(
    "Filters"
)


scenario_filter = st.sidebar.multiselect(
    "EMI Scenario",
    options=sorted(
        df["emi_scenario"].unique()
    ),
    default=sorted(
        df["emi_scenario"].unique()
    )
)


eligibility_filter = st.sidebar.multiselect(
    "Eligibility",
    options=sorted(
        df["emi_eligibility"].unique()
    ),
    default=sorted(
        df["emi_eligibility"].unique()
    )
)


filtered_df = df[
    df["emi_scenario"].isin(
        scenario_filter
    )
    &
    df["emi_eligibility"].isin(
        eligibility_filter
    )
]


# ============================================
# SUMMARY
# ============================================

st.header(
    "Dataset Summary"
)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Records",
        f"{len(filtered_df):,}"
    )

with c2:
    st.metric(
        "Average Salary",
        f"₹{filtered_df['monthly_salary'].mean():,.0f}"
    )

with c3:
    st.metric(
        "Average Credit Score",
        f"{filtered_df['credit_score'].mean():,.0f}"
    )

with c4:
    st.metric(
        "Average Max EMI",
        f"₹{filtered_df['max_monthly_emi'].mean():,.0f}"
    )


# ============================================
# ELIGIBILITY
# ============================================

st.divider()

st.header(
    "EMI Eligibility Distribution"
)

eligibility_counts = (
    filtered_df[
        "emi_eligibility"
    ]
    .value_counts()
)


st.bar_chart(
    eligibility_counts
)


# ============================================
# SCENARIO
# ============================================

st.header(
    "EMI Scenario Distribution"
)

scenario_counts = (
    filtered_df[
        "emi_scenario"
    ]
    .value_counts()
)


st.bar_chart(
    scenario_counts
)


# ============================================
# CREDIT SCORE
# ============================================

st.header(
    "Credit Score Analysis"
)

fig, ax = plt.subplots(
    figsize=(10, 5)
)

sns.histplot(
    data=filtered_df,
    x="credit_score",
    hue="emi_eligibility",
    kde=True,
    ax=ax
)

ax.set_title(
    "Credit Score vs EMI Eligibility"
)

st.pyplot(fig)


# ============================================
# SALARY VS MAX EMI
# ============================================

st.header(
    "Income vs Maximum EMI"
)

fig, ax = plt.subplots(
    figsize=(10, 6)
)

sns.scatterplot(
    data=filtered_df.sample(
        min(10000, len(filtered_df)),
        random_state=42
    ),
    x="monthly_salary",
    y="max_monthly_emi",
    hue="emi_eligibility",
    alpha=0.5,
    ax=ax
)

ax.set_xlabel(
    "Monthly Salary"
)

ax.set_ylabel(
    "Maximum Monthly EMI"
)

st.pyplot(fig)


# ============================================
# DATA TABLE
# ============================================

st.divider()

st.header(
    "Filtered Data"
)

st.dataframe(
    filtered_df.head(1000),
    use_container_width=True
)