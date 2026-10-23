# ============================================================
# EMIPredict AI
# PHASE 7 — STREAMLIT APPLICATION
# PAGE 1 — EMI PREDICTION
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np

from utils.model_utils import (
    load_classification_model,
    load_regression_model,
    load_label_encoder,
    load_feature_info
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EMI Prediction",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

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



# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0px;
    }

    .sub-title {
        font-size: 18px;
        color: #777;
        margin-bottom: 25px;
    }

    .result-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #ddd;
        margin-bottom: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    classification_model = (
        load_classification_model()
    )

    regression_model = (
        load_regression_model()
    )

    label_encoder = (
        load_label_encoder()
    )

    feature_info = (
        load_feature_info()
    )

    return (
        classification_model,
        regression_model,
        label_encoder,
        feature_info
    )


# ============================================================
# TRY LOADING MODELS
# ============================================================

try:

    (
        classification_model,
        regression_model,
        label_encoder,
        feature_info
    ) = load_models()

except Exception as e:

    st.error(
        "❌ Unable to load the trained models."
    )

    st.exception(e)

    st.stop()


# ============================================================
# FEATURE INFORMATION
# ============================================================

try:

    classification_features = (
        feature_info[
            "classification_features"
        ]
    )

    regression_features = (
        feature_info[
            "regression_features"
        ]
    )

except Exception as e:

    st.error(
        "❌ Feature information could not be loaded."
    )

    st.exception(e)

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">EMIPredict AI · Decision intelligence</div>
      <div class="hero-title">EMI Eligibility & Affordability</div>
      <div class="hero-copy">Evaluate a customer's eligibility and estimate a responsible monthly EMI limit using your trained classification and regression models.</div>
    </div>
    """, unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "📌 Prediction Overview"
    )

    st.markdown(
        """
        This application performs two AI predictions:

        **1️⃣ Classification**

        Determines the customer's EMI
        eligibility / risk category.

        **2️⃣ Regression**

        Estimates the maximum monthly EMI
        the customer can afford.
        """
    )

    st.divider()

    st.info(
        """
        **Classification**

        "Is the customer suitable for this EMI?"

        **Regression**

        "How much EMI can the customer afford?"
        """
    )


# ============================================================
# CUSTOMER INFORMATION
# ============================================================

st.header(
    "👤 Customer Information"
)

col1, col2, col3 = st.columns(3)


with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30,
        step=1
    )

    gender = st.selectbox(
        "Gender",
        [
            "Male",
            "Female"
        ]
    )

    marital_status = st.selectbox(
        "Marital Status",
        [
            "Single",
            "Married"
        ]
    )


with col2:

    education = st.selectbox(
        "Education",
        [
            "High School",
            "Graduate",
            "Post Graduate",
            "Professional"
        ]
    )

    employment_type = st.selectbox(
        "Employment Type",
        [
            "Private",
            "Government",
            "Self-employed"
        ]
    )

    company_type = st.selectbox(
        "Company Type",
        [
            "Small",
            "Medium",
            "Large"
        ]
    )


with col3:

    years_of_employment = st.number_input(
        "Years of Employment",
        min_value=0.0,
        max_value=50.0,
        value=5.0,
        step=0.5
    )

    house_type = st.selectbox(
        "House Type",
        [
            "Rented",
            "Own",
            "Family"
        ]
    )

    family_size = st.number_input(
        "Family Size",
        min_value=1,
        max_value=20,
        value=4,
        step=1
    )


# ============================================================
# INCOME AND EXPENSES
# ============================================================

st.header(
    "💵 Income & Monthly Expenses"
)

col1, col2, col3 = st.columns(3)


with col1:

    monthly_salary = st.number_input(
        "Monthly Salary (₹)",
        min_value=0.0,
        value=50000.0,
        step=1000.0
    )

    monthly_rent = st.number_input(
        "Monthly Rent (₹)",
        min_value=0.0,
        value=10000.0,
        step=500.0
    )

    dependents = st.number_input(
        "Dependents",
        min_value=0,
        max_value=20,
        value=2,
        step=1
    )


with col2:

    school_fees = st.number_input(
        "School Fees (₹)",
        min_value=0.0,
        value=3000.0,
        step=500.0
    )

    college_fees = st.number_input(
        "College Fees (₹)",
        min_value=0.0,
        value=0.0,
        step=500.0
    )

    travel_expenses = st.number_input(
        "Travel Expenses (₹)",
        min_value=0.0,
        value=3000.0,
        step=500.0
    )


with col3:

    groceries_utilities = st.number_input(
        "Groceries & Utilities (₹)",
        min_value=0.0,
        value=8000.0,
        step=500.0
    )

    other_monthly_expenses = st.number_input(
        "Other Monthly Expenses (₹)",
        min_value=0.0,
        value=3000.0,
        step=500.0
    )


# ============================================================
# CREDIT INFORMATION
# ============================================================

st.header(
    "🏦 Credit & Existing Loans"
)

col1, col2, col3 = st.columns(3)


with col1:

    existing_loans = st.selectbox(
        "Existing Loans",
        [
            "Yes",
            "No"
        ]
    )

    current_emi_amount = st.number_input(
        "Current EMI Amount (₹)",
        min_value=0.0,
        value=5000.0,
        step=500.0
    )


with col2:

    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=850,
        value=700,
        step=1
    )

    bank_balance = st.number_input(
        "Bank Balance (₹)",
        min_value=0.0,
        value=100000.0,
        step=5000.0
    )


with col3:

    emergency_fund = st.number_input(
        "Emergency Fund (₹)",
        min_value=0.0,
        value=50000.0,
        step=5000.0
    )


# ============================================================
# LOAN APPLICATION
# ============================================================

st.header(
    "💳 Loan Application"
)

col1, col2, col3 = st.columns(3)


with col1:

    emi_scenario = st.selectbox(
        "EMI Scenario",
        [
            "E-commerce Shopping EMI",
            "Home Appliances EMI",
            "Vehicle EMI",
            "Personal Loan EMI",
            "Education EMI"
        ]
    )


with col2:

    requested_amount = st.number_input(
        "Requested Loan Amount (₹)",
        min_value=1000.0,
        value=100000.0,
        step=5000.0
    )


with col3:

    requested_tenure = st.number_input(
        "Requested Tenure (Months)",
        min_value=1,
        max_value=120,
        value=24,
        step=1
    )


# ============================================================
# FEATURE PREPARATION FUNCTION
# ============================================================

def prepare_model_input(
    input_data,
    expected_features
):

    """
    Prepare input exactly according to the
    features used during model training.
    """

    input_data = input_data.copy()

    provided_features = set(
        input_data.columns
    )

    expected_features_set = set(
        expected_features
    )

    missing_features = (
        expected_features_set
        - provided_features
    )

    extra_features = (
        provided_features
        - expected_features_set
    )

    # ----------------------------------------
    # Missing feature check
    # ----------------------------------------

    if missing_features:

        st.error(
            "❌ Required model features are missing:"
        )

        st.code(
            "\n".join(
                sorted(
                    missing_features
                )
            )
        )

        return None

    # ----------------------------------------
    # Remove extra features
    # ----------------------------------------

    if extra_features:

        input_data = input_data.drop(
            columns=list(
                extra_features
            )
        )

    # ----------------------------------------
    # Exact feature order
    # ----------------------------------------

    input_data = input_data[
        expected_features
    ]

    return input_data


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

predict_button = st.button(
    "🔮 Predict EMI Eligibility & Affordability",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION PROCESS
# ============================================================

if predict_button:

    # ========================================================
    # VALIDATION
    # ========================================================

    if monthly_salary <= 0:

        st.error(
            "Monthly salary must be greater than zero."
        )

        st.stop()


    if requested_amount <= 0:

        st.error(
            "Requested loan amount must be greater than zero."
        )

        st.stop()


    if requested_tenure <= 0:

        st.error(
            "Loan tenure must be greater than zero."
        )

        st.stop()


    # ========================================================
    # RAW INPUT DATA
    # ========================================================

    input_data = pd.DataFrame({

        "age": [age],

        "gender": [gender],

        "marital_status": [
            marital_status
        ],

        "education": [
            education
        ],

        "monthly_salary": [
            monthly_salary
        ],

        "employment_type": [
            employment_type
        ],

        "years_of_employment": [
            years_of_employment
        ],

        "company_type": [
            company_type
        ],

        "house_type": [
            house_type
        ],

        "monthly_rent": [
            monthly_rent
        ],

        "family_size": [
            family_size
        ],

        "dependents": [
            dependents
        ],

        "school_fees": [
            school_fees
        ],

        "college_fees": [
            college_fees
        ],

        "travel_expenses": [
            travel_expenses
        ],

        "groceries_utilities": [
            groceries_utilities
        ],

        "other_monthly_expenses": [
            other_monthly_expenses
        ],

        "existing_loans": [
            existing_loans
        ],

        "current_emi_amount": [
            current_emi_amount
        ],

        "credit_score": [
            credit_score
        ],

        "bank_balance": [
            bank_balance
        ],

        "emergency_fund": [
            emergency_fund
        ],

        "emi_scenario": [
            emi_scenario
        ],

        "requested_amount": [
            requested_amount
        ],

        "requested_tenure": [
            requested_tenure
        ]
    })


    # ========================================================
    # FEATURE ENGINEERING
    # ========================================================

    total_monthly_expenses = (
        monthly_rent
        + school_fees
        + college_fees
        + travel_expenses
        + groceries_utilities
        + other_monthly_expenses
    )


    total_debt_burden = (
        current_emi_amount
        + total_monthly_expenses
    )


    remaining_income = (
        monthly_salary
        - total_debt_burden
    )


    # ========================================================
    # DERIVED FEATURES
    # ========================================================

    input_data[
        "total_monthly_expenses"
    ] = total_monthly_expenses


    input_data[
        "total_debt_burden"
    ] = total_debt_burden


    input_data[
        "remaining_income"
    ] = remaining_income


    # ========================================================
    # RATIO FEATURES
    # ========================================================

    input_data[
        "debt_to_income_ratio"
    ] = (
        total_debt_burden
        / max(
            monthly_salary,
            1
        )
    )


    # IMPORTANT:
    # This feature was required by the trained
    # classification pipeline.

    input_data[
        "affordability_ratio"
    ] = (
        remaining_income
        / max(
            monthly_salary,
            1
        )
    )


    input_data[
        "expense_to_income_ratio"
    ] = (
        total_monthly_expenses
        / max(
            monthly_salary,
            1
        )
    )


    input_data[
        "current_emi_to_income"
    ] = (
        current_emi_amount
        / max(
            monthly_salary,
            1
        )
    )


    input_data[
        "loan_to_salary_ratio"
    ] = (
        requested_amount
        / max(
            monthly_salary,
            1
        )
    )


    input_data[
        "balance_to_salary_ratio"
    ] = (
        bank_balance
        / max(
            monthly_salary,
            1
        )
    )


    input_data[
        "emergency_coverage_ratio"
    ] = (
        emergency_fund
        / max(
            total_monthly_expenses,
            1
        )
    )


    input_data[
        "estimated_requested_emi"
    ] = (
        requested_amount
        / max(
            requested_tenure,
            1
        )
    )


    # ========================================================
    # CLASSIFICATION INPUT
    # ========================================================

    classification_input = (
        prepare_model_input(
            input_data,
            classification_features
        )
    )


    if classification_input is None:

        st.stop()


    # ========================================================
    # REGRESSION INPUT
    # ========================================================

    regression_input = (
        prepare_model_input(
            input_data,
            regression_features
        )
    )


    if regression_input is None:

        st.stop()


    # ========================================================
    # CLASSIFICATION PREDICTION
    # ========================================================

    with st.spinner(
        "Running EMI eligibility classification..."
    ):

        try:

            eligibility_encoded = (
                classification_model.predict(
                    classification_input
                )
            )

            # Convert encoded prediction
            # back to original class

            eligibility = (
                label_encoder.inverse_transform(
                    eligibility_encoded
                )[0]
            )

            # Probability

            if hasattr(
                classification_model,
                "predict_proba"
            ):

                probabilities = (
                    classification_model
                    .predict_proba(
                        classification_input
                    )[0]
                )

            else:

                probabilities = None


        except Exception as e:

            st.error(
                "❌ Classification prediction failed."
            )

            st.exception(e)

            st.stop()


    # ========================================================
    # REGRESSION PREDICTION
    # ========================================================

    with st.spinner(
        "Estimating maximum affordable EMI..."
    ):

        try:

            predicted_max_emi = (
                regression_model.predict(
                    regression_input
                )[0]
            )

            predicted_max_emi = float(
                predicted_max_emi
            )

            # Prevent negative prediction

            predicted_max_emi = max(
                0.0,
                predicted_max_emi
            )

        except Exception as e:

            st.error(
                "❌ Regression prediction failed."
            )

            st.exception(e)

            st.stop()


    # ========================================================
    # REQUESTED EMI
    # ========================================================

    requested_emi = (
        requested_amount
        / max(
            requested_tenure,
            1
        )
    )


    emi_difference = (
        predicted_max_emi
        - requested_emi
    )


    # ========================================================
    # CLASSIFICATION INTERPRETATION
    # ========================================================

    eligibility_text = str(
        eligibility
    )


    normalized_eligibility = (
        eligibility_text
        .lower()
        .replace("_", " ")
        .strip()
    )


    if normalized_eligibility in [
        "eligible",
        "low risk",
        "approved",
        "yes"
    ]:

        classification_good = True

    else:

        classification_good = False


    # ========================================================
    # REGRESSION INTERPRETATION
    # ========================================================

    regression_good = (
        requested_emi
        <= predicted_max_emi
    )


    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    st.divider()

    st.header(
        "📊 Prediction Results"
    )


    # ========================================================
    # CLASSIFICATION + REGRESSION
    # ========================================================

    classification_col, regression_col = (
        st.columns(2)
    )


    # ========================================================
    # CLASSIFICATION RESULT
    # ========================================================

    with classification_col:

        st.subheader(
            "1️⃣ EMI Eligibility Classification"
        )


        if classification_good:

            st.success(
                f"### ✅ {eligibility_text}"
            )

            st.write(
                """
                The classification model predicts
                that the customer has acceptable
                EMI eligibility.
                """
            )

        else:

            st.warning(
                f"### ⚠️ {eligibility_text}"
            )

            st.write(
                """
                The classification model indicates
                that the customer may require
                additional financial review.
                """
            )


        # --------------------------------------------
        # PROBABILITY
        # --------------------------------------------

        if probabilities is not None:

            st.markdown(
                "#### Classification Probability"
            )


            probability_df = pd.DataFrame({

                "Class": (
                    label_encoder.classes_
                ),

                "Probability": (
                    probabilities * 100
                )

            })


            st.dataframe(
                probability_df,
                use_container_width=True,
                hide_index=True
            )


            st.bar_chart(
                probability_df.set_index(
                    "Class"
                )[
                    "Probability"
                ]
            )


    # ========================================================
    # REGRESSION RESULT
    # ========================================================

    with regression_col:

        st.subheader(
            "2️⃣ Maximum Affordable EMI"
        )


        st.metric(
            "Maximum Affordable Monthly EMI",
            f"₹{predicted_max_emi:,.2f}"
        )


        st.metric(
            "Requested Monthly EMI",
            f"₹{requested_emi:,.2f}"
        )


        st.metric(
            "EMI Capacity Difference",
            f"₹{emi_difference:,.2f}"
        )


        if regression_good:

            st.success(
                """
                ✅ Requested EMI is within
                the predicted affordability limit.
                """
            )

        else:

            st.error(
                """
                ❌ Requested EMI exceeds the
                predicted affordability limit.
                """
            )


    # ========================================================
    # COMBINED DECISION
    # ========================================================

    st.divider()

    st.header(
        "🎯 Combined Financial Assessment"
    )


    decision_col1, decision_col2, decision_col3 = (
        st.columns(3)
    )


    # Classification status

    with decision_col1:

        st.markdown(
            "### Classification"
        )

        if classification_good:

            st.success(
                eligibility_text
            )

        else:

            st.warning(
                eligibility_text
            )


    # Regression status

    with decision_col2:

        st.markdown(
            "### EMI Affordability"
        )

        if regression_good:

            st.success(
                "Affordable"
            )

        else:

            st.error(
                "Not Affordable"
            )


    # Final result

    with decision_col3:

        st.markdown(
            "### Final Assessment"
        )


        if (
            classification_good
            and regression_good
        ):

            st.success(
                "✅ RECOMMENDED"
            )

            final_message = (
                "Customer is eligible and "
                "requested EMI is affordable."
            )


        elif (
            classification_good
            and not regression_good
        ):

            st.warning(
                "⚠️ REVIEW REQUIRED"
            )

            final_message = (
                "Customer may be eligible, "
                "but requested EMI exceeds "
                "the predicted affordability."
            )


        else:

            st.error(
                "❌ HIGH RISK"
            )

            final_message = (
                "Customer requires additional "
                "financial assessment."
            )


    st.info(
        final_message
    )


    # ========================================================
    # FINANCIAL SUMMARY
    # ========================================================

    st.divider()

    st.header(
        "💰 Financial Summary"
    )


    c1, c2, c3, c4, c5 = (
        st.columns(5)
    )


    with c1:

        st.metric(
            "Monthly Salary",
            f"₹{monthly_salary:,.0f}"
        )


    with c2:

        st.metric(
            "Existing EMI",
            f"₹{current_emi_amount:,.0f}"
        )


    with c3:

        st.metric(
            "Requested EMI",
            f"₹{requested_emi:,.0f}"
        )


    with c4:

        st.metric(
            "Maximum Affordable EMI",
            f"₹{predicted_max_emi:,.0f}"
        )


    with c5:

        st.metric(
            "Credit Score",
            f"{credit_score}"
        )


    # ========================================================
    # AFFORDABILITY ANALYSIS
    # ========================================================

    st.divider()

    st.header(
        "📈 Affordability Analysis"
    )


    aff_col1, aff_col2, aff_col3 = (
        st.columns(3)
    )


    affordability_ratio = (
        input_data[
            "affordability_ratio"
        ].iloc[0]
    )


    debt_to_income_ratio = (
        input_data[
            "debt_to_income_ratio"
        ].iloc[0]
    )


    expense_to_income_ratio = (
        input_data[
            "expense_to_income_ratio"
        ].iloc[0]
    )


    with aff_col1:

        st.metric(
            "Affordability Ratio",
            f"{affordability_ratio:.2%}"
        )


    with aff_col2:

        st.metric(
            "Debt-to-Income Ratio",
            f"{debt_to_income_ratio:.2%}"
        )


    with aff_col3:

        st.metric(
            "Expense-to-Income Ratio",
            f"{expense_to_income_ratio:.2%}"
        )


    # ========================================================
    # EMI COMPARISON
    # ========================================================

    st.subheader(
        "Requested EMI vs Affordable EMI"
    )


    comparison_df = pd.DataFrame({

        "EMI Type": [
            "Requested EMI",
            "Maximum Affordable EMI"
        ],

        "Amount": [
            requested_emi,
            predicted_max_emi
        ]

    })


    st.bar_chart(
        comparison_df.set_index(
            "EMI Type"
        )
    )


    # ========================================================
    # RECOMMENDATION
    # ========================================================

    st.divider()

    st.header(
        "💡 Recommendation"
    )


    if (
        classification_good
        and regression_good
    ):

        st.success(
            f"""
            The customer is classified as
            **{eligibility_text}**.

            The requested EMI of
            **₹{requested_emi:,.2f}**
            is within the predicted affordable
            EMI of
            **₹{predicted_max_emi:,.2f}**.

            The application can be considered
            financially suitable based on the
            model predictions.
            """
        )


    elif (
        classification_good
        and not regression_good
    ):

        st.warning(
            f"""
            The customer is classified as
            **{eligibility_text}**, but the requested
            EMI of **₹{requested_emi:,.2f}** exceeds
            the predicted affordable EMI of
            **₹{predicted_max_emi:,.2f}**.

            Consider reducing the requested loan
            amount or increasing the repayment tenure.
            """
        )


    else:

        st.error(
            f"""
            The customer is classified as
            **{eligibility_text}**.

            The application should undergo
            additional financial assessment
            before approval.
            """
        )


    # ========================================================
    # MODEL INPUT DEBUG INFORMATION
    # ========================================================

    with st.expander(
        "🔧 View Model Input Details"
    ):

        st.write(
            "### Classification Features"
        )

        st.code(
            "\n".join(
                classification_features
            )
        )


        st.write(
            "### Regression Features"
        )

        st.code(
            "\n".join(
                regression_features
            )
        )


        st.write(
            "### Classification Input"
        )

        st.dataframe(
            classification_input,
            use_container_width=True
        )


        st.write(
            "### Regression Input"
        )

        st.dataframe(
            regression_input,
            use_container_width=True
        )
