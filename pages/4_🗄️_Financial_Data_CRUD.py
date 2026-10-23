import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Financial Data CRUD",
    page_icon="🗄️",
    layout="wide"
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
# PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

os.makedirs(
    DATA_DIR,
    exist_ok=True
)

CRUD_FILE = os.path.join(
    DATA_DIR,
    "financial_records.csv"
)


# ============================================================
# DEFAULT COLUMNS
# ============================================================

DEFAULT_COLUMNS = [
    "customer_id",
    "age",
    "gender",
    "marital_status",
    "education",
    "monthly_salary",
    "employment_type",
    "years_of_employment",
    "company_type",
    "house_type",
    "monthly_rent",
    "family_size",
    "dependents",
    "school_fees",
    "college_fees",
    "travel_expenses",
    "groceries_utilities",
    "other_monthly_expenses",
    "existing_loans",
    "current_emi_amount",
    "credit_score",
    "bank_balance",
    "emergency_fund",
    "emi_scenario",
    "requested_amount",
    "requested_tenure",
    "created_at"
]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if os.path.exists(CRUD_FILE):

        try:

            df = pd.read_csv(
                CRUD_FILE
            )

            # Ensure expected columns exist

            for column in DEFAULT_COLUMNS:

                if column not in df.columns:

                    df[column] = np.nan

            return df[
                DEFAULT_COLUMNS
            ]

        except Exception:

            return pd.DataFrame(
                columns=DEFAULT_COLUMNS
            )

    return pd.DataFrame(
        columns=DEFAULT_COLUMNS
    )


# ============================================================
# SAVE DATA
# ============================================================

def save_data(df):

    df.to_csv(
        CRUD_FILE,
        index=False
    )


# ============================================================
# CREATE EMPTY DATASET
# ============================================================

if not os.path.exists(CRUD_FILE):

    empty_df = pd.DataFrame(
        columns=DEFAULT_COLUMNS
    )

    save_data(
        empty_df
    )


# ============================================================
# LOAD CURRENT DATA
# ============================================================

df = load_data()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero"><div class="eyebrow">Administration · Record operations</div>
    <div class="hero-title">Financial data management</div>
    <div class="hero-copy">Create, search, update, export, and remove customer financial records from the application's CSV-backed administrative store.</div></div>
    """, unsafe_allow_html=True,
)


# ============================================================
# DATABASE INFORMATION
# ============================================================

st.sidebar.header(
    "📁 Data Storage"
)

st.sidebar.write(
    "Storage file:"
)

st.sidebar.code(
    CRUD_FILE
)

st.sidebar.metric(
    "Total Records",
    len(df)
)


# ============================================================
# CRUD TABS
# ============================================================

tab_create, tab_read, tab_update, tab_delete = (
    st.tabs(
        [
            "➕ Create",
            "📖 Read",
            "✏️ Update",
            "🗑️ Delete"
        ]
    )
)


# ============================================================
# CREATE
# ============================================================

with tab_create:

    st.header(
        "➕ Add Financial Record"
    )

    st.write(
        "Enter customer financial information."
    )

    # --------------------------------------------------------
    # CUSTOMER INFORMATION
    # --------------------------------------------------------

    st.subheader(
        "👤 Customer Information"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        create_age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=30,
            key="crud_create_age"
        )

        create_gender = st.selectbox(
            "Gender",
            [
                "Male",
                "Female"
            ],
            key="crud_create_gender"
        )

        create_marital = st.selectbox(
            "Marital Status",
            [
                "Single",
                "Married"
            ],
            key="crud_create_marital"
        )

    with col2:

        create_education = st.selectbox(
            "Education",
            [
                "High School",
                "Graduate",
                "Post Graduate",
                "Professional"
            ],
            key="crud_create_education"
        )

        create_employment = st.selectbox(
            "Employment Type",
            [
                "Private",
                "Government",
                "Self-employed"
            ],
            key="crud_create_employment"
        )

        create_company = st.selectbox(
            "Company Type",
            [
                "Small",
                "Medium",
                "Large"
            ],
            key="crud_create_company"
        )

    with col3:

        create_years = st.number_input(
            "Years of Employment",
            min_value=0.0,
            max_value=50.0,
            value=5.0,
            key="crud_create_years"
        )

        create_house = st.selectbox(
            "House Type",
            [
                "Rented",
                "Own",
                "Family"
            ],
            key="crud_create_house"
        )

        create_family = st.number_input(
            "Family Size",
            min_value=1,
            max_value=20,
            value=4,
            key="crud_create_family"
        )


    # --------------------------------------------------------
    # INCOME
    # --------------------------------------------------------

    st.subheader(
        "💵 Income & Expenses"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        create_salary = st.number_input(
            "Monthly Salary (₹)",
            min_value=0.0,
            value=50000.0,
            step=1000.0,
            key="crud_create_salary"
        )

        create_rent = st.number_input(
            "Monthly Rent (₹)",
            min_value=0.0,
            value=10000.0,
            step=500.0,
            key="crud_create_rent"
        )

        create_dependents = st.number_input(
            "Dependents",
            min_value=0,
            max_value=20,
            value=2,
            key="crud_create_dependents"
        )

    with col2:

        create_school = st.number_input(
            "School Fees (₹)",
            min_value=0.0,
            value=3000.0,
            step=500.0,
            key="crud_create_school"
        )

        create_college = st.number_input(
            "College Fees (₹)",
            min_value=0.0,
            value=0.0,
            step=500.0,
            key="crud_create_college"
        )

        create_travel = st.number_input(
            "Travel Expenses (₹)",
            min_value=0.0,
            value=3000.0,
            step=500.0,
            key="crud_create_travel"
        )

    with col3:

        create_groceries = st.number_input(
            "Groceries & Utilities (₹)",
            min_value=0.0,
            value=8000.0,
            step=500.0,
            key="crud_create_groceries"
        )

        create_other = st.number_input(
            "Other Monthly Expenses (₹)",
            min_value=0.0,
            value=3000.0,
            step=500.0,
            key="crud_create_other"
        )


    # --------------------------------------------------------
    # CREDIT
    # --------------------------------------------------------

    st.subheader(
        "🏦 Credit Information"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        create_loans = st.selectbox(
            "Existing Loans",
            [
                "Yes",
                "No"
            ],
            key="crud_create_loans"
        )

        create_current_emi = st.number_input(
            "Current EMI (₹)",
            min_value=0.0,
            value=5000.0,
            step=500.0,
            key="crud_create_current_emi"
        )

    with col2:

        create_credit = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=850,
            value=700,
            key="crud_create_credit"
        )

        create_balance = st.number_input(
            "Bank Balance (₹)",
            min_value=0.0,
            value=100000.0,
            step=5000.0,
            key="crud_create_balance"
        )

    with col3:

        create_emergency = st.number_input(
            "Emergency Fund (₹)",
            min_value=0.0,
            value=50000.0,
            step=5000.0,
            key="crud_create_emergency"
        )


    # --------------------------------------------------------
    # LOAN
    # --------------------------------------------------------

    st.subheader(
        "💳 Loan Information"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        create_scenario = st.selectbox(
            "EMI Scenario",
            [
                "E-commerce Shopping EMI",
                "Home Appliances EMI",
                "Vehicle EMI",
                "Personal Loan EMI",
                "Education EMI"
            ],
            key="crud_create_scenario"
        )

    with col2:

        create_amount = st.number_input(
            "Requested Amount (₹)",
            min_value=1000.0,
            value=100000.0,
            step=5000.0,
            key="crud_create_amount"
        )

    with col3:

        create_tenure = st.number_input(
            "Requested Tenure (Months)",
            min_value=1,
            max_value=120,
            value=24,
            key="crud_create_tenure"
        )


    # --------------------------------------------------------
    # CREATE BUTTON
    # --------------------------------------------------------

    if st.button(
        "➕ Add Record",
        type="primary",
        use_container_width=True,
        key="crud_create_button"
    ):

        # Generate ID

        if len(df) == 0:

            customer_id = 1

        else:

            try:

                customer_id = (
                    pd.to_numeric(
                        df["customer_id"],
                        errors="coerce"
                    )
                    .max()
                    + 1
                )

            except Exception:

                customer_id = (
                    len(df) + 1
                )


        new_record = {

            "customer_id": int(
                customer_id
            ),

            "age": create_age,

            "gender": create_gender,

            "marital_status": create_marital,

            "education": create_education,

            "monthly_salary": create_salary,

            "employment_type": create_employment,

            "years_of_employment": create_years,

            "company_type": create_company,

            "house_type": create_house,

            "monthly_rent": create_rent,

            "family_size": create_family,

            "dependents": create_dependents,

            "school_fees": create_school,

            "college_fees": create_college,

            "travel_expenses": create_travel,

            "groceries_utilities": create_groceries,

            "other_monthly_expenses": create_other,

            "existing_loans": create_loans,

            "current_emi_amount": create_current_emi,

            "credit_score": create_credit,

            "bank_balance": create_balance,

            "emergency_fund": create_emergency,

            "emi_scenario": create_scenario,

            "requested_amount": create_amount,

            "requested_tenure": create_tenure,

            "created_at": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        }


        new_row = pd.DataFrame(
            [new_record]
        )


        df = pd.concat(
            [
                df,
                new_row
            ],
            ignore_index=True
        )


        save_data(df)


        st.success(
            f"✅ Customer #{int(customer_id)} "
            "added successfully!"
        )

        st.rerun()


# ============================================================
# READ
# ============================================================

with tab_read:

    st.header(
        "📖 View Financial Records"
    )


    if df.empty:

        st.info(
            "No financial records available."
        )

    else:

        st.write(
            f"Showing {len(df)} financial records."
        )


        # Search

        search_text = st.text_input(
            "🔍 Search Customer",
            placeholder=(
                "Enter customer ID, "
                "gender, employment type..."
            ),
            key="crud_read_search"
        )


        filtered_df = df

        if search_text.strip():
            searchable_columns = [
                c for c in [
                    "customer_id", "gender", "marital_status", "education",
                    "employment_type", "company_type", "house_type",
                    "existing_loans", "emi_scenario"
                ] if c in filtered_df.columns
            ]
            text_mask = filtered_df[searchable_columns].astype("string").apply(
                lambda col: col.str.contains(search_text.strip(), case=False, na=False, regex=False)
            ).any(axis=1) if searchable_columns else pd.Series(False, index=filtered_df.index)
            numeric_mask = pd.Series(False, index=filtered_df.index)
            if "customer_id" in filtered_df.columns and search_text.strip().isdigit():
                numeric_mask = pd.to_numeric(filtered_df["customer_id"], errors="coerce").eq(int(search_text.strip()))
            filtered_df = filtered_df.loc[text_mask | numeric_mask]


        st.dataframe(
            filtered_df,
            use_container_width=True,
            height=500
        )


        st.download_button(
            "⬇️ Download CSV",
            data=filtered_df.to_csv(
                index=False
            ),
            file_name="financial_records.csv",
            mime="text/csv",
            key="crud_download_csv"
        )


# ============================================================
# UPDATE
# ============================================================

with tab_update:

    st.header(
        "✏️ Update Financial Record"
    )


    if df.empty:

        st.info(
            "No records available to update."
        )

    else:

        # --------------------------------------------
        # Select customer
        # --------------------------------------------

        customer_options = (
            df[
                "customer_id"
            ]
            .astype(str)
            .tolist()
        )


        selected_customer = st.selectbox(
            "Select Customer ID",
            customer_options,
            key="crud_update_customer"
        )


        selected_id = int(
            selected_customer
        )


        selected_rows = df[
            df["customer_id"]
            == selected_id
        ]


        if not selected_rows.empty:

            row = selected_rows.iloc[0]


            st.subheader(
                "Update Financial Information"
            )


            col1, col2, col3 = (
                st.columns(3)
            )


            with col1:

                update_salary = st.number_input(
                    "Monthly Salary (₹)",
                    min_value=0.0,
                    value=float(
                        row["monthly_salary"]
                    ),
                    step=1000.0,
                    key="crud_update_salary"
                )


                update_rent = st.number_input(
                    "Monthly Rent (₹)",
                    min_value=0.0,
                    value=float(
                        row["monthly_rent"]
                    ),
                    step=500.0,
                    key="crud_update_rent"
                )


                update_credit = st.number_input(
                    "Credit Score",
                    min_value=300,
                    max_value=850,
                    value=int(
                        row["credit_score"]
                    ),
                    key="crud_update_credit"
                )


            with col2:

                update_emi = st.number_input(
                    "Current EMI (₹)",
                    min_value=0.0,
                    value=float(
                        row[
                            "current_emi_amount"
                        ]
                    ),
                    step=500.0,
                    key="crud_update_emi"
                )


                update_balance = st.number_input(
                    "Bank Balance (₹)",
                    min_value=0.0,
                    value=float(
                        row["bank_balance"]
                    ),
                    step=5000.0,
                    key="crud_update_balance"
                )


                update_emergency = st.number_input(
                    "Emergency Fund (₹)",
                    min_value=0.0,
                    value=float(
                        row["emergency_fund"]
                    ),
                    step=5000.0,
                    key="crud_update_emergency"
                )


            with col3:

                update_amount = st.number_input(
                    "Requested Amount (₹)",
                    min_value=1000.0,
                    value=float(
                        row[
                            "requested_amount"
                        ]
                    ),
                    step=5000.0,
                    key="crud_update_amount"
                )


                update_tenure = st.number_input(
                    "Requested Tenure",
                    min_value=1,
                    max_value=120,
                    value=int(
                        row[
                            "requested_tenure"
                        ]
                    ),
                    key="crud_update_tenure"
                )


                update_dependents = st.number_input(
                    "Dependents",
                    min_value=0,
                    max_value=20,
                    value=int(
                        row["dependents"]
                    ),
                    key="crud_update_dependents"
                )


            update_employment = st.selectbox(
                "Employment Type",
                [
                    "Private",
                    "Government",
                    "Self-employed"
                ],
                index=(
                    [
                        "Private",
                        "Government",
                        "Self-employed"
                    ].index(
                        row["employment_type"]
                    )
                    if row[
                        "employment_type"
                    ]
                    in [
                        "Private",
                        "Government",
                        "Self-employed"
                    ]
                    else 0
                ),
                key="crud_update_employment"
            )


            update_loans = st.selectbox(
                "Existing Loans",
                [
                    "Yes",
                    "No"
                ],
                index=(
                    0
                    if row[
                        "existing_loans"
                    ] == "Yes"
                    else 1
                ),
                key="crud_update_loans"
            )


            if st.button(
                "💾 Update Record",
                type="primary",
                use_container_width=True,
                key="crud_update_button"
            ):

                row_index = (
                    df.index[
                        df[
                            "customer_id"
                        ]
                        == selected_id
                    ][0]
                )


                df.loc[
                    row_index,
                    "monthly_salary"
                ] = update_salary


                df.loc[
                    row_index,
                    "monthly_rent"
                ] = update_rent


                df.loc[
                    row_index,
                    "credit_score"
                ] = update_credit


                df.loc[
                    row_index,
                    "current_emi_amount"
                ] = update_emi


                df.loc[
                    row_index,
                    "bank_balance"
                ] = update_balance


                df.loc[
                    row_index,
                    "emergency_fund"
                ] = update_emergency


                df.loc[
                    row_index,
                    "requested_amount"
                ] = update_amount


                df.loc[
                    row_index,
                    "requested_tenure"
                ] = update_tenure


                df.loc[
                    row_index,
                    "dependents"
                ] = update_dependents


                df.loc[
                    row_index,
                    "employment_type"
                ] = update_employment


                df.loc[
                    row_index,
                    "existing_loans"
                ] = update_loans


                save_data(df)


                st.success(
                    f"✅ Customer #{selected_id} "
                    "updated successfully!"
                )

                st.rerun()


# ============================================================
# DELETE
# ============================================================

with tab_delete:

    st.header(
        "🗑️ Delete Financial Record"
    )


    if df.empty:

        st.info(
            "No records available to delete."
        )

    else:

        st.warning(
            "⚠️ Deleted records cannot be recovered."
        )


        # ====================================================
        # DELETE BY CUSTOMER ID
        # ====================================================

        customer_options = (
            df[
                "customer_id"
            ]
            .astype(int)
            .tolist()
        )


        delete_customer_id = st.selectbox(
            "Select Customer ID to Delete",
            customer_options,
            key="crud_delete_customer_id"
        )


        # ====================================================
        # DELETE BY TABLE INDEX
        # ====================================================

        st.markdown(
            "### Or delete using table index"
        )


        delete_index = st.number_input(
            "Record Index",
            min_value=0,
            max_value=max(
                len(df) - 1,
                0
            ),
            value=0,
            step=1,
            key="crud_delete_index"
        )


        # ====================================================
        # DELETE BUTTON
        # ====================================================

        if st.button(
            "🗑️ Delete Selected Record",
            type="primary",
            use_container_width=True,
            key="crud_delete_button"
        ):

            # --------------------------------------------
            # Determine selected customer
            # --------------------------------------------

            try:

                selected_id = int(
                    delete_customer_id
                )

            except Exception:

                selected_id = None


            # --------------------------------------------
            # Delete confirmation
            # --------------------------------------------

            if selected_id is not None:

                matching_rows = df[
                    df[
                        "customer_id"
                    ]
                    == selected_id
                ]


                if matching_rows.empty:

                    st.error(
                        "Customer record not found."
                    )

                else:

                    df = df[
                        df[
                            "customer_id"
                        ]
                        != selected_id
                    ].reset_index(
                        drop=True
                    )


                    save_data(df)


                    st.success(
                        f"🗑️ Customer #{selected_id} "
                        "deleted successfully!"
                    )


                    st.rerun()


# ============================================================
# DATASET STATISTICS
# ============================================================

st.divider()

st.header(
    "📊 Dataset Statistics"
)


current_df = load_data()


stat1, stat2, stat3, stat4 = (
    st.columns(4)
)


with stat1:

    st.metric(
        "Total Records",
        len(current_df)
    )


with stat2:

    if not current_df.empty:

        avg_salary = pd.to_numeric(
            current_df[
                "monthly_salary"
            ],
            errors="coerce"
        ).mean()

        st.metric(
            "Average Salary",
            f"₹{avg_salary:,.0f}"
        )

    else:

        st.metric(
            "Average Salary",
            "₹0"
        )


with stat3:

    if not current_df.empty:

        avg_credit = pd.to_numeric(
            current_df[
                "credit_score"
            ],
            errors="coerce"
        ).mean()

        st.metric(
            "Average Credit Score",
            f"{avg_credit:.0f}"
        )

    else:

        st.metric(
            "Average Credit Score",
            "0"
        )


with stat4:

    if not current_df.empty:

        avg_loan = pd.to_numeric(
            current_df[
                "requested_amount"
            ],
            errors="coerce"
        ).mean()

        st.metric(
            "Average Loan Amount",
            f"₹{avg_loan:,.0f}"
        )

    else:

        st.metric(
            "Average Loan Amount",
            "₹0"
        )
