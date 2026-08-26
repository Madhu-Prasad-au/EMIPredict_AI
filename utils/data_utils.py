# ============================================
# EMIPredict AI - DATA UTILITIES
# ============================================

import os
import requests
import pandas as pd
import streamlit as st


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)


DATA_FILE = os.path.join(
    DATA_DIR,
    "emi_prediction_cleaned.csv"
)


# ============================================================
# HUGGING FACE DATASET URL
# ============================================================

DATASET_URL = (
    "https://huggingface.co/"
    "M-Prasad/EMIPredict_AI/"
    "resolve/main/"
    "emi_prediction_cleaned.csv"
    "?download=true"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    # --------------------------------------------------------
    # LOCAL DEVELOPMENT
    # --------------------------------------------------------

    if os.path.exists(DATA_FILE):

        return pd.read_csv(
            DATA_FILE
        )


    # --------------------------------------------------------
    # STREAMLIT CLOUD
    # --------------------------------------------------------

    st.info(
        "⏳ Downloading EMI prediction dataset..."
    )

    try:

        os.makedirs(
            DATA_DIR,
            exist_ok=True
        )

        response = requests.get(
            DATASET_URL,
            timeout=600
        )

        response.raise_for_status()

        with open(
            DATA_FILE,
            "wb"
        ) as file:

            file.write(
                response.content
            )


    except Exception as e:

        raise RuntimeError(
            f"Failed to download EMI dataset: {e}"
        )


    return pd.read_csv(
        DATA_FILE
    )