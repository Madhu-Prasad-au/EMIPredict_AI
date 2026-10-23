# ============================================
# EMIPredict AI - DATA UTILITIES
# Memory-safer Hugging Face CSV loader
# ============================================

import os
import requests
import pandas as pd
import streamlit as st


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_FILE = os.path.join(
    DATA_DIR,
    "emi_prediction_cleaned.csv"
)

TEMP_FILE = DATA_FILE + ".part"


# ============================================
# HUGGING FACE DATASET URL
# ============================================

DATASET_URL = (
    "https://huggingface.co/"
    "M-Prasad/EMIPredict_AI/"
    "resolve/main/"
    "emi_prediction_cleaned.csv"
    "?download=true"
)


# ============================================
# DOWNLOAD CSV USING STREAMING
# ============================================

def download_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)

    try:
        with requests.get(
            DATASET_URL,
            stream=True,
            timeout=(15, 120)
        ) as response:

            response.raise_for_status()

            with open(TEMP_FILE, "wb") as file:
                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):
                    if chunk:
                        file.write(chunk)

        # Replace the final file only after a successful download.
        if not os.path.exists(TEMP_FILE):
            raise RuntimeError("The downloaded file was not created.")

        if os.path.getsize(TEMP_FILE) == 0:
            raise RuntimeError("The downloaded CSV is empty.")

        os.replace(TEMP_FILE, DATA_FILE)

    except Exception as exc:
        if os.path.exists(TEMP_FILE):
            try:
                os.remove(TEMP_FILE)
            except OSError:
                pass

        raise RuntimeError(
            f"Failed to download EMI dataset from Hugging Face: {exc}"
        ) from exc


# ============================================
# LOAD DATA
# ============================================

@st.cache_data(show_spinner="Loading EMI dataset...")
def load_data():

    # ----------------------------------------
    # USE EXISTING LOCAL OR CACHED CSV FILE
    # ----------------------------------------

    if os.path.exists(DATA_FILE) and os.path.getsize(DATA_FILE) > 0:
        try:
            return pd.read_csv(DATA_FILE)
        except (pd.errors.ParserError, UnicodeDecodeError, ValueError):
            # The existing file may be incomplete or invalid.
            try:
                os.remove(DATA_FILE)
            except OSError:
                pass

    # ----------------------------------------
    # DOWNLOAD FROM HUGGING FACE IF REQUIRED
    # ----------------------------------------

    st.info("Downloading EMI prediction dataset from Hugging Face...")

    download_dataset()

    # ----------------------------------------
    # PARSE CSV
    # ----------------------------------------

    try:
        data = pd.read_csv(DATA_FILE)
    except Exception as exc:
        raise RuntimeError(
            f"The dataset downloaded, but Pandas could not read the CSV: {exc}"
        ) from exc

    if data.empty:
        raise RuntimeError("The EMI prediction dataset contains no rows.")

    return data
