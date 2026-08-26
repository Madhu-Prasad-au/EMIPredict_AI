# ============================================
# EMIPredict AI - MODEL UTILITIES
# ============================================

import os
import joblib
import requests
import streamlit as st


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)


# ============================================================
# HUGGING FACE MODEL URL
# ============================================================

REGRESSION_MODEL_URL = (
    "https://huggingface.co/"
    "M-Prasad/EMIPredict_AI/"
    "resolve/main/"
    "best_regression_model_compressed.pkl"
    "?download=true"
)


# ============================================================
# CLASSIFICATION MODEL
# ============================================================

@st.cache_resource
def load_classification_model():

    path = os.path.join(
        MODEL_DIR,
        "best_classification_model.pkl"
    )

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Classification model not found:\n{path}"
        )

    return joblib.load(path)


# ============================================================
# REGRESSION MODEL
# ============================================================

@st.cache_resource
def load_regression_model():

    # Local model path
    local_path = os.path.join(
        MODEL_DIR,
        "best_regression_model_compressed.pkl"
    )

    # --------------------------------------------------------
    # LOCAL DEVELOPMENT
    # --------------------------------------------------------

    if os.path.exists(local_path):

        return joblib.load(local_path)


    # --------------------------------------------------------
    # STREAMLIT CLOUD
    # --------------------------------------------------------

    st.info(
        "⏳ Downloading regression model..."
    )

    try:

        os.makedirs(
            MODEL_DIR,
            exist_ok=True
        )

        response = requests.get(
            REGRESSION_MODEL_URL,
            stream=True,
            timeout=900
        )

        response.raise_for_status()

        total_size = int(
            response.headers.get(
                "content-length",
                0
            )
        )

        downloaded = 0

        with open(
            local_path,
            "wb"
        ) as file:

            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):

                if chunk:

                    file.write(chunk)

                    downloaded += len(chunk)


        st.success(
            "✅ Regression model downloaded successfully."
        )


    except Exception as e:

        raise RuntimeError(
            f"Failed to download regression model: {e}"
        )


    return joblib.load(
        local_path
    )


# ============================================================
# LABEL ENCODER
# ============================================================

@st.cache_resource
def load_label_encoder():

    path = os.path.join(
        MODEL_DIR,
        "classification_label_encoder.pkl"
    )

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Label encoder not found:\n{path}"
        )

    return joblib.load(
        path
    )


# ============================================================
# FEATURE INFORMATION
# ============================================================

@st.cache_resource
def load_feature_info():

    path = os.path.join(
        MODEL_DIR,
        "feature_info.pkl"
    )

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Feature information not found:\n{path}"
        )

    return joblib.load(
        path
    )