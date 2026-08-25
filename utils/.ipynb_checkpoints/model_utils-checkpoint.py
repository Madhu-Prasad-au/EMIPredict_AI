# ============================================
# EMIPredict AI - MODEL UTILITIES
# ============================================

import os
import joblib


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)


def load_classification_model():

    path = os.path.join(
        MODEL_DIR,
        "best_classification_model.pkl"
    )

    return joblib.load(path)


def load_regression_model():

    path = os.path.join(
        MODEL_DIR,
        "best_regression_model.pkl"
    )

    return joblib.load(path)


def load_label_encoder():

    path = os.path.join(
        MODEL_DIR,
        "classification_label_encoder.pkl"
    )

    return joblib.load(path)


def load_feature_info():

    path = os.path.join(
        MODEL_DIR,
        "feature_info.pkl"
    )

    return joblib.load(path)