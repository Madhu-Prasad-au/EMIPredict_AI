# ============================================
# EMIPredict AI - DATA UTILITIES
# ============================================

import os
import pandas as pd


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "emi_prediction_cleaned.csv"
)


def load_data():

    return pd.read_csv(
        DATA_PATH,
        low_memory=False
    )


def save_data(df):

    df.to_csv(
        DATA_PATH,
        index=False
    )


def add_record(record):

    df = load_data()

    new_record = pd.DataFrame(
        [record]
    )

    df = pd.concat(
        [df, new_record],
        ignore_index=True
    )

    save_data(df)

    return df


def update_record(index, record):

    df = load_data()

    for column, value in record.items():

        if column in df.columns:
            df.loc[index, column] = value

    save_data(df)

    return df


def delete_record(index):

    df = load_data()

    if index in df.index:

        df = df.drop(index)

        df = df.reset_index(
            drop=True
        )

        save_data(df)

    return df