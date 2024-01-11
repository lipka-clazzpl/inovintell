"""
Author: Przemysław Lipka
Project: Diabetic Macular Oedema Treatment Response Prediction
Date: 11 January 2024

This code is the intellectual property of Przemysław Lipka. It is designed
for the purpose of research and analysis in a non-commercial setting.
Unauthorized use, distribution, or replication for commercial purposes is
strictly prohibited.

Additional preprocessing of the merged dataset

This pipeline step will 
    - limit columns to the eye being treated
    - limit columns to some extent (for the demo purpose - not enough time to normalize, etc)
    - shuffle 

This pipeline step requires all previous steps to be executed before this one.

"""

from datetime import datetime
from pathlib import Path
import pandas as pd
import fire


def main():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    df = pd.read_csv("data/merged/merged_data_2024-01-11_10-12-58.csv")

    eye_treatment_columns = [
        "DMETrtHx",
        "DMETrtFocalLasHx",
        "DMETrtIVTHx",
        "DMETrtPBTHx",
        "DMETrtVitHx",
        "DMETrtVEGFHx",
    ]

    df = _drop_non_treatment_eye_columns(df, eye_treatment_columns)

    # Shrink categorical values space to minimum
    df = _normalize_categorical_data(df)
    df = _normalize_numerical_data(df)

    # This is arbitrary choice to speed up development
    colunns_to_keep = [
        "Gender",
        "AgeAsOfEnrollDt",
        "Ethnicity",
        "Race",
        "DiabAge",
        "DiabetesType",
        "InsulinUsed",
        "InsTrtAge",
        "PtCurrMed",
        "PreExistMedCond",
        "TTrtGroup",
        "Response",
    ]

    colunns_to_keep += eye_treatment_columns

    df = df[colunns_to_keep]

    df_to_save = df.sample(frac=1).reset_index(drop=True)

    df_to_save.info()

    filename_with_timestamp = f"processed_data_{timestamp}.csv"

    df_to_save.to_csv(Path("data/processed") / filename_with_timestamp, index=False)


def _normalize_numerical_data(df):
    """
    Normalize numerical data in the given DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing the numerical data.

    Returns:
        pandas.DataFrame: The DataFrame with normalized numerical data.
    """

    columns_to_normalize = ["InsTrtAge"]

    for column in columns_to_normalize:
        df[column] = df[column].apply(
            lambda x: 0 if pd.isna(pd.to_numeric(x, errors="coerce")) else int(x)
        )

    return df


def _drop_non_treatment_eye_columns(df, eye_treatment_columns) -> pd.DataFrame:
    """
    Drops non-treatment eye columns from the DataFrame and returns the modified DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        eye_treatment_columns (list): List of column names representing treatment eye columns.

    Returns:
        pd.DataFrame: The modified DataFrame with non-treatment eye columns dropped.
    """

    for eye_treatment_column in eye_treatment_columns:
        df[eye_treatment_column] = df.apply(
            lambda x: x[f"{eye_treatment_column}OD"]
            if x["StudyEye"] == "OD"
            else x[f"{eye_treatment_column}OS"],
            axis=1,
        )

    return df


def _normalize_categorical_data(df) -> pd.DataFrame:
    """
    Normalize categorical data in the given DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing the categorical data.
        columns_to_normalize (list): List of column names to normalize.

    Returns:
        pandas.DataFrame: The DataFrame with normalized categorical data.
    """
    boolean_encoded_columns = [
        "DMETrtHx",
        "DMETrtFocalLasHx",
        "DMETrtIVTHx",
        "DMETrtPBTHx",
        "DMETrtVitHx",
        "DMETrtVEGFHx",
    ]

    yes_no_encoded_columns = ["PtCurrMed", "PreExistMedCond", "InsulinUsed"]

    for column in boolean_encoded_columns:
        df[column] = df[column].apply(lambda x: True if x == 1 else False)

    for column in yes_no_encoded_columns:
        df[column] = df[column].apply(lambda x: True if x.lower() == "yes" else False)

    return df


if __name__ == "__main__":
    fire.Fire(main)
