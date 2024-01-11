"""
Author: Przemysław Lipka
Project: Diabetic Macular Oedema Treatment Response Prediction
Date: 11 January 2024

This code is the intellectual property of Przemysław Lipka. It is designed
for the purpose of research and analysis in a non-commercial setting.
Unauthorized use, distribution, or replication for commercial purposes is
strictly prohibited.

Encode the categorical variables and split the data into training and testing sets.
In general tree estimator can handle missing values, but for neural network approach we 
need to encode those, as for simpler ML techniques.

This pipeline step requires all previous steps to be executed before this one.

"""
from pathlib import Path
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from datetime import datetime

import pandas as pd
import fire

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _create_preprocessor(X):
    """
    Create a preprocessor for encoding and scaling the features in the input data.

    Parameters:
    X (pandas.DataFrame): The input data.

    Returns:
    preprocessor (ColumnTransformer): The preprocessor object that can be used to transform the input data.
    """

    categorical_cols = X.select_dtypes(include=["object", "bool"]).columns
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns

    categorical_transformer = OneHotEncoder()
    numerical_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    return preprocessor


def _transform_features(preprocessor, X_train, X_test):
    """
    Apply the preprocessor transformation to the training and test data.

    Args:
        preprocessor: The preprocessor object used for feature transformation.
        X_train: The training data.
        X_test: The test data.

    Returns:
        X_train_transformed: The transformed training data.
        X_test_transformed: The transformed test data.
    """

    pipeline = Pipeline(steps=[("preprocessor", preprocessor)])
    X_train_transformed = pipeline.fit_transform(X_train)
    X_test_transformed = pipeline.transform(X_test)

    # Store preprocessor for inference
    with open(f"data/training/preprocessor_{timestamp}.pkl", "wb") as f:
        pickle.dump(pipeline.named_steps["preprocessor"], f)

    X_test_transformed_df = pd.DataFrame(
        X_test_transformed,
        columns=pipeline.named_steps["preprocessor"].get_feature_names_out(),
    )
    X_test_transformed_df.to_csv(f"data/training/test_columns_{timestamp}.csv")

    return X_train_transformed, X_test_transformed


def main():
    df = pd.read_csv("data/processed/processed_data_2024-01-11_10-35-21.csv")

    # Define the target variable and features
    X = df.drop("Response", axis=1)
    y = df["Response"]

    # This is really small dataset, so make every element count
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=31337, stratify=y
    )

    preprocessor = _create_preprocessor(X)

    X_train_transformed, X_test_transformed = _transform_features(
        preprocessor, X_train, X_test
    )

    np.savez(
        Path("data/training")
        / f"model_data_stratified_10_percent_test_{timestamp}.npz",
        X_train=X_train_transformed,
        X_test=X_test_transformed,
        y_train=y_train,
        y_test=y_test,
    )


if __name__ == "__main__":
    fire.Fire(main)
