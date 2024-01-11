"""
Author: Przemysław Lipka
Project: Diabetic Macular Oedema Treatment Response Prediction
Date: 11 January 2024

This code is the intellectual property of Przemysław Lipka. It is designed
for the purpose of research and analysis in a non-commercial setting.
Unauthorized use, distribution, or replication for commercial purposes is
strictly prohibited.

"""

from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from datetime import datetime

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

model_data = np.load(
    "data/training/model_data_stratified_10_percent_test_2024-01-11_12-20-45.npz"
)

X_train = model_data["X_train"]
X_test = model_data["X_test"]
y_train = model_data["y_train"]
y_test = model_data["y_test"]

# Initialize the Logistic Regression model
logreg = LogisticRegression()

# Train the model
logreg.fit(X_train, y_train)

# Predict on the test set
y_pred = logreg.predict(X_test)

print(classification_report(y_test, y_pred))

np.savez(
    Path("data/predictions") / f"baseline_predictions_{timestamp}.npz",
    X_test=X_test,
    y_test=y_test,
    y_pred=y_pred,
)
