import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load data
df = pd.read_csv("../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Cleaning
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df.dropna(inplace=True)

df.drop("customerID", axis=1, inplace=True)

# Encoding
df = pd.get_dummies(
    df,
    drop_first=True
)

# Features and Target
X = df.drop("Churn_Yes", axis=1)
y = df["Churn_Yes"]

# Train model
model = LogisticRegression(
    max_iter=2000,
    solver="liblinear",
    class_weight="balanced"
)

model.fit(X, y)

# Save model
joblib.dump(model, "../models/churn_model.pkl")

# Save feature names
joblib.dump(list(X.columns), "../models/feature_names.pkl")

print("Model saved successfully!")