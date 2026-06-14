import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Load dataset
df = pd.read_csv("../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Data cleaning
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

# Features
X = df.drop("Churn_Yes", axis=1)

# Target
y = df["Churn_Yes"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = LogisticRegression(
    max_iter=2000,
    solver="liblinear",
    class_weight="balanced"
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy :", round(accuracy_score(y_test, y_pred)*100,2), "%")
print("Precision:", round(precision_score(y_test, y_pred)*100,2), "%")
print("Recall   :", round(recall_score(y_test, y_pred)*100,2), "%")
print("F1 Score :", round(f1_score(y_test, y_pred)*100,2), "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))