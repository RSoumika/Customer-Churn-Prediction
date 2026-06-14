import pandas as pd

df = pd.read_csv("../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Convert TotalCharges
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Remove missing values
df.dropna(inplace=True)

# Remove customerID
df.drop("customerID", axis=1, inplace=True)

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print(df.head())