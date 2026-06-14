import pandas as pd

df = pd.read_csv("../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Convert TotalCharges
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df.dropna(inplace=True)

# Remove ID column
df.drop("customerID", axis=1, inplace=True)

# Convert categorical columns
df_encoded = pd.get_dummies(
    df,
    drop_first=True
)

print("Rows:", df_encoded.shape[0])
print("Columns:", df_encoded.shape[1])

print(df_encoded.head())