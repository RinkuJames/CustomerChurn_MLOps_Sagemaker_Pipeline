import pandas as pd
import os
from sklearn.model_selection import train_test_split

input_path = "/opt/ml/processing/input/customer_data.csv"

df = pd.read_csv(input_path)

# Drop unnecessary columns
if "customerID" in df.columns:
    df.drop("customerID", axis=1, inplace=True)
    print("Dropped customerID column.")

# Convert target variable churn to 0/1
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

# One-hot encode categorical columns
categorical_cols = df.select_dtypes(include=["object"]).columns
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# -----------------------------
# Handle Missing Values
# -----------------------------

# Fill numeric with median
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
for col in numeric_cols:
    df[col].fillna(df[col].median(), inplace=True)

# Fill categorical with mode
for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)
print("Missing values handled.")

# -----------------------------
# Outlier Handling (Clipping)
# -----------------------------
for col in numeric_cols:
    q1 = df[col].quantile(0.01)
    q99 = df[col].quantile(0.99)
    df[col] = np.clip(df[col], q1, q99)

print("Outliers clipped.")

train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["Churn"]
)

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# Create output dirs
os.makedirs("/opt/ml/processing/train", exist_ok=True)
os.makedirs("/opt/ml/processing/test", exist_ok=True)

# Save Outputs as csv files
train_df.to_csv("/opt/ml/processing/train/train.csv", index=False)
test_df.to_csv("/opt/ml/processing/test/test.csv", index=False)

print("Preprocessing completed successfully.")
