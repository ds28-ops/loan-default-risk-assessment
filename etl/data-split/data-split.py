import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Paths
INPUT_CSV = "/mnt/object/loan-default-data/train_ready.csv"
OUTPUT_DIR = "/mnt/object/loan-default-data/"
TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
EVAL_DIR = os.path.join(OUTPUT_DIR, "eval")
VAL_DIR = os.path.join(OUTPUT_DIR, "val")

# Create directories
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv(INPUT_CSV)

# Drop columns with >40% missing values
threshold = len(df) * 0.4
na_cols = df.isnull().sum()
to_drop = na_cols[na_cols > threshold].index.tolist()
df.drop(columns=to_drop, inplace=True)

# Drop rows with missing 'loan_status'
df.dropna(subset=['loan_status'], inplace=True)

# Map loan_status to risk_level
def map_risk_level(status):
    if status in ["Fully Paid", "Current"]:
        return "Low"
    elif status in ["Late (16-30 days)", "In Grace Period"]:
        return "Medium"
    else:
        return "High"

df["risk_level"] = df["loan_status"].apply(map_risk_level)
df.drop(columns=["loan_status"], inplace=True)

# First split: 80% train, 20% temp
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["risk_level"])

# Second split: 10% eval, 10% val (from 20% temp)
eval_df, val_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["risk_level"])

# Save
train_df.to_csv(os.path.join(TRAIN_DIR, "train.csv"), index=False)
eval_df.to_csv(os.path.join(EVAL_DIR, "eval.csv"), index=False)
val_df.to_csv(os.path.join(VAL_DIR, "val.csv"), index=False)

print("✅ Dataset successfully cleaned, labeled, and split:")
print(f"- Dropped columns with >40% NaNs: {to_drop}")
print(f"- Train: {len(train_df)} samples → {TRAIN_DIR}")
print(f"- Eval:  {len(eval_df)} samples → {EVAL_DIR}")
print(f"- Val:   {len(val_df)} samples → {VAL_DIR}")
