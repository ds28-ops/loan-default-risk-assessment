import pandas as pd
import os

RAW_PATH = "/mnt/data/raw/loan_data.parquet"
OUTPUT_BASE = "/mnt/object/loan-default-data-test"
OUTPUT_TRAIN = os.path.join(OUTPUT_BASE, "train", "train_clean.parquet")
OUTPUT_VAL = os.path.join(OUTPUT_BASE, "val", "val_clean.parquet")
OUTPUT_EVAL = os.path.join(OUTPUT_BASE, "eval", "eval_clean.parquet")

# Create output directories
os.makedirs(os.path.join(OUTPUT_BASE, "train"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_BASE, "val"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_BASE, "eval"), exist_ok=True)

def map_risk_level(status):
    if status in ["Fully Paid", "Current"]:
        return "Low"
    elif status in ["Late (16-30 days)", "In Grace Period"]:
        return "Medium"
    else:
        return "High"

# Load full Parquet dataset
df = pd.read_parquet(RAW_PATH)

# Drop rows without loan_status
df = df.dropna(subset=['loan_status'])

# Map to risk_level
df['risk_level'] = df['loan_status'].apply(map_risk_level)

# Drop original target
df.drop(columns=['loan_status'], inplace=True)

# Drop columns with >40% missing
threshold = len(df) * 0.4
df.dropna(thresh=threshold, axis=1, inplace=True)

# Shuffle
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

# Split 80-10-10
train_end = int(0.8 * len(df))
val_end = int(0.9 * len(df))

# Save as Parquet
df.iloc[:train_end].to_parquet(OUTPUT_TRAIN, index=False)
df.iloc[train_end:val_end].to_parquet(OUTPUT_VAL, index=False)
df.iloc[val_end:].to_parquet(OUTPUT_EVAL, index=False)

print("Minimal ETL complete.")
