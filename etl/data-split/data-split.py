import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Paths
INPUT_CSV = "/mnt/data/raw/accepted_2007_to_2018Q4.csv"
OUTPUT_DIR = "/mnt/object/loan-default-data-sharded/"
TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
EVAL_DIR = os.path.join(OUTPUT_DIR, "eval")
VAL_DIR = os.path.join(OUTPUT_DIR, "val")
CHUNK_SIZE = 50000

# Ensure output dirs exist
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

# Risk level mapping
def map_risk_level(status):
    if status in ["Fully Paid", "Current"]:
        return "Low"
    elif status in ["Late (16-30 days)", "In Grace Period"]:
        return "Medium"
    else:
        return "High"

# Process and split each chunk into train/eval/val
for i, chunk in enumerate(pd.read_csv(INPUT_CSV, chunksize=CHUNK_SIZE, low_memory=False)):
    if 'loan_status' not in chunk.columns:
        continue

    print(f"✅ Processing chunk {i}...")

    # Filter invalid rows
    chunk = chunk[chunk['loan_status'].notna()]
    chunk['risk_level'] = chunk['loan_status'].apply(map_risk_level)
    chunk.drop(columns=['loan_status'], inplace=True)
    chunk.dropna(inplace=True)

    # Drop if chunk is too small to split
    if chunk.shape[0] < 100:
        print(f"⚠️ Skipping chunk {i} — too few rows.")
        continue

    # Stratified split for this chunk
    try:
        train_df, temp_df = train_test_split(chunk, test_size=0.2, random_state=42, stratify=chunk['risk_level'])
        eval_df, val_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['risk_level'])
    except ValueError as e:
        print(f"⚠️ Skipping chunk {i} — stratification failed: {e}")
        continue

    # Save to numbered files
    train_df.to_csv(os.path.join(TRAIN_DIR, f"train_{i}.csv"), index=False)
    eval_df.to_csv(os.path.join(EVAL_DIR, f"eval_{i}.csv"), index=False)
    val_df.to_csv(os.path.join(VAL_DIR, f"val_{i}.csv"), index=False)

print("✅ All chunks processed and split into separate shard files.")
