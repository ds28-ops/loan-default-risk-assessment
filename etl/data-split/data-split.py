import os
import csv
import pandas as pd
from sklearn.model_selection import train_test_split

# Paths
INPUT_CSV = "/mnt/data/raw/accepted_2007_to_2018Q4.csv"
OUTPUT_DIR = "/mnt/object/loan-default-data-test/"
INTERMEDIATE_CSV = os.path.join(OUTPUT_DIR, "cleaned_streamed.csv")
TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
EVAL_DIR = os.path.join(OUTPUT_DIR, "eval")
VAL_DIR = os.path.join(OUTPUT_DIR, "val")
CHUNK_SIZE = 50000

# Ensure output dirs exist
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

# Map loan_status to risk_level
def map_risk_level(status):
    if status in ["Fully Paid", "Current"]:
        return "Low"
    elif status in ["Late (16-30 days)", "In Grace Period"]:
        return "Medium"
    else:
        return "High"

# Step 1: Load and process chunks
chunks = []
i = 0
for chunk in pd.read_csv(INPUT_CSV, chunksize=CHUNK_SIZE, low_memory=False):
    if 'loan_status' not in chunk.columns:
        continue
    print("chunk", i, " is done!!!")    
    i+=1    
    # Drop rows with missing loan_status
    chunk = chunk.dropna(subset=['loan_status'])

    # Map risk level
    chunk['risk_level'] = chunk['loan_status'].apply(map_risk_level)
    chunk.drop(columns=['loan_status'], inplace=True)

    chunks.append(chunk)

# Step 2: Concatenate all chunks
if not chunks:
    raise ValueError("No valid data found in chunks.")

df = pd.concat(chunks, ignore_index=True)

# Stratified 80-10-10 split
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["risk_level"])
eval_df, val_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["risk_level"])

# Save final splits
train_df.to_csv(os.path.join(TRAIN_DIR, "train.csv"), index=False)
eval_df.to_csv(os.path.join(EVAL_DIR, "eval.csv"), index=False)
val_df.to_csv(os.path.join(VAL_DIR, "val.csv"), index=False)

print("✅ Dataset successfully cleaned, labeled, and split:")
print(f"- Total: {len(df)} rows")
print(f"- Train: {len(train_df)} → {TRAIN_DIR}")
print(f"- Eval : {len(eval_df)} → {EVAL_DIR}")
print(f"- Val  : {len(val_df)} → {VAL_DIR}")
