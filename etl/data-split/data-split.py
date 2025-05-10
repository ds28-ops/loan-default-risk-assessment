import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Paths
INPUT_CSV = "/mnt/data/raw/accepted_2007_to_2018Q4.csv"
OUTPUT_DIR = "/mnt/object/loan-default-data/"
TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
EVAL_DIR = os.path.join(OUTPUT_DIR, "eval")
VAL_DIR = os.path.join(OUTPUT_DIR, "val")

CHUNK_SIZE = 10000

# Create directories
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

# Step 1: Load and process chunks
chunks = []
for chunk in pd.read_csv(INPUT_CSV, chunksize=CHUNK_SIZE, low_memory=False):
    if 'loan_status' not in chunk.columns:
        continue

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

# Step 3: Drop columns with >40% missing
threshold = len(df) * 0.4
na_cols = df.isnull().sum()
to_drop = na_cols[na_cols > threshold].index.tolist()
df.drop(columns=to_drop, inplace=True)

# Step 4: Stratified split
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["risk_level"])
eval_df, val_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["risk_level"])

# Step 5: Save
train_df.to_csv(os.path.join(TRAIN_DIR, "train.csv"), index=False)
eval_df.to_csv(os.path.join(EVAL_DIR, "eval.csv"), index=False)
val_df.to_csv(os.path.join(VAL_DIR, "val.csv"), index=False)

print("✅ Dataset successfully chunked, cleaned, labeled, and split:")
print(f"- Dropped columns with >40% NaNs: {to_drop}")
print(f"- Train: {len(train_df)} samples → {TRAIN_DIR}")
print(f"- Eval:  {len(eval_df)} samples → {EVAL_DIR}")
print(f"- Val:   {len(val_df)} samples → {VAL_DIR}")
