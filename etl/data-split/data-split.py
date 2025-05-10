import pandas as pd
import os

RAW_PATH = "/mnt/data/raw/accepted_2007_to_2018Q4.csv"
OUTPUT_BASE = "/mnt/object/loan-default-data-test"
OUTPUT_TRAIN = os.path.join(OUTPUT_BASE, "train", "train_clean.csv")
OUTPUT_VAL = os.path.join(OUTPUT_BASE, "val", "val_clean.csv")
OUTPUT_EVAL = os.path.join(OUTPUT_BASE, "eval", "eval_clean.csv")
CHUNK_SIZE = 10000

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

df_list = []

# Read and process chunks
for chunk in pd.read_csv(RAW_PATH, chunksize=CHUNK_SIZE, low_memory=False):
    if 'loan_status' not in chunk.columns:
        continue
    chunk = chunk.dropna(subset=['loan_status'])
    chunk['risk_level'] = chunk['loan_status'].apply(map_risk_level)
    chunk.drop(columns=['loan_status'], inplace=True)
    df_list.append(chunk)

# Combine all chunks
if not df_list:
    raise ValueError("No valid data found.")
df = pd.concat(df_list, ignore_index=True)

# Drop columns with >40% missing
threshold = len(df) * 0.4
df.dropna(thresh=threshold, axis=1, inplace=True)

# Shuffle
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

# Split 80-10-10
train_end = int(0.8 * len(df))
val_end = int(0.9 * len(df))

df.iloc[:train_end].to_csv(OUTPUT_TRAIN, index=False)
df.iloc[train_end:val_end].to_csv(OUTPUT_VAL, index=False)
df.iloc[val_end:].to_csv(OUTPUT_EVAL, index=False)

print("Minimal ETL complete.")
