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

# First split: 70% train, 30% temp
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df["risk_level"])

# Second split: 10% eval, 20% val (from 30% temp)
eval_df, val_df = train_test_split(temp_df, test_size=2/3, random_state=42, stratify=temp_df["risk_level"])

# Save
train_df.to_csv(os.path.join(TRAIN_DIR, "train.csv"), index=False)
eval_df.to_csv(os.path.join(EVAL_DIR, "eval.csv"), index=False)
val_df.to_csv(os.path.join(VAL_DIR, "val.csv"), index=False)

print("✅ Dataset successfully split and saved:")
print(f"- Train: {len(train_df)} samples → {TRAIN_DIR}")
print(f"- Eval:  {len(eval_df)} samples → {EVAL_DIR}")
print(f"- Val:   {len(val_df)} samples → {VAL_DIR}")
