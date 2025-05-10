import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder

RAW_PATH = "/mnt/data/raw/accepted_2007_to_2018Q4.csv"
OUTPUT_BASE = "/mnt/object/loan-default-data"
OUTPUT_CLEAN = os.path.join(OUTPUT_BASE, "cleaned_data.csv")
OUTPUT_TRAIN = os.path.join(OUTPUT_BASE, "train", "train_clean.csv")
OUTPUT_VAL = os.path.join(OUTPUT_BASE, "val", "val_clean.csv")
OUTPUT_EVAL = os.path.join(OUTPUT_BASE, "eval", "eval_clean.csv")
LOG_FILE = os.path.join(OUTPUT_BASE, "etl_log.txt")
CHUNK_SIZE = 10000

os.makedirs(os.path.join(OUTPUT_BASE, "train"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_BASE, "val"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_BASE, "eval"), exist_ok=True)

LEAKAGE_COLS = [
    'id', 'member_id', 'url', 'desc', 'title', 'zip_code', 'addr_state',
    'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp',
    'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee',
    'last_pymnt_d', 'next_pymnt_d', 'last_credit_pull_d', 'last_pymnt_amnt',
    'debt_settlement_flag', 'debt_settlement_flag_date', 'settlement_status',
    'settlement_date', 'settlement_amount', 'settlement_percentage', 'settlement_term',
    'hardship_flag', 'hardship_type', 'hardship_reason', 'hardship_status', 'deferral_term',
    'hardship_amount', 'hardship_start_date', 'hardship_end_date',
    'payment_plan_start_date', 'hardship_length', 'hardship_dpd',
    'hardship_loan_status', 'orig_projected_additional_accrued_interest',
    'hardship_payoff_balance_amount', 'hardship_last_payment_amount',
    'emp_title'  # high cardinality
]

def map_risk_level(status):
    if status in ["Fully Paid", "Current"]:
        return "Low"
    elif status in ["Late (16-30 days)", "In Grace Period"]:
        return "Medium"
    else:
        return "High"

log_msgs = []
df_list = []

# Read and process chunks
for i, chunk in enumerate(pd.read_csv(RAW_PATH, chunksize=CHUNK_SIZE, low_memory=False)):
    original_cols = set(chunk.columns)

    # Drop leakage and identifier columns
    drop_cols = [col for col in LEAKAGE_COLS if col in chunk.columns]
    chunk.drop(columns=drop_cols, inplace=True)

    # Drop rows with missing loan_status
    if 'loan_status' not in chunk.columns:
        continue
    chunk = chunk.dropna(subset=['loan_status'])

    # Map risk_level
    chunk['risk_level'] = chunk['loan_status'].apply(map_risk_level)
    chunk.drop(columns=['loan_status'], inplace=True)

    df_list.append(chunk)

# Combine all chunks
if not df_list:
    raise ValueError("No valid chunks processed.")
df = pd.concat(df_list, ignore_index=True)

# Drop columns with >40% missing
threshold = len(df) * 0.4
na_cols = df.isnull().sum()
to_drop = na_cols[na_cols > threshold].index.tolist()
df.drop(columns=to_drop, inplace=True)
log_msgs.append(f"Dropped columns with >40% missing: {to_drop}")

# Handle remaining NaNs
for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype in ['float64', 'int64']:
            median = df[col].median()
            df[col].fillna(median, inplace=True)
        elif df[col].dtype == 'object':
            mode = df[col].mode()
            if not mode.empty:
                df[col].fillna(mode[0], inplace=True)
            else:
                df[col].fillna("Unknown", inplace=True)

# Encode categorical variables
for col in df.select_dtypes(include='object').columns:
    if df[col].nunique() <= 30:
        df = pd.get_dummies(df, columns=[col], prefix=col)
    else:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

# Shuffle
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

# Save full cleaned CSV
df.to_csv(OUTPUT_CLEAN, index=False)

# Split 70-15-15
train_end = int(0.7 * len(df))
val_end = int(0.85 * len(df))

df.iloc[:train_end].to_csv(OUTPUT_TRAIN, index=False)
df.iloc[train_end:val_end].to_csv(OUTPUT_VAL, index=False)
df.iloc[val_end:].to_csv(OUTPUT_EVAL, index=False)

# Save log
with open(LOG_FILE, 'w') as f:
    for line in log_msgs:
        f.write(line + "\n")

print("ETL complete.")