import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split

RAW_PATH = "/mnt/data/raw/accepted_2007_to_2018Q4.csv"
OUTPUT_BASE = "/mnt/object/loan-default-data"
CHUNK_SIZE = 50000

OUTPUT_TRAIN_DIR = os.path.join(OUTPUT_BASE, "train")
OUTPUT_VAL_DIR = os.path.join(OUTPUT_BASE, "val")
OUTPUT_EVAL_DIR = os.path.join(OUTPUT_BASE, "eval")
LOG_FILE = os.path.join(OUTPUT_BASE, "etl_log.txt")

os.makedirs(OUTPUT_TRAIN_DIR, exist_ok=True)
os.makedirs(OUTPUT_VAL_DIR, exist_ok=True)
os.makedirs(OUTPUT_EVAL_DIR, exist_ok=True)

DROP_COLS = [
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
    'emp_title', 'sub_grade', 'issue_d', 'application_type', 'disbursement_method', 'initial_list_status'
]

KEEP_CATEGORICAL = ['term', 'emp_length', 'home_ownership', 'verification_status', 'purpose']
LABEL_COL = 'risk_level'

def map_risk_level(status):
    if status in ["Fully Paid", "Current"]:
        return "Low"
    elif status in ["Late (16-30 days)", "In Grace Period"]:
        return "Medium"
    else:
        return "High"

log_msgs = []
df_list = []

for i, chunk in enumerate(pd.read_csv(RAW_PATH, chunksize=CHUNK_SIZE, low_memory=False)):
    log_msgs.append(f"\n🔹 Processing chunk {i+1} - original shape: {chunk.shape}")
    orig_cols = chunk.columns.tolist()

    # Drop specified columns
    to_drop = [col for col in DROP_COLS if col in chunk.columns]
    chunk.drop(columns=to_drop, inplace=True)
    log_msgs.append(f"Dropped known leakage/unwanted columns: {to_drop}")

    # Drop rows where loan_status is missing
    if 'loan_status' not in chunk.columns:
        continue
    chunk = chunk.dropna(subset=['loan_status'])
    log_msgs.append(f"Remaining rows after dropping missing loan_status: {len(chunk)}")

    # Map risk_level and drop loan_status
    chunk[LABEL_COL] = chunk['loan_status'].apply(map_risk_level)
    chunk.drop(columns=['loan_status'], inplace=True)
    log_msgs.append("Mapped 'loan_status' to 'risk_level'.")

    # Drop columns with >40% NaNs (chunk-level)
    threshold = int(0.4 * len(chunk))
    na_cols = chunk.isnull().sum()
    high_nan_cols = na_cols[na_cols > threshold].index.tolist()
    chunk.drop(columns=high_nan_cols, inplace=True)
    log_msgs.append(f"Dropped cols with >40% NaNs: {high_nan_cols}")

    # Fill missing values
    # for col in chunk.columns:
    #     if chunk[col].isnull().sum() > 0:
    #         if chunk[col].dtype in ['float64', 'int64']:
    #             median = chunk[col].median()
    #             chunk[col].fillna(median, inplace=True)
    #             log_msgs.append(f"Filled NaNs in numeric column '{col}' with median: {median}")
    #         elif chunk[col].dtype == 'object':
    #             mode = chunk[col].mode()
    #             if not mode.empty:
    #                 chunk[col].fillna(mode[0], inplace=True)
    #                 log_msgs.append(f"Filled NaNs in categorical column '{col}' with mode: {mode[0]}")
    #             else:
    #                 chunk[col].fillna("Unknown", inplace=True)
    #                 log_msgs.append(f"Filled NaNs in column '{col}' with 'Unknown'")

    # Drop all object columns not in KEEP_CATEGORICAL
    obj_cols = chunk.select_dtypes(include='object').columns.tolist()
    for col in obj_cols:
        if col not in KEEP_CATEGORICAL:
            chunk.drop(columns=[col], inplace=True)
            log_msgs.append(f"Dropped high-cardinality/unapproved column: {col}")

    # # One-hot encode approved categorical columns
    # for col in KEEP_CATEGORICAL:
    #     if col in chunk.columns:
    #         dummies = pd.get_dummies(chunk[col], prefix=col)
    #         chunk = pd.concat([chunk.drop(columns=[col]), dummies], axis=1)
    #         log_msgs.append(f"One-hot encoded: {col}")

    # Drop any remaining rows with NaNs
    pre_drop_shape = chunk.shape
    chunk.dropna(inplace=True)
    dropped = pre_drop_shape[0] - chunk.shape[0]
    if dropped > 0:
        log_msgs.append(f"Dropped rows with remaining NaNs: {dropped}")

    df_list.append(chunk)

# Combine all chunks
if not df_list:
    raise ValueError("No valid chunks processed.")

df = pd.concat(df_list, ignore_index=True)
log_msgs.append(f"\n✅ Combined processed shape before final NaN drop: {df.shape}")
print(f"\n✅ Combined processed shape before final NaN drop: {df.shape}")
print("Concat Done!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
# Final drop of rows with NaNs (just in case)
# pre_final = df.shape[0]
# df.dropna(inplace=True)
# log_msgs.append(f"Final NaN row drop after merge: {pre_final - df.shape[0]} rows")

# Shuffle before split
# df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

# Split 80/10/10
n_total = len(df)
n_train = int(0.8 * n_total)
n_val = int(0.1 * n_total)
n_eval = n_total - n_train - n_val

train_df = df.iloc[:n_train]
val_df = df.iloc[n_train:n_train + n_val]
eval_df = df.iloc[n_train + n_val:]

train_df.to_csv(os.path.join(OUTPUT_TRAIN_DIR, "train.csv"), index=False, chunksize=10000)
val_df.to_csv(os.path.join(OUTPUT_VAL_DIR, "val.csv"), index=False, chunksize=5000)
eval_df.to_csv(os.path.join(OUTPUT_EVAL_DIR, "eval.csv"), index=False, chunksize=5000)

log_msgs.append(f"\n📝 Saved splits:")
log_msgs.append(f"Train: {len(train_df)} rows")
log_msgs.append(f"Val: {len(val_df)} rows")
log_msgs.append(f"Eval: {len(eval_df)} rows")

# Write logs
with open(LOG_FILE, 'w') as f:
    for msg in log_msgs:
        f.write(msg + "\n")

print("🎯 ETL pipeline complete.")
