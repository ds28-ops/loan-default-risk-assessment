import pandas as pd
import os

RAW_PATH = "/mnt/data/raw/accepted_2007_to_2018Q4.csv"
OUTPUT_DIR = "/mnt/data/processed"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "train_ready.csv")
LOG_FILE = os.path.join(OUTPUT_DIR, "etl_log.txt")
CHUNK_SIZE = 10000

os.makedirs(OUTPUT_DIR, exist_ok=True)

PREDICTIVE_FEATURES = [
    'delinq_2yrs', 'fico_range_low', 'revol_util', 'pub_rec', 'acc_now_delinq',
    'collections_12_mths_ex_med', 'chargeoff_within_12_mths', 'percent_bc_gt_75',
    'inq_last_6mths', 'mths_since_last_delinq', 'num_tl_90g_dpd_24m', 'total_rev_hi_lim',
    'all_util', 'num_accts_ever_120_pd', 'mths_since_recent_revol_delinq',
    'mths_since_recent_bc_dlq', 'tot_coll_amt', 'inq_last_12m', 'open_il_12m',
    'open_rv_12m', 'total_bal_il', 'tot_cur_bal', 'total_cu_tl', 'inq_fi',
    'open_acc_6m', 'acc_open_past_24mths', 'mths_since_last_record',
    'num_tl_120dpd_2m', 'num_tl_30dpd', 'num_op_rev_tl'
]

USER_INPUT_FEATURES = [
    'loan_amnt', 'term', 'emp_length', 'home_ownership', 'annual_inc',
    'purpose', 'application_type'
]

LEAKAGE_COLUMNS = [
    'id', 'member_id', 'issue_d', 'url', 'desc', 'title', 'zip_code', 'addr_state',
    'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp',
    'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee',
    'last_pymnt_d', 'next_pymnt_d', 'last_credit_pull_d'
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
    original_columns = set(chunk.columns)
    
    # Drop leakage columns
    chunk.drop(columns=[col for col in LEAKAGE_COLUMNS if col in chunk.columns], inplace=True)

    # Drop rows with missing loan_status
    if 'loan_status' not in chunk.columns:
        continue
    chunk = chunk.dropna(subset=['loan_status'])

    # Map risk level
    chunk['risk_level'] = chunk['loan_status'].apply(map_risk_level)
    chunk.drop(columns=['loan_status'], inplace=True)

    # Keep only relevant columns
    available_features = [col for col in PREDICTIVE_FEATURES if col in chunk.columns]
    selected_columns = available_features + USER_INPUT_FEATURES + ['risk_level']
    selected_columns = [col for col in selected_columns if col in chunk.columns]
    chunk = chunk[selected_columns]

    df_list.append(chunk)

# Concatenate all processed chunks
if not df_list:
    raise ValueError("No valid chunks processed. Check column names or data filtering.")
df = pd.concat(df_list, ignore_index=True)

# Drop columns with >40% missing
threshold = len(df) * 0.4
na_cols = df.isnull().sum()
to_drop = na_cols[na_cols > threshold].index.tolist()
df.drop(columns=to_drop, inplace=True)
log_msgs.append(f"Dropped columns with >40% missing: {to_drop}")

# Drop remaining rows with missing
before = len(df)
df = df.dropna()
after = len(df)
log_msgs.append(f"Dropped {before - after} rows with missing values")

# Track missing predictive features across all chunks
all_columns = set(df.columns)
missing_features = [col for col in PREDICTIVE_FEATURES if col not in all_columns]
log_msgs.append("Missing predictive features (after chunking): " + ", ".join(missing_features))

# Save
df.to_csv(OUTPUT_CSV, index=False)

# Log
with open(LOG_FILE, 'w') as f:
    for line in log_msgs:
        f.write(line + "\n")

print(f"ETL completed. Saved to {OUTPUT_CSV}. Log written to {LOG_FILE}.")
