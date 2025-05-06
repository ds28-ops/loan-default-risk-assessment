import pandas as pd
import os

RAW_PATH = "/mnt/data/loan_subset.csv"
OUTPUT_DIR = "/mnt/data/processed"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "train_ready.csv")
LOG_FILE = os.path.join(OUTPUT_DIR, "etl_log.txt")

# Create output dir if not exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Features selected based on prior importance and availability
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

# Load
df = pd.read_csv(RAW_PATH)
original_columns = set(df.columns)
log_msgs = []

# Drop leakage columns
df.drop(columns=[col for col in LEAKAGE_COLUMNS if col in df.columns], inplace=True)
log_msgs.append("Dropped leakage columns: " + ", ".join([col for col in LEAKAGE_COLUMNS if col in original_columns]))

# Drop rows with missing loan_status
df = df.dropna(subset=['loan_status'])

# Map label
df['risk_level'] = df['loan_status'].apply(map_risk_level)
df.drop(columns=['loan_status'], inplace=True)

# Track missing predictive features
missing_features = [col for col in PREDICTIVE_FEATURES if col not in df.columns]
available_features = [col for col in PREDICTIVE_FEATURES if col in df.columns]
log_msgs.append("Missing predictive features: " + ", ".join(missing_features))

# Compose final feature list
final_features = available_features + USER_INPUT_FEATURES + ['risk_level']
df = df[[col for col in final_features if col in df.columns]]

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

# Save
df.to_csv(OUTPUT_CSV, index=False)

# Log
with open(LOG_FILE, 'w') as f:
    for line in log_msgs:
        f.write(line + "\n")

print(f"ETL completed. Saved to {OUTPUT_CSV}. Log written to {LOG_FILE}.")
