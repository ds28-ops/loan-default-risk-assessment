import pandas as pd
import os

raw_path = "/mnt/data/raw/accepted_2007_to_2018Q4.csv"
output_path = "/mnt/data/processed/cleaned_data.csv"
os.makedirs("/mnt/data/processed", exist_ok=True)

chunks = pd.read_csv(raw_path, chunksize=10000, low_memory=False)
first_chunk = True

# Columns that are known only *after* loan is issued → must be dropped
leakage_cols = [
    "id", "member_id", "issue_d", "url", "desc", "title",
    "zip_code", "addr_state", "pymnt_plan", "application_type",
    "emp_title", "policy_code", "out_prncp", "out_prncp_inv", "total_pymnt",
    "total_pymnt_inv", "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
    "recoveries", "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d", "last_credit_pull_d"
]

# Only features *known at application time* that are predictive
selected_features = [
    "delinq_2yrs", "fico_range_low", "revol_util", "pub_rec", "acc_now_delinq",
    "collections_12_mths_ex_med", "chargeoff_within_12_mths", "percent_bc_gt_75",
    "inq_last_6mths", "mths_since_last_delinq", "loan_status"
]

def map_risk_level(status):
    if status in ["Fully Paid", "Current"]:
        return "Low"
    elif status in ["Late (16-30 days)", "In Grace Period"]:
        return "Medium"
    else:
        return "High"

for chunk in chunks:
    # Drop leakage columns
    chunk.drop(columns=[col for col in leakage_cols if col in chunk.columns], inplace=True, errors="ignore")

    # Filter to keep only selected features
    chunk = chunk[[col for col in selected_features if col in chunk.columns]]

    # Drop rows without target
    chunk.dropna(subset=["loan_status"], inplace=True)

    # Map risk level
    chunk["risk_level"] = chunk["loan_status"].apply(map_risk_level)

    # Drop original loan_status
    chunk.drop(columns=["loan_status"], inplace=True)

    # Write cleaned chunk
    chunk.to_csv(output_path, mode='a', index=False, header=first_chunk)
    first_chunk = False

print("✅ Cleaning complete. Data saved to:", output_path)
