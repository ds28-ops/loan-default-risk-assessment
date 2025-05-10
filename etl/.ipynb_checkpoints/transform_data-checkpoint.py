import pandas as pd
import numpy as np
import argparse
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder

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

def is_large_numeric(col):
    return np.issubdtype(col.dtype, np.number) and col.max() > 1000

def main(input_path, output_path):
    log_msgs = []
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    log_path = os.path.join(os.path.dirname(output_path), "etl_log.txt")

    df = pd.read_csv(input_path, low_memory=False)
    log_msgs.append(f"Original shape: {df.shape}")

    # Separate label if exists
    label_series = None
    if LABEL_COL in df.columns:
        label_series = df[LABEL_COL]
        df = df.drop(columns=[LABEL_COL])
        log_msgs.append(f"Preserved label column '{LABEL_COL}' separately.")

    # Drop known leakage/unwanted columns
    to_drop = [col for col in DROP_COLS if col in df.columns]
    df.drop(columns=to_drop, inplace=True)
    log_msgs.append(f"Dropped columns: {to_drop}")

    # Drop rows with too many NaNs
    row_nan_threshold = int(0.4 * df.shape[1])
    df = df[df.isnull().sum(axis=1) <= row_nan_threshold]
    log_msgs.append(f"Remaining rows after NaN row filter: {len(df)}")

    # Fill missing values (excluding label)
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if np.issubdtype(df[col].dtype, np.number):
                median = df[col].median()
                df[col].fillna(median, inplace=True)
                log_msgs.append(f"Filled NaNs in numeric column '{col}' with median: {median}")
            else:
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col].fillna(mode_val[0], inplace=True)
                    log_msgs.append(f"Filled NaNs in categorical column '{col}' with mode: {mode_val[0]}")
                else:
                    df[col].fillna("Unknown", inplace=True)
                    log_msgs.append(f"Filled NaNs in column '{col}' with 'Unknown'")

    # Normalize large numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    large_numeric = [col for col in numeric_cols if is_large_numeric(df[col])]
    if large_numeric:
        scaler = StandardScaler()
        df[large_numeric] = scaler.fit_transform(df[large_numeric])
        log_msgs.append(f"Standard-scaled numeric columns: {large_numeric}")

    # Encode categorical columns
    object_cols = df.select_dtypes(include='object').columns.tolist()
    for col in object_cols:
        if col not in KEEP_CATEGORICAL:
            df.drop(columns=[col], inplace=True)
            log_msgs.append(f"Dropped high-cardinality/unapproved column: '{col}'")
            continue

        n_unique = df[col].nunique()
        if n_unique <= 30:
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
            log_msgs.append(f"One-hot encoded: '{col}'")
        else:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            log_msgs.append(f"Label-encoded: '{col}'")

    # Re-attach label if it was present
    if label_series is not None:
        df[LABEL_COL] = label_series.reset_index(drop=True)
        log_msgs.append(f"Re-attached label column '{LABEL_COL}' to transformed data.")

    # Save
    df.to_csv(output_path, index=False)
    log_msgs.append(f"Saved transformed file to: {output_path}")

    with open(log_path, 'w') as f:
        for line in log_msgs:
            f.write(line + "\n")

    print("✅ Transformation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Loan Feature Transformer")
    parser.add_argument("--input_path", required=True, help="Path to input CSV")
    parser.add_argument("--output_path", required=True, help="Path to output transformed CSV")
    args = parser.parse_args()
    main(args.input_path, args.output_path)
