# docker/etl/prepare_features.py

import pandas as pd

input_path = "/mnt/data/processed/cleaned_data.csv"
output_path = "/mnt/data/processed/train_ready.csv"

# Load cleaned data
df = pd.read_csv(input_path)

# Step 1: Keep only features of interest
model_features = [
    # Top predictors
    "delinq_2yrs", "fico_range_low", "revol_util", "pub_rec", "acc_now_delinq",
    "collections_12_mths_ex_med", "chargeoff_within_12_mths", "percent_bc_gt_75",
    "inq_last_6mths", "mths_since_last_delinq",

    # Medium importance
    "num_tl_90g_dpd_24m", "total_rev_hi_lim", "all_util", "num_accts_ever_120_pd",
    "mths_since_recent_revol_delinq", "mths_since_recent_bc_dlq", "tot_coll_amt",
    "inq_last_12m", "open_il_12m", "open_rv_12m",

    # Low importance
    "total_bal_il", "tot_cur_bal", "total_cu_tl", "inq_fi", "open_acc_6m",
    "acc_open_past_24mths", "mths_since_last_record", "num_tl_120dpd_2m",
    "num_tl_30dpd", "num_op_rev_tl", "num_rev_tl_bal_gt_0", "open_act_il",
    "fico_range_high", "last_fico_range_low", "last_fico_range_high",

    # Target
    "risk_level"
]

df = df[[col for col in model_features if col in df.columns]]

# Step 2: Drop *any* row with NaN
before_rows = len(df)
df.dropna(inplace=True)
after_rows = len(df)

print(f"Dropped {before_rows - after_rows} rows due to missing values.")

# Step 3: Save result
df.to_csv(output_path, index=False)
print(f"✅ train_ready.csv saved with shape {df.shape}")
