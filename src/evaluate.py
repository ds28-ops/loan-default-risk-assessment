import pandas as pd

LABEL_COL = "risk_level"

def transform(df: pd.DataFrame):
    required_cols = [
        "loan_amnt", "funded_amnt", "funded_amnt_inv", "int_rate", "installment",
        "sub_grade", "annual_inc", "dti", "delinq_2yrs", "fico_range_low", "fico_range_high",
        "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
        "collections_12_mths_ex_med", "acc_now_delinq", "tot_coll_amt", "tot_cur_bal",
        "open_act_il", "open_il_12m", "total_bal_il", "open_rv_12m", "max_bal_bc", "all_util",
        "total_rev_hi_lim", "inq_fi", "total_cu_tl", "inq_last_12m", "avg_cur_bal",
        "bc_open_to_buy", "bc_util", "chargeoff_within_12_mths", "delinq_amnt", "mort_acc",
        "mths_since_recent_inq", "num_actv_bc_tl", "num_actv_rev_tl", "num_bc_sats",
        "num_bc_tl", "num_il_tl", "num_op_rev_tl", "num_rev_accts", "num_rev_tl_bal_gt_0",
        "num_sats", "num_tl_120dpd_2m", "num_tl_30dpd", "num_tl_90g_dpd_24m",
        "num_tl_op_past_12m", "pct_tl_nvr_dlq", "percent_bc_gt_75", "pub_rec_bankruptcies",
        "tax_liens", "tot_hi_cred_lim", "total_bal_ex_mort", "total_bc_limit",
        "total_il_high_credit_limit", "home_ownership_ANY", "home_ownership_MORTGAGE",
        "home_ownership_NONE", "home_ownership_OTHER", "home_ownership_OWN",
        "home_ownership_RENT", "verification_status_Not Verified",
        "verification_status_Source Verified", "verification_status_Verified", "risk_level"
    ]

    df = df[required_cols]
    y = df["risk_level"]
    X = df.drop(columns=["risk_level"])
    return X, y


def predict(model, X):
    y_pred = model.predict(X)
    probs = model.predict_proba(X)
    return y_pred, probs
