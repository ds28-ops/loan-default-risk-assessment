import pandas as pd

LABEL_COL = "risk_level"

def transform(df: pd.DataFrame):
    y = df[LABEL_COL]
    X = df.drop(columns=[LABEL_COL])
    return X, y

def predict(model, X):
    y_pred = model.predict(X)
    probs = model.predict_proba(X)
    return y_pred, probs
