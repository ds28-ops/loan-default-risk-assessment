from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.preprocessing import LabelEncoder

app = FastAPI()

# Load model and transform artifacts
model = joblib.load("model.pth")
artifacts = joblib.load("transform_artifacts.pkl")

KEEP_COLS = artifacts["keep_cols"]
CATEGORICAL_ONEHOT = artifacts["categorical_onehot"]
CATEGORICAL_LABEL = artifacts["categorical_label"]
median_values = artifacts["median_values"]
mode_values = artifacts["mode_values"]
label_encoders = artifacts["label_encoders"]
onehot_columns_train = artifacts["onehot_columns"]
scaler = artifacts["scaler"]
numeric_cols = artifacts["numeric_cols"]
LABEL_COL = "risk_level"

def parse_txt_to_df(text: str):
    data = {}
    for line in text.strip().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            try:
                data[k.strip()] = float(v.strip())
            except ValueError:
                data[k.strip()] = v.strip()
    df = pd.DataFrame([data])
    return df

def transform_input_df(df):

    df = df[KEEP_COLS]
    label_series = df.pop(LABEL_COL).map(lambda x: 0 if x == "Low" else 1)

    # Fill NaNs using training stats
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if col in mode_values:
                df[col] = df[col].fillna(mode_values[col])
            elif col in median_values:
                df[col] = df[col].fillna(median_values[col])
            else:
                df[col] = df[col].fillna(0)

    # One-hot encoding
    for col in CATEGORICAL_ONEHOT:
        dummies = pd.get_dummies(df[col], prefix=col)
    
        # Filter relevant onehot columns for current variable
        relevant_cols = [c for c in onehot_columns_train if c.startswith(f"{col}_")]
    
            # Add missing dummy columns
        for dummy_col in relevant_cols:
            if dummy_col not in dummies.columns:
                dummies[dummy_col] = 0

        dummies = dummies[relevant_cols]
        df = pd.concat([df.drop(columns=[col]), dummies], axis=1)


    # Label encoding
    for col in CATEGORICAL_LABEL:
        le = label_encoders[col]
        df[col] = le.transform(df[col].astype(str))

    # Standard scaling
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    df[LABEL_COL] = label_series.reset_index(drop=True)

    return df

@app.post("/predict_loan_risk")
async def predict_txt(file: UploadFile = File(...)):
    contents = await file.read()
    text = contents.decode("utf-8")

    try:
        raw_df = parse_txt_to_df(text)
        raw_features = raw_df.to_dict(orient="records")[0]

        transformed_df = transform_input_df(raw_df)
        true_label = transformed_df[LABEL_COL].iloc[0] if LABEL_COL in transformed_df.columns else None
        transformed_df = transformed_df.drop(LABEL_COL)
        prediction = model.predict(transformed_df.values)[0]

        return JSONResponse({
            "predicted_class": int(prediction),
            "true_label": true_label,
            "raw_extracted_features": raw_features,
            "features_used": transformed_df.to_dict(orient="records")[0]
        })
    except Exception as e:
        return JSONResponse({"error": str(e)})

