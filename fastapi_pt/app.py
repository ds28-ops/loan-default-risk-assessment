from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import joblib
import os
import uuid
import json
import subprocess
from fastapi import Request

from sklearn.preprocessing import LabelEncoder
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Histogram, Counter


app = FastAPI()

confidence_histogram = Histogram(
    "prediction_confidence",
    "Model prediction confidence",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

class_counter = Counter(
    "predicted_class_total",
    "Count of predictions per class",
    ['class_name']
)
classes = ["Low", "High"]


# Instrumentator().instrument(app).expose(app)
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
    # model = joblib.load("model.pth")
    # artifacts = joblib.load("transform_artifacts.pkl")

    contents = await file.read()
    text = contents.decode("utf-8")

    try:
        raw_df = parse_txt_to_df(text)
        raw_features = raw_df.to_dict(orient="records")[0]
        # if LABEL_COL in raw_df.columns:
        #     raw_df = raw_df.drop(columns=[LABEL_COL])

        transformed_df = transform_input_df(raw_df)
        true_label = transformed_df[LABEL_COL].iloc[0] if LABEL_COL in transformed_df.columns else None
        transformed_df = transformed_df.drop(columns=[LABEL_COL])
        prediction = model.predict(transformed_df.values)[0]
        confidence= model.predict_proba(transformed_df.values)[0][prediction]
        class_name = "Low" if int(prediction) == 0 else "High"
        confidence_histogram.observe(confidence)
        class_counter.labels(class_name=class_name).inc()  # ✅ CORRECT

        

        return JSONResponse({
            "predicted_class": int(prediction),
            "class_name": class_name,
            "confidence": float(confidence),
            "true_label": int(true_label) if true_label is not None else None,
            "features_used": transformed_df.to_dict(orient="records")[0]
        })
    except Exception as e:
        return JSONResponse({"error": str(e)})


@app.post("/feedback")
async def save_feedback(request: Request):
    data = await request.json()
    is_correct = data["is_correct"]
    record = data["record"]

    if not is_correct:
        record["risk_level"] = 1 - int(record["risk_level"])

    filename = f"/mnt/object/production_data/{uuid.uuid4()}.json"
    with open(filename, "w") as f:
        json.dump(record, f)

    dest_path = f"chi_uc:/production-artifacts/{os.path.basename(filename)}"
    try:
        subprocess.run(["rclone", "copy", filename, dest_path], check=True)
        return {"status": "saved", "flipped": not is_correct}
    except subprocess.CalledProcessError as e:
        return JSONResponse({"error": f"Failed to copy via rclone: {str(e)}"}, status_code=500)
    

    
Instrumentator().instrument(app).expose(app)
