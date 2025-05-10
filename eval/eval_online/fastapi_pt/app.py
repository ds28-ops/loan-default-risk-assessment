
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import xgboost as xgb
import os
from prometheus_client import Counter, Histogram, make_asgi_app

# Create FastAPI app
app = FastAPI(title="Loan Default Risk Assessment API")

# Set up Prometheus metrics
prediction_counter = Counter(
    "loan_default_predictions_total", 
    "Total number of loan default predictions made",
    ["prediction"]
)
prediction_time = Histogram(
    "loan_default_prediction_time_seconds",
    "Time taken to make loan default predictions"
)

# Create Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Define request model
class LoanApplication(BaseModel):
    # Add your features here based on your model
    annual_income: float
    loan_amount: float
    loan_term: int
    interest_rate: float
    employment_years: float
    home_ownership: str
    loan_purpose: str
    # Add other features as needed

# Define response model
class PredictionResponse(BaseModel):
    risk_category: str
    probability: float
    explanation: list[str]

# Load model when app starts
model = None

@app.on_event("startup")
async def startup_event():
    global model
    try:
        # Update this path if your model has a different name or location
        model_path = "/app/model.pth"
        if os.path.exists(model_path):
            model = xgb.Booster()
            model.load_model(model_path)
        else:
            # Fallback to a default location if model not found
            # This allows the API to start even if the model isn't ready yet
            print(f"WARNING: Model not found at {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")

@app.get("/")
def read_root():
    return {"message": "Loan Default Risk Assessment API", "status": "active"}

@app.get("/health")
def health_check():
    if model is None:
        return {"status": "warning", "message": "Model not loaded yet"}
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(application: LoanApplication):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    with prediction_time.time():
        try:
            # Transform input data based on your model's requirements
            # This is a placeholder - adjust according to your model features
            features = [
                application.annual_income,
                application.loan_amount,
                application.loan_term,
                application.interest_rate,
                application.employment_years,
                # Add feature encoding for categorical variables
                # Encode home_ownership
                # Encode loan_purpose
                # ...
            ]
            
            # Prepare data for XGBoost prediction
            dmatrix = xgb.DMatrix([features])
            
            # Get prediction
            prediction = model.predict(dmatrix)[0]
            
            # Determine risk category based on prediction
            if prediction < 0.3:
                risk_category = "Low"
                explanation = ["Good income to loan ratio", "Favorable credit history"]
            elif prediction < 0.7:
                risk_category = "Medium"
                explanation = ["Moderate debt-to-income ratio", "Average credit profile"]
            else:
                risk_category = "High"
                explanation = ["High loan amount relative to income", "Limited credit history"]
            
            # Track prediction with Prometheus
            prediction_counter.labels(prediction=risk_category).inc()
            
            return {
                "risk_category": risk_category,
                "probability": float(prediction),
                "explanation": explanation
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")