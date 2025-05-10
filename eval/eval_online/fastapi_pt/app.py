
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from prometheus_client import Counter, make_asgi_app
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="Loan Default Risk Assessment API")

# Set up Prometheus metrics
prediction_counter = Counter(
    "loan_default_predictions_total", 
    "Total number of loan default predictions made",
    ["prediction"]
)

# Create Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Define request model
class LoanApplication(BaseModel):
    annual_income: float
    loan_amount: float
    loan_term: int
    interest_rate: float
    employment_years: float
    home_ownership: str
    loan_purpose: str

# Define response model
class PredictionResponse(BaseModel):
    risk_category: str
    probability: float
    explanation: list[str]

# Model placeholder
model = None

@app.on_event("startup")
async def startup_event():
    global model
    try:
        model_path = "/app/model.pth"
        if os.path.exists(model_path):
            logger.info(f"Loading model from {model_path}")
            # Actual model loading will happen here when model is ready
            # For now, just log that we found the file
        else:
            logger.warning(f"Model not found at {model_path}. Using fallback predictions.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")

@app.get("/")
def read_root():
    return {"message": "Loan Default Risk Assessment API", "status": "active"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
async def predict(application: LoanApplication):
    logger.info(f"Received prediction request: {application}")
    
    # Since we don't have a real model yet, return mock predictions
    # This allows testing the API integration without a real model
    
    # Mock logic based on loan amount to income ratio
    ratio = application.loan_amount / application.annual_income
    
    if ratio < 0.2:
        risk_category = "Low"
        probability = 0.1
        explanation = ["Good income to loan ratio", "Favorable application profile"]
    elif ratio < 0.4:
        risk_category = "Medium"
        probability = 0.5
        explanation = ["Moderate debt-to-income ratio", "Average application profile"]
    else:
        risk_category = "High" 
        probability = 0.8
        explanation = ["High loan amount relative to income", "Risky application profile"]
    
    # Track prediction with Prometheus
    prediction_counter.labels(prediction=risk_category).inc()
    
    return {
        "risk_category": risk_category,
        "probability": probability,
        "explanation": explanation
    }
