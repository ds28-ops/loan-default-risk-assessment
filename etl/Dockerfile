# Use official lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install required packages
RUN pip install pandas xgboost scikit-learn matplotlib seaborn joblib

# Copy all ETL and training scripts
COPY transform_data.py .
COPY model_train.py .

# No CMD here — execution is controlled via docker-compose
