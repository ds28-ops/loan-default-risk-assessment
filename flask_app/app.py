from flask import Flask, render_template, request, jsonify
import requests
import os
import json

app = Flask(__name__)

# Get FastAPI URL from environment or use default
FASTAPI_URL = os.environ.get('FASTAPI_SERVER_URL', 'http://fastapi_server:8000')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        loan_data = request.json
        
        # Forward to FastAPI
        response = requests.post(f"{FASTAPI_URL}/predict", 
                                json=loan_data,
                                timeout=10)
        
        # Return response from FastAPI
        return jsonify(response.json())
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Could not connect to prediction service"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health')
def health():
    try:
        # Check FastAPI health
        response = requests.get(f"{FASTAPI_URL}/health", timeout=5)
        api_status = response.json()
        
        # Return health status
        return jsonify({
            "flask_app": "healthy",
            "prediction_service": api_status
        })
    except:
        return jsonify({
            "flask_app": "healthy",
            "prediction_service": "unavailable"
        }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)