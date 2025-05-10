from flask import Flask, request, render_template, jsonify
import requests
import uuid, os, json

app = Flask(__name__)
FASTAPI_URL = os.environ.get("FASTAPI_URL", "http://fastapi_server:8000/predict")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        form_data = request.form.to_dict()
        response = requests.post(FASTAPI_URL, json=form_data)
        prediction = response.json()
        return render_template("index.html", prediction=prediction, data=form_data)
    return render_template("index.html")

@app.route("/api", methods=["POST"])
def api_predict():
    data = request.get_json()
    response = requests.post(FASTAPI_URL, json=data)
    return jsonify(response.json())
