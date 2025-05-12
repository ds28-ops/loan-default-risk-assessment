from flask import Flask, request, render_template_string
import requests
import json

app = Flask(__name__)
FASTAPI_SERVER_URL = "http://fastapi_server:8000"

TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Loan Risk Predictor</title>
</head>
<body>
    <h1>Upload Loan Document (.txt)</h1>
    <form method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept=".txt" required>
        <input type="submit" value="Submit">
    </form>

    {% if prediction and prediction.status %}
        <h2 style="color: green;">✅ Saved to production store</h2>
        {% if prediction.flipped %}
            <p><strong>Note:</strong> risk_level was flipped before saving.</p>
        {% endif %}
    {% elif prediction %}
        <hr>
        <h2>Prediction</h2>
        <p><strong>Predicted Class (Raw Integer):</strong> {{ prediction.predicted_class }}</p>
        <p><strong>Class Label:</strong> {{ prediction.class_name }}</p>
        <p><strong>Confidence:</strong> {{ (prediction.confidence * 100) | round(2) }}%</p>
        <p><strong>True Label (if provided):</strong> {{ prediction.true_label }}</p>
        <form method="post">
            <textarea name="feedback_record" style="display:none;">{{ prediction.features_used | tojson }}</textarea>
            <button name="is_correct" value="true">✅ Prediction is Correct</button>
            <button name="is_correct" value="false">❌ Prediction is Wrong</button>
        </form>
    {% elif error %}
        <hr>
        <h2 style="color:red;">Error</h2>
        <pre>{{ error }}</pre>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    if request.method == "POST":
        if "file" in request.files:
            file = request.files["file"]
            try:
                response = requests.post(
                    f"{FASTAPI_SERVER_URL}/predict_loan_risk",
                    files={"file": (file.filename, file.read(), file.content_type)}
                )
                data = response.json()
                if "error" in data:
                    error = data["error"]
                else:
                    prediction = data
            except Exception as e:
                error = str(e)
        elif "feedback_record" in request.form:
            is_correct = request.form["is_correct"] == "true"
            record = json.loads(request.form["feedback_record"])
            try:
                response = requests.post(
                    f"{FASTAPI_SERVER_URL}/feedback",
                    json={"is_correct": is_correct, "record": record}
                )
                data = response.json()
                if "error" in data:
                    error = data["error"]
                else:
                    prediction = {"status": data["status"], "flipped": data.get("flipped", False)}
            except Exception as e:
                error = str(e)

    return render_template_string(TEMPLATE, prediction=prediction, error=error)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

