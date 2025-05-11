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
    {% if prediction %}
        <hr>
        <h2>Prediction</h2>
        <p><strong>Predicted Class:</strong> {{ prediction.predicted_class }}</p>
        <p><strong>True Label (if provided):</strong> {{ prediction.true_label }}</p>
        <!-- Removed features_used block -->
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
        file = request.files["file"]
        if file:
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

    return render_template_string(TEMPLATE, prediction=prediction, error=error)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
