from flask import Flask, render_template, request
import requests

app = Flask(__name__)
FASTAPI_SERVER_URL = "http://fastapi_server:8000"

@app.route("/", methods=["GET", "POST"])
def index():
    content = ""
    if request.method == "POST":
        file = request.files["file"]
        if file:
            response = requests.post(
                f"{FASTAPI_SERVER_URL}/read_txt",
                files={"file": (file.filename, file.stream, file.content_type)},
            )
            content = response.text
    return f"""
    <h1>Upload Loan Document (.txt)</h1>
    <form method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept=".txt" required>
        <input type="submit" value="Submit">
    </form>
    <hr>
    <h2>File Content:</h2>
    <pre>{content}</pre>
    """

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
