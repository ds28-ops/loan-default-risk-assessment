FROM python:3.11-slim

WORKDIR /app

COPY fastapi_pt/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ..
COPY ../train/models/xgb-kfold-binary-13/model.pth .
COPY fastapi_pt/transform_artifacts.pkl .


EXPOSE 8000
CMD ["uvicorn", "fastapi_pt.app:app", "--host", "0.0.0.0", "--port", "8000"]
