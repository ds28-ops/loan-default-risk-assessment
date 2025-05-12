FROM python:3.11-slim

WORKDIR /app

COPY fastapi_pt/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y curl unzip && \
    curl -O https://downloads.rclone.org/rclone-current-linux-amd64.zip && \
    unzip rclone-current-linux-amd64.zip && \
    cp rclone-*-linux-amd64/rclone /usr/bin/rclone && \
    chmod 755 /usr/bin/rclone && \
    rm -rf rclone-*-linux-amd64 rclone-current-linux-amd64.zip
COPY . .
COPY ../train/models/xgb-kfold-binary-13/model.pth .
COPY fastapi_pt/transform_artifacts.pkl .


EXPOSE 8000
CMD ["uvicorn", "fastapi_pt.app:app", "--host", "0.0.0.0", "--port", "8000"]
