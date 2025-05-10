# etl/Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY transform_data_old.py .

RUN pip install pandas numpy scikit-learn

CMD ["python", "transform_data_old.py"]
