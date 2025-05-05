FROM python:3.9

RUN pip install pandas scikit-learn pyarrow

WORKDIR /app

COPY etl/etl_pipeline.py .

CMD ["python", "etl_pipeline.py"]
