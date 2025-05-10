FROM python:3.11-slim

WORKDIR /app

COPY data-split.py .

RUN pip install pandas scikit-learn pyarrow

CMD ["python", "data-split.py"]