FROM python:3.11-slim

WORKDIR /app

COPY data-select.py .

RUN pip install pandas scikit-learn pyarrow

CMD ["python", "data-select.py"]