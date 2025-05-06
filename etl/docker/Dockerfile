# etl/docker/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install pandas

# Copy the check script
COPY check_data.py .

CMD ["python", "check_data.py"]
