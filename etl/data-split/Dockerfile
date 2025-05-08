# Use slim Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install required Python packages
RUN pip install pandas scikit-learn

# Copy script into container
COPY data-split.py .

# Set default command
CMD ["python", "data-split.py"]