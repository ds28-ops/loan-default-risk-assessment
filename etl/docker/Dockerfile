# Use official lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install required packages
RUN pip install --no-cache-dir pandas

# Copy the transformation script
COPY transform_data.py .

# Define the default command
CMD ["python", "transform_data.py"]
