# docker/Dockerfile

FROM python:3.9

# Install required Python libraries
RUN pip install pandas scikit-learn pyarrow

# Set working directory inside container
WORKDIR /app

# Copy feature preparation script into container
# Assumes prepare_features.py is in the root project directory
COPY ../prepare_features.py .

# Default command to run when container starts
CMD ["python", "prepare_features.py"]