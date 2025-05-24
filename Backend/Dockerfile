FROM python:3.9-slim

# Multi-stage build to reduce final image size
FROM python:3.9-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install pip dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install OpenCV with minimal dependencies
RUN pip install --no-cache-dir opencv-python-headless

# Copy the rest of the application
COPY . .

# Create final image
FROM python:3.9-slim

WORKDIR /app

# Copy only necessary files from builder
COPY --from=builder /app/requirements.txt .
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /app/app.py .
COPY --from=builder /app/predict /app/predict

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_RUN_PORT=5000
ENV FLASK_ENV=production
ENV PYTHONPATH=/app

# Expose port 5000
EXPOSE 5000

# Start the application with gunicorn for better performance
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--threads", "4", "app:app"]
