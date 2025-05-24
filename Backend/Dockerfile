FROM python:3.9-slim

FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for OpenCV, PyTorch, and git
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install pip dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Make sure the app runs on port 5000
ENV FLASK_APP=app.py
ENV FLASK_RUN_PORT=5000
ENV FLASK_ENV=production

# Expose port 5000
EXPOSE 5000

# Start the application
CMD ["flask", "run", "--host=0.0.0.0"]
