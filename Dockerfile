# Dockerfile
FROM python:3.10.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set default port (can be overridden)
ARG APP_PORT=8501
ENV STREAMLIT_SERVER_PORT=${APP_PORT}
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    make \
    wget \
    git \
    curl \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories that might be needed
RUN mkdir -p data_collection/metadata data_collection/marketdata

# The port to expose (using build argument)
EXPOSE ${APP_PORT}

# Run the application
CMD ["streamlit", "run", "main.py"]