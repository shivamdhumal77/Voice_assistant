# Use a lightweight Python base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install necessary system dependencies and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    python3-dev \
    libportaudio-dev \
    espeak-ng \
    wget \
    cmake \
    portaudio19-dev \
    pkg-config \
    libportaudio2 \
    libportaudiocpp0 \
    libasound2-dev \
    ffmpeg && \
    python-pyaudio \
    rm -rf /var/lib/apt/lists/*

# Install additional Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clone the Kokoro dependencies (if applicable)
RUN git lfs install && \
    git clone https://huggingface.co/hexgrad/Kokoro-82M /app/Kokoro-82M && \
    cd /app/Kokoro-82M && \
    pip install -q phonemizer torch transformers scipy munch

# Copy the application code into the container
COPY app/ ./app

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to start the FastAPI application

