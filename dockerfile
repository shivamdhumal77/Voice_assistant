# Use a lightweight Python base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install PortAudio from source
RUN wget https://files.portaudio.com/archives/pa_stable_v190600_20161030.tgz && \
    tar -xvzf pa_stable_v190600_20161030.tgz && \
    cd portaudio && \
    ./configure && make && make install && \
    cd .. && rm -rf portaudio*

# Install necessary packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    python3-dev \
    portaudio19-dev \
    build-essential \
    python3-pyaudio \
    espeak-ng && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir phonemizer torch transformers scipy munch

# Clone Kokoro dependencies
RUN git lfs install && \
    git clone https://huggingface.co/hexgrad/Kokoro-82M && \
    cd Kokoro-82M && pip install -q phonemizer torch transformers scipy munch

# Copy the application code into the container
COPY app/ ./app

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to start the FastAPI application
CMD ["uvicorn", "app.main_file:app", "--host", "0.0.0.0", "--port", "8000"]
