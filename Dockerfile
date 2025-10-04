# Gradient-ready Flask app with optional GPU Torch
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps (libgl for OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

# 1) Install GPU-enabled PyTorch (CUDA 12.1). If no GPU is present at runtime, Torch will still run on CPU.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121

# 2) Install remaining dependencies without touching torch
RUN pip install --no-cache-dir -r requirements.txt --no-deps

COPY . .

EXPOSE 8080
ENV HOST=0.0.0.0 \
    PORT=8080

CMD ["python", "app.py"]
