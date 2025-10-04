# Dockerfile for Polyp Detection Pipeline
# GPU-ready with CUDA support

# Base image with CUDA support
# If CUDA not available, use: FROM python:3.10-slim
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/{train,val,test}/{images,labels} \
    data_preprocessed/{train,val,test}/{images,labels} \
    logs/checkpoints \
    demo_outputs \
    results

# Set permissions
RUN chmod -R 755 /app

# Expose TensorBoard port
EXPOSE 6006

# Default command (can be overridden)
CMD ["/bin/bash"]

# --- INSTRUCTIONS FOR NON-GPU ENVIRONMENTS ---
# If running without GPU/CUDA support:
# 1. Change the base image to: FROM python:3.10-slim
# 2. Remove CUDA-related ENV variables
# 3. In requirements.txt, use CPU-only PyTorch:
#    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
#
# Build command (GPU):
#   docker build -t polyp-detection:gpu .
#
# Build command (CPU):
#   docker build -f Dockerfile.cpu -t polyp-detection:cpu .
#
# Run command (GPU):
#   docker run --gpus all -it -v $(pwd)/data:/app/data -p 6006:6006 polyp-detection:gpu
#
# Run command (CPU):
#   docker run -it -v $(pwd)/data:/app/data -p 6006:6006 polyp-detection:cpu
#
# Run training:
#   docker run --gpus all -it polyp-detection:gpu python train.py --dataset data_preprocessed --epochs 200
#
# Run evaluation:
#   docker run --gpus all -it polyp-detection:gpu python evaluate.py --checkpoint logs/checkpoints/best.pt
#
# Run inference:
#   docker run --gpus all -it polyp-detection:gpu python infer.py --input data/test/images --out demo_outputs
