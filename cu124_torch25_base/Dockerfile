# Step 1: Base image for Python and PyTorch
FROM continuumio/miniconda3:latest AS python_base

# Update and install essential packages
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y bash curl git wget software-properties-common

# Create Conda environment
ENV PYTHON_VERSION=3.10
RUN conda create --name conda python=${PYTHON_VERSION} pip

# Upgrade pip
RUN /opt/conda/envs/conda/bin/pip install --no-cache-dir --upgrade pip

# Activate Conda environment and install necessary packages
RUN /opt/conda/bin/conda run -n conda pip install torch torchvision torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cu124

# Step 2: Final image
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Copy Conda environment from dependencies stage
COPY --from=python_base /opt/conda /opt/conda
ENV PATH /opt/conda/bin:$PATH

# Install essential packages
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y bash curl git wget software-properties-common && \
    apt-get install -y libgl1 gnupg2 moreutils tk libglib2.0-0 libaio-dev && \
    apt-get install -y unzip

# Install git-lfs
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get install -y git-lfs && \
    git lfs install

# Create user and set permissions
ARG USERNAME
ARG DOCKER_USER
RUN UID=$(echo ${DOCKER_USER} | cut -d: -f1) && \
    GID=$(echo ${DOCKER_USER} | cut -d: -f2) && \
    groupadd --gid $GID $USERNAME && \
    useradd --uid $UID --gid $GID -m $USERNAME

# Set HuggingFace credentials
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}
ENV HF_HOME=/data/.cache/huggingface

# Set environment variables for PyTorch CUDA
# https://pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf
ENV PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.8,max_split_size_mb:128"
