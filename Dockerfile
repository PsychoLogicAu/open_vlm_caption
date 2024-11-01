# Step 1: Base image for Python and PyTorch
FROM continuumio/miniconda3:latest AS base

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
# TODO: torch 2.5 (drop the version specifiers)
# https://pytorch.org/get-started/locally/
# This was taking too long to build and I am impatient
RUN /opt/conda/bin/conda run -n conda pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --extra-index-url https://download.pytorch.org/whl/cu124

# Step 2: Final image
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

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

# Copy Conda environment from dependencies stage
COPY --from=base /opt/conda /opt/conda

# # Set environment variables
ENV PATH /opt/conda/bin:$PATH
ENV HF_HOME=/data/.cache/huggingface

ARG DIR
COPY ${DIR} /app/

# Set ownership of /app/ to the user
RUN chown -R $USERNAME /app/

# Switch to the user
USER $USERNAME

WORKDIR /app/
RUN export MAKEFLAGS="-j$(nproc)" && \
    test -f requirements.txt && /opt/conda/bin/conda run -n conda pip install --no-cache-dir -v -r requirements.txt || echo "No requirements.txt found"

# Set HuggingFace credentials
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}
# RUN echo "HF_TOKEN at build: $HF_TOKEN"

# Skip downloading large files with Git LFS
ENV GIT_LFS_SKIP_SMUDGE=1

ARG GIT_DOMAIN
ARG GIT_ORG
ARG GIT_REPO
# if GIT_DOMAIN is not empty, clone the repository
RUN test -z "$GIT_DOMAIN" || git clone https://${GIT_DOMAIN}/${GIT_ORG}/${GIT_REPO}.git

WORKDIR /app/${GIT_REPO}

ARG GIT_BRANCH
RUN test -z "$GIT_BRANCH" || git checkout $GIT_BRANCH

# If the repository has a requirements.txt file, install the dependencies
RUN export MAKEFLAGS="-j$(nproc)" && \
    test -f requirements.txt && /opt/conda/bin/conda run -n conda pip install --no-cache-dir -r requirements.txt || echo "No requirements.txt found"

# If setup.py exists, install the package, do not fail if it does not exist
RUN test -f setup.py && /opt/conda/bin/conda run -n conda pip install --no-cache-dir -e . || echo "No setup.py found"

# Set environment variables for PyTorch CUDA
# https://pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf
ENV PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.8,max_split_size_mb:128"

# Set entrypoint to activate Conda environment and start bash shell
ENTRYPOINT ["/bin/bash", "-c", "source activate conda && /bin/bash -c 'python /app/caption.py'"]
