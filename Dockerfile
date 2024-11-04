# # Step 1: Base image for Python and PyTorch
# FROM continuumio/miniconda3:latest AS base

# # Update and install essential packages
# ENV DEBIAN_FRONTEND=noninteractive
# RUN apt-get update && \
#     apt-get install -y bash curl git wget software-properties-common

# # Create Conda environment
# ENV PYTHON_VERSION=3.10
# RUN conda create --name conda python=${PYTHON_VERSION} pip

# # Upgrade pip
# RUN /opt/conda/envs/conda/bin/pip install --no-cache-dir --upgrade pip

# # Activate Conda environment and install necessary packages
# # RUN /opt/conda/bin/conda run -n conda pip install torch torchvision torchaudio \
# ENV PIP_NO_CACHE_DIR=off
# ENV PIP_CACHE_DIR=/data/.cache/pip
# ENV MAKEFLAGS="-j$(nproc)"
# ENV MAX_JOBS="-j$(nproc)"
# RUN /opt/conda/bin/conda run -n conda pip install --cache-dir ${PIP_CACHE_DIR} torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
#     --extra-index-url https://download.pytorch.org/whl/cu124

# Step 2: Final image
# FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

FROM nvcr.io/nvidia/pytorch:24.10-py3 AS final

# Install essential packages
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y bash curl git wget software-properties-common && \
    apt-get install -y libgl1 gnupg2 moreutils tk libglib2.0-0 libaio-dev && \
    apt-get install -y unzip && \
    apt-get install -y libc++1


# Install git-lfs
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get install -y git-lfs && \
    git lfs install

SHELL ["/bin/bash", "-c"]

# Copy Conda environment from dependencies stage
# COPY --from=base /opt/conda /opt/conda

# # Set environment variables
ENV PATH /opt/conda/bin:${PATH}
ENV HF_HOME=/data/.cache/huggingface
ENV PIP_NO_CACHE_DIR=off
ENV PIP_CACHE_DIR=/data/.cache/pip
ENV MAKEFLAGS="-j$(nproc)"
# Maximum number of jobs for Ninja
ENV MAX_JOBS="$(nproc)"
# Skip downloading large files with Git LFS
ENV GIT_LFS_SKIP_SMUDGE=1

# Set HuggingFace credentials
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

ARG DIR
COPY ${DIR} /app/
RUN find /app



WORKDIR /app/
# RUN /opt/conda/bin/conda run -n conda pip install --cache-dir ${PIP_CACHE_DIR} flash-attn --no-build-isolation
# RUN test -f requirements.txt && /opt/conda/bin/conda run -n conda pip install --cache-dir ${PIP_CACHE_DIR} -v -r requirements.txt || echo "No requirements.txt found"

# close...
# RUN pip install --cache-dir ${PIP_CACHE_DIR} flash-attn --no-build-isolation --extra-index-url https://download.pytorch.org/whl/cu124

RUN git clone https://github.com/Dao-AILab/flash-attention
WORKDIR /app/flash-attention
# Compilation of flash-attn requires a *lot* of memory if leaving Ninja in charge
ENV MAX_JOBS=4
RUN echo "MAX_JOBS: ${MAX_JOBS}" && python setup.py install

RUN test -f requirements.txt && pip install --cache-dir ${PIP_CACHE_DIR} -v -r requirements.txt || echo "No requirements.txt found"

WORKDIR /app

ARG GIT_DOMAIN
ARG GIT_ORG
ARG GIT_REPO
# if GIT_DOMAIN is not empty, clone the repository
ARG CACHEBUST=3
RUN test -z "$GIT_DOMAIN" || git clone https://${GIT_DOMAIN}/${GIT_ORG}/${GIT_REPO}.git

WORKDIR /app/${GIT_REPO}

ARG GIT_BRANCH
RUN test -z "$GIT_BRANCH" || git checkout $GIT_BRANCH

# If the repository has a requirements.txt file, install the dependencies
ARG DISABLE_REPOSITORY_DEPENDENCIES
ENV DISABLE_REPOSITORY_DEPENDENCIES=${DISABLE_REPOSITORY_DEPENDENCIES}

# RUN if [ -z "${DISABLE_REPOSITORY_DEPENDENCIES}" ]; then \
#     /opt/conda/bin/conda run -n conda pip install --cache-dir ${PIP_CACHE_DIR} -v -r requirements.txt; \
#     fi
RUN if [ -z "${DISABLE_REPOSITORY_DEPENDENCIES}" ]; then \
    pip install --cache-dir ${PIP_CACHE_DIR} -v -r requirements.txt; \
    fi

# If setup.py exists, install the package, do not fail if it does not exist
# RUN test -f setup.py && /opt/conda/bin/conda run -n conda pip install --no-cache-dir -e . || echo "No setup.py found"
ARG CACHEBUST=1
RUN test -f setup.py && pip install --no-cache-dir -e . || echo "No setup.py found"

# Create user and set permissions
ARG USERNAME
ARG DOCKER_USER
RUN USER_ID=$(echo ${DOCKER_USER} | cut -d: -f1) && \
    GROUP_ID=$(echo ${DOCKER_USER} | cut -d: -f2) && \ 
    groupadd --gid $GROUP_ID $USERNAME && \
    useradd --uid $USER_ID --gid $GROUP_ID -m $USERNAME

# Set ownership of /app/ to the user
RUN chown -R $USERNAME /app/

# Switch to the user
USER $USERNAME

# RUN test -f setup.py && /opt/conda/bin/conda run -n conda pip list
# RUN test -f setup.py && pip list

# Set environment variables for PyTorch CUDA
# https://pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf
ENV PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.8,max_split_size_mb:128"

# Set entrypoint to activate Conda environment and start bash shell
# ENTRYPOINT ["/bin/bash", "-c", "source activate conda && /bin/bash -c 'python /app/caption.py'"]
ENTRYPOINT ["/bin/bash", "-c", "python /app/caption.py"]
# ENTRYPOINT ["/bin/bash", "-c", "find /app -name caption.py"]
# ENTRYPOINT ["/bin/bash", "-c", "find /app -name caption.py && which python"]
