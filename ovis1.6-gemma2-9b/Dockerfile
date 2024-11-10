FROM cu124_torch25_base:latest AS base

RUN export MAKEFLAGS="-j$(nproc)" && \
    /opt/conda/bin/conda run -n conda pip install --no-cache-dir -v \
    torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 numpy==1.24.3 transformers==4.44.2 pillow==10.3.0 gekko pandas
    # numpy transformers pillow gekko pandas

ENV TORCH_CUDA_ARCH_LIST="8.9"
RUN git clone https://github.com/AIDC-AI/AutoGPTQ.git && \
    cd AutoGPTQ && \
    /opt/conda/bin/conda run -n conda pip install -vvv --no-build-isolation -e .

WORKDIR /app/

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

# # If the repository has a requirements.txt file, install the dependencies
# RUN test -f requirements.txt && \
#     export MAKEFLAGS="-j$(nproc)" && \
#     /opt/conda/bin/conda run -n conda pip install --no-cache-dir -r requirements.txt || echo "No requirements.txt found"

# # If setup.py exists, install the package, do not fail if it does not exist
# RUN test -f setup.py && \
#     export MAKEFLAGS="-j$(nproc)" && \
#     /opt/conda/bin/conda run -n conda pip install --no-cache-dir -e . || echo "No setup.py found"

ARG DIR
COPY ${DIR}/caption.py /app

# Set entrypoint to activate Conda environment and start bash shell
ENTRYPOINT ["/bin/bash", "-c", "source activate conda && /bin/bash -c 'python /app/caption.py'"]
