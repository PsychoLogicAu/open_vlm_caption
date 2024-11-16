FROM cu124_torch25_base:latest AS base

RUN export MAKEFLAGS="-j$(nproc)" && \
    /opt/conda/bin/conda run -n conda pip install --no-cache-dir -v \
    accelerate bitsandbytes flash_attn qwen-vl-utils transformers

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

ARG DIR
COPY ${DIR}/caption.py /app

# Set entrypoint to activate Conda environment and start bash shell
ENTRYPOINT ["/bin/bash", "-c", "source activate conda && /bin/bash -c 'python /app/caption.py'"]
