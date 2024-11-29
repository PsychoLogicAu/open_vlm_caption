FROM cu124_torch25_base:latest AS base

RUN export MAKEFLAGS="-j$(nproc)" && \
    echo "MAKEFLAGS=${MAKEFLAGS}" && \
    /opt/conda/bin/conda run -n conda pip install --no-cache-dir -v \
    einops timm transformers

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

# If the repository has a requirements.txt file, install the dependencies
RUN test -f requirements.txt && \
    export MAKEFLAGS="-j$(nproc)" && \
    echo "MAKEFLAGS=${MAKEFLAGS}" && \
    /opt/conda/bin/conda run -n conda pip install --no-cache-dir -r requirements.txt || echo "No requirements.txt found"

# If setup.py exists, install the package, do not fail if it does not exist
RUN test -f setup.py && \
    export MAKEFLAGS="-j$(nproc)" && \
    echo "MAKEFLAGS=${MAKEFLAGS}" && \
    /opt/conda/bin/conda run -n conda pip install --no-cache-dir -e . || echo "No setup.py found"

ARG DIR
COPY ${DIR}/caption.py /app

# Set entrypoint to activate Conda environment and start bash shell
ENTRYPOINT ["/bin/bash", "-c", "source activate conda && /bin/bash -c 'python /app/caption.py'"]
