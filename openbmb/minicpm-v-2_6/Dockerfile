FROM cu124_torch25_base:latest AS base

RUN export MAKEFLAGS="-j$(nproc)" && \
    /opt/conda/bin/conda run -n conda pip install --no-cache-dir -v \
    accelerate bitsandbytes flash_attn huggingface_hub sentencepiece transformers==4.40.0

WORKDIR /app/

ARG DIR
COPY ${DIR}/caption.py /app

# Set entrypoint to activate Conda environment and start bash shell
ENTRYPOINT ["/bin/bash", "-c", "source activate conda && /bin/bash -c 'python /app/caption.py'"]
