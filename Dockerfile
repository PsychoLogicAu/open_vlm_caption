FROM cu124_torch25_base:latest AS base

RUN export MAKEFLAGS="-j$(nproc)" && \
    /opt/conda/bin/conda run -n conda pip install --no-cache-dir -v \
    accelerate bitsandbytes flash_attn huggingface_hub sentencepiece timm transformers

COPY src/ /app/
RUN chmod +x /app/main.py

WORKDIR /app/

ENTRYPOINT ["bash", "-c", "source activate conda && exec python /app/main.py $*"]

# Default CMD with model and quantize flags
CMD ["--model", "minicpm-v-2_6", "--quantize"]
