FROM cu124_torch25_base:latest AS base

# Disable Python output buffering
ENV PYTHONUNBUFFERED=1

RUN export MAKEFLAGS="-j$(nproc)" && \
    /opt/conda/bin/conda run -n conda pip install --no-cache-dir -v \
    accelerate bitsandbytes flash_attn huggingface_hub sentencepiece transformers==4.40.0

COPY src/ /app/
RUN chmod +x /app/main.py

WORKDIR /app/

# # Set entrypoint to activate Conda environment and start bash shell
# ENTRYPOINT ["/bin/bash", "-c", "source activate conda && /bin/bash -c 'python /app/main.py --model minicpm-v-2_6 --quantize'"]

# works, but output is not streamed in real-time
# ENTRYPOINT ["/opt/conda/bin/conda", "run", "-n", "conda", "python"]
# CMD ["/app/main.py"]


ENTRYPOINT ["bash", "-c", "source activate conda && exec python /app/main.py $*"]

# Default CMD with model and quantize flags
CMD ["--model", "minicpm-v-2_6", "--quantize"]