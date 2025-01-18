FROM cu124_torch25_base:latest AS base

RUN export MAKEFLAGS="-j$(nproc)" && \
    /opt/conda/bin/conda run -n conda pip install --no-cache-dir -v \
    accelerate bitsandbytes flash_attn huggingface_hub sentencepiece timm transformers

ENV TORCH_CUDA_ARCH_LIST="8.9"
RUN export MAKEFLAGS="-j$(nproc)" && \ 
    git clone https://github.com/AIDC-AI/AutoGPTQ.git && \
    cd AutoGPTQ && \
    /opt/conda/bin/conda run -n conda pip install -vvv --no-build-isolation -e .

RUN export MAKEFLAGS="-j$(nproc)" && \
    git clone https://github.com/WePOINTS/WePOINTS.git && \
    cd WePOINTS && \
    /opt/conda/bin/conda run -n conda pip install -vvv -e .

ADD requirements.txt /tmp/requirements.txt
RUN export MAKEFLAGS="-j$(nproc)" && \
    /opt/conda/bin/conda run -n conda pip install --no-cache-dir -r /tmp/requirements.txt

COPY src/ /app/
RUN chmod +x /app/main.py

WORKDIR /app/

ENTRYPOINT ["bash", "-c", "source activate conda && exec python /app/main.py $*"]

# Default CMD with model and quantize flags
CMD ["--model", "minicpm-v-2_6", "--quantize"]
