FROM cu128_torch27_base:latest AS base

ARG JOBS=4
ENV MAKEFLAGS="-j${JOBS}"
ENV MAX_JOBS=${JOBS}

ENV FORCE_CUDA=1
# ENV TORCH_CUDA_ARCH_LIST="8.9"
ENV TORCH_CUDA_ARCH_LIST="12.0"

RUN /opt/conda/bin/conda run -n conda \
    pip install --upgrade \
    pip wheel setuptools ninja

RUN /opt/conda/bin/conda run -n conda \
    pip install --no-cache-dir -v --upgrade \
    accelerate bitsandbytes huggingface_hub sentencepiece timm 
    
#transformers==4.57.0
RUN /opt/conda/bin/conda run -n conda \
    pip install -v git+https://github.com/huggingface/transformers

# RUN /opt/conda/bin/conda run -n conda \
#     pip install vllm==0.9.1 blobfile --no-build-isolation
    
RUN /opt/conda/bin/conda run -n conda env MAX_JOBS=${MAX_JOBS} \
    pip install --no-cache-dir -v --no-build-isolation \
    flash-attn

# RUN  git clone https://github.com/AIDC-AI/AutoGPTQ.git && \
#     cd AutoGPTQ && \
#     /opt/conda/bin/conda run -n conda pip install -vvv --no-build-isolation -e .

RUN git clone https://github.com/WePOINTS/WePOINTS.git && \
    cd WePOINTS && \
    /opt/conda/bin/conda run -n conda pip install -vvv -e .

ADD requirements.txt /tmp/requirements.txt
RUN /opt/conda/bin/conda run -n conda pip install --no-cache-dir -r /tmp/requirements.txt

COPY src/ /app/
RUN chmod +x /app/main.py

WORKDIR /app/

ENTRYPOINT ["bash", "-c", "source activate conda && exec python /app/main.py \"$@\"", "--"]

# Default CMD with model and quantize flags
CMD ["--model", "internvl3", "--quantize"]
