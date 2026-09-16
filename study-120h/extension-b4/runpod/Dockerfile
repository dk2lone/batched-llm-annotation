# Build and publish only for a separately authorized Runpod smoke test.
ARG BASE_IMAGE=runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
FROM ${BASE_IMAGE}

# vLLM controls its torch dependency. Keep it isolated from the template's torch.
RUN uv python install 3.11 && uv venv --python 3.11 /opt/b4 && \
    uv pip install --python /opt/b4/bin/python \
      vllm==0.28.0 huggingface-hub==1.31.0 numpy==2.2.6 scipy==1.16.3
ENV PATH="/opt/b4/bin:${PATH}"
ENV HF_HOME=/workspace/hf HF_HUB_CACHE=/workspace/hf \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
    VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_BATCH_INVARIANT=1
