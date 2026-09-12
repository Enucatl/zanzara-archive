# syntax=docker/dockerfile:1.7

# The only CUDA/PyTorch lineage in the service tree starts here.  The upstream
# digest is immutable; the promoted local image digest is recorded in
# models.lock.json after the build.
FROM nvidia/cuda:13.0.2-base-ubuntu24.04@sha256:605fb0c8acf8674e164d822da8a8521f3a655056e569f0899e72ae940e1fe7dc

COPY --from=ghcr.io/astral-sh/uv:0.12.13@sha256:b485bd65cc2cf1c9a93b3554012c9c3778cf7b1b5fd3d3096ce9e1226c97e1e6 /uv /uvx /usr/local/bin/

ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=0 \
    TRITON_CACHE_DIR=/opt/triton-cache \
    PATH=/opt/venv/bin:$PATH

# This layer is the single shared Python/ML stack.  The uv cache is a bounded
# BuildKit cache mount and is never copied into the image.
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libsndfile1 \
        ffmpeg gcc libc6-dev \
    && uv python install 3.14 \
    && uv venv --python 3.14 /opt/venv \
    && uv pip install --python /opt/venv/bin/python \
        --index-url https://download.pytorch.org/whl/cu130 \
        torch==2.14.0 torchaudio==2.11.0 \
    && uv pip install --python /opt/venv/bin/python \
        --index-url https://pypi.org/simple \
        accelerate==1.10.1 \
        numpy==2.5.3 \
        scipy==1.18.1 \
        scikit-learn==1.9.1 \
        soundfile==0.13.1 \
        pyyaml==6.0.3 \
        sentencepiece==0.2.2 \
        torchcodec==0.16.0 \
        transformers==5.17.0 \
    && /opt/venv/bin/python -c 'import sys, torch, torchaudio, transformers; assert sys.version_info[:2] == (3, 14); print(sys.version); print(torch.__version__); print(torchaudio.__version__); print(transformers.__version__)' \
    # Triton ships ptxas for CUDA kernels but JIT-builds one small C driver \
    # helper. Compile that helper once, then remove the host compiler from the \
    # final inference image; Triton does not require nvcc at runtime. \
    && ln -sf /usr/local/cuda-13.0/compat/libcuda.so.1 /lib/x86_64-linux-gnu/libcuda.so.1 \
    && ldconfig \
    && /opt/venv/bin/python -c \
        'from triton.backends.nvidia.driver import CudaUtils; CudaUtils()' \
    && rm -f /lib/x86_64-linux-gnu/libcuda.so.1 \
    && ldconfig \
    && apt-get purge -y --auto-remove gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONDONTWRITEBYTECODE=1
