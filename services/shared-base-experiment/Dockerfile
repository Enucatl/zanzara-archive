# syntax=docker/dockerfile:1.7

FROM nvidia/cuda:13.0.2-base-ubuntu24.04@sha256:605fb0c8acf8674e164d822da8a8521f3a655056e569f0899e72ae940e1fe7dc

# Keep uv out of the package environment and avoid a per-image pip-install layer.
COPY --from=ghcr.io/astral-sh/uv:0.12.13@sha256:b485bd65cc2cf1c9a93b3554012c9c3778cf7b1b5fd3d3096ce9e1226c97e1e6 /uv /uvx /usr/local/bin/

ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_LINK_MODE=copy \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/uv \
    uv python install 3.14 \
    && uv venv --python 3.14 /opt/venv \
    && uv pip install --python /opt/venv/bin/python \
        --index-url https://download.pytorch.org/whl/cu130 \
        torch==2.14.0 \
    && uv pip install --python /opt/venv/bin/python \
        --index-url https://pypi.org/simple \
        torchaudio==2.11.0 \
        transformers==5.17.0 \
        accelerate==1.10.1 \
        numpy==2.4.6 \
        scipy==1.17.1 \
        scikit-learn==1.9.1 \
        soundfile==0.13.1 \
        pyyaml==6.0.2 \
        sentencepiece==0.2.2 \
    && /opt/venv/bin/python -c 'import sys, torch, torchaudio, transformers; print(sys.version); print(torch.__version__); print(torchaudio.__version__); print(transformers.__version__)'

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1
