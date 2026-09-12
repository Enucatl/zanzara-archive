ARG SHARED_BASE_IMAGE=zanzara-archive/shared-base:py314-cu130
FROM ${SHARED_BASE_IMAGE}

COPY services/resnet293/pyproject.toml services/resnet293/uv.lock /opt/service/
WORKDIR /opt/service
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
# WeSpeaker vendors S3PRL code written for Torchaudio's removed global backend
# API. The setter was already a no-op; the MOS module's sox_effects import is
# also unused by ResNet293 but otherwise prevents the package importing on 2.11.
# Its fbank helper also returns CPU features even after set_device(); move those
# features to the selected device so the CUDA model can consume them.
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends git \
    && uv sync --frozen --inexact --no-install-project --python /opt/venv/bin/python \
        --no-install-package torch --no-install-package torchaudio --no-install-package transformers \
    && test -f /opt/venv/lib/python3.14/site-packages/s3prl/upstream/byol_s/byol_a/common.py \
    && sed -i '/^[[:space:]]*torchaudio\.set_audio_backend("sox_io")[[:space:]]*$/d' \
        /opt/venv/lib/python3.14/site-packages/s3prl/upstream/byol_s/byol_a/common.py \
    && ! grep -Fq 'torchaudio.set_audio_backend("sox_io")' \
        /opt/venv/lib/python3.14/site-packages/s3prl/upstream/byol_s/byol_a/common.py \
    && sed -i '/^[[:space:]]*from torchaudio\.sox_effects import apply_effects_tensor[[:space:]]*$/d' \
        /opt/venv/lib/python3.14/site-packages/s3prl/upstream/mos_prediction/expert.py \
    && ! grep -Fq 'from torchaudio.sox_effects import apply_effects_tensor' \
        /opt/venv/lib/python3.14/site-packages/s3prl/upstream/mos_prediction/expert.py \
    && sed -i 's/^[[:space:]]*feat = feat\.unsqueeze(0)  # \[1, T, D\]$/            feat = feat.unsqueeze(0).to(self.device)  # [1, T, D]/' \
        /opt/venv/lib/python3.14/site-packages/wespeaker/cli/speaker.py \
    && grep -Fq 'feat = feat.unsqueeze(0).to(self.device)' \
        /opt/venv/lib/python3.14/site-packages/wespeaker/cli/speaker.py \
    && apt-get purge -y --auto-remove git \
    && rm -rf /var/lib/apt/lists/*
COPY services/common /opt/common
COPY services/resnet293/src /opt/service/src
ENV PATH=/opt/venv/bin:$PATH PYTHONPATH=/opt/common:/opt/service/src
CMD ["python", "-m", "resnet293_service.main"]
