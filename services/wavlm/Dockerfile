ARG SHARED_BASE_IMAGE=zanzara-archive/shared-base:py314-cu130
FROM ${SHARED_BASE_IMAGE}

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
RUN git init /opt/UniSpeech \
    && git -C /opt/UniSpeech remote add origin https://github.com/microsoft/UniSpeech.git \
    && git -C /opt/UniSpeech fetch --depth 1 origin 6112826ac13a4327f4c9a7afa2a505e35b763514 \
    && git -C /opt/UniSpeech checkout --detach --quiet FETCH_HEAD \
    && mkdir -p /opt/wavlm-runtime/models \
    && cp /opt/UniSpeech/downstreams/speaker_verification/models/ecapa_tdnn.py /opt/wavlm-runtime/models/ecapa_tdnn.py \
    && touch /opt/wavlm-runtime/models/__init__.py \
    && printf '%s\n' 'class UpstreamExpert: pass' > /opt/wavlm-runtime/models/utils.py \
    && rm -rf /opt/UniSpeech/.git
COPY services/wavlm/pyproject.toml services/wavlm/uv.lock /opt/service/
WORKDIR /opt/service
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --frozen --inexact --no-install-project --python /opt/venv/bin/python \
        --no-install-package torch --no-install-package torchaudio --no-install-package transformers
COPY services/common /opt/common
COPY services/wavlm/src /opt/service/src
ENV PATH=/opt/venv/bin:$PATH PYTHONPATH=/opt/common:/opt/service/src
CMD ["python", "-m", "wavlm_service.main"]
