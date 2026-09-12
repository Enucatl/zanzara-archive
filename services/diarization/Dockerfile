ARG SHARED_BASE_IMAGE=zanzara-archive/shared-base:py314-cu130
FROM ${SHARED_BASE_IMAGE}

COPY services/diarization/pyproject.toml services/diarization/uv.lock /opt/service/
WORKDIR /opt/service
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --frozen --inexact --no-install-project --python /opt/venv/bin/python \
        --no-install-package torch --no-install-package torchaudio --no-install-package transformers
COPY services/common /opt/common
COPY services/diarization/src /opt/service/src
ENV PATH=/opt/venv/bin:$PATH PYTHONPATH=/opt/common:/opt/service/src TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
CMD ["python", "-m", "diarization_service.main"]
