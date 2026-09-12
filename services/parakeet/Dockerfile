ARG SHARED_BASE_IMAGE=zanzara-archive/shared-base:py314-cu130
FROM ${SHARED_BASE_IMAGE}

COPY services/parakeet/pyproject.toml services/parakeet/uv.lock /opt/service/
WORKDIR /opt/service
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --frozen --inexact --no-install-project --python /opt/venv/bin/python \
        --no-install-package torch --no-install-package torchaudio --no-install-package transformers
COPY services/common /opt/common
COPY services/parakeet/src /opt/service/src
ENV PATH=/opt/venv/bin:$PATH PYTHONPATH=/opt/common:/opt/service/src
CMD ["python", "-m", "parakeet_service.main"]
