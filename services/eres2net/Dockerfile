ARG SHARED_BASE_IMAGE=zanzara-archive/shared-base:py314-cu130
FROM ${SHARED_BASE_IMAGE}

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
RUN git init /opt/3D-Speaker \
    && git -C /opt/3D-Speaker remote add origin https://github.com/alibaba-damo-academy/3D-Speaker.git \
    && git -C /opt/3D-Speaker fetch --depth 1 origin 065629c313eaf1a01c65c640c46d77e61e9607b4 \
    && git -C /opt/3D-Speaker checkout --detach --quiet FETCH_HEAD \
    && rm -rf /opt/3D-Speaker/.git
COPY services/eres2net/pyproject.toml services/eres2net/uv.lock /opt/service/
WORKDIR /opt/service
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --frozen --inexact --no-install-project --python /opt/venv/bin/python \
        --no-install-package torch --no-install-package torchaudio --no-install-package transformers
COPY services/common /opt/common
COPY services/eres2net/src /opt/service/src
ENV PATH=/opt/venv/bin:$PATH PYTHONPATH=/opt/common:/opt/service/src
CMD ["python", "-m", "eres2net_service.main"]
