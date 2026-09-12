ARG SHARED_BASE_IMAGE=zanzara-archive/shared-base:py314-cu130
FROM ${SHARED_BASE_IMAGE}

COPY services/model-smoke/runner.py /opt/model-smoke/runner.py
ENTRYPOINT ["python", "/opt/model-smoke/runner.py"]
