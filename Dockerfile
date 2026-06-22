# syntax=docker/dockerfile:1
#
# Linux/Python 3.12 image. We run the pipeline in a container because the host
# (Windows + Smart App Control) blocks unsigned native ML wheels; the container
# also makes builds host-independent and reproducible.

FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    MPLBACKEND=Agg

WORKDIR /app

# Dependency layer (cached unless requirements change).
COPY requirements.txt requirements-dev.txt ./
ARG INSTALL_DEV=true
RUN if [ "$INSTALL_DEV" = "true" ]; then \
        pip install -r requirements-dev.txt ; \
    else \
        pip install -r requirements.txt ; \
    fi

# Project layer.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install -e . --no-deps

# Runtime-only env (placed late to preserve the cached dependency layer):
# silence MLflow's git probe and opt out of third-party telemetry.
ENV GIT_PYTHON_REFRESH=quiet \
    MLFLOW_TELEMETRY_OPT_OUT=true \
    DO_NOT_TRACK=1

# Non-root runtime user.
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
CMD ["cardiorisk", "--help"]
