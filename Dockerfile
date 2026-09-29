FROM ghcr.io/astral-sh/uv:0.12.6 AS uv
FROM python:3.12.14-slim

COPY --from=uv /uv /uvx /usr/local/bin/
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_CACHE_DIR=/tmp/uv-cache \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    OMP_NUM_THREADS=2
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY configs/ configs/
COPY src/ src/
RUN uv sync --frozen --extra cpu --no-dev && rm -rf /tmp/uv-cache
ENTRYPOINT ["/opt/venv/bin/python", "-m", "calolab_reco", "evaluate"]
CMD ["--help"]
