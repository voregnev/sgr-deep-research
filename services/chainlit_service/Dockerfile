FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /app

# Copy the entire project first
COPY sgr_deep_research ./sgr_deep_research
COPY pyproject.toml ./
COPY README.md ./
COPY .chainlit ./.chainlit

# Create virtual environment and install the package
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /opt/venv && \
    uv pip install --python /opt/venv/bin/python -e .


FROM python:3.13-slim-bookworm AS runner

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PATH="/opt/venv/bin:$PATH"

RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN groupadd --gid 10001 appgroup && \
    useradd --no-create-home --shell /bin/false --uid 10000 --gid 10001 --system appuser

COPY --from=builder --chown=appuser:appgroup /opt/venv /opt/venv
COPY --from=builder --chown=appuser:appgroup /app/sgr_deep_research /app/sgr_deep_research
COPY --from=builder --chown=appuser:appgroup /app/pyproject.toml /app/
COPY --from=builder --chown=appuser:appgroup /app/README.md /app/
COPY --from=builder --chown=appuser:appgroup /app/.chainlit /app/.chainlit

WORKDIR /app
USER appuser:appgroup

# Create reports directory
RUN mkdir -p /app/reports

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run Chainlit application
CMD ["chainlit", "run", "sgr_deep_research/gui_app.py", "--host", "0.0.0.0", "--port", "8000"]