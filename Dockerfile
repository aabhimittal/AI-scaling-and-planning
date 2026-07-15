# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Multi-stage build: install into a venv in a builder stage, then copy just the
# runtime into a slim final image. Keeps the shipped image small and free of
# build tooling.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Create an isolated virtual environment we can copy wholesale later.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies first (better layer caching), then the package itself.
COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .

# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Run as an unprivileged user.
RUN useradd --create-home --uid 10001 appuser
WORKDIR /home/appuser

COPY --from=builder /opt/venv /opt/venv

USER appuser
EXPOSE 8000

# Basic container health check hitting the API's /health endpoint.
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

# Serve the API by default; override the command to run CLI subcommands.
CMD ["predictive-scaling", "serve", "--host", "0.0.0.0", "--port", "8000"]
