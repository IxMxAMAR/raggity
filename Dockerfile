# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Stage 1 — builder: compile wheels for the server + storage extras.
#   We install build tools here and discard them in the final image.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# Copy only what pip needs to resolve the package
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

# Build a wheel so the final stage can pip-install without build tools
RUN pip install --upgrade pip && \
    pip wheel --no-cache-dir --wheel-dir /wheels ".[server,openai,qdrant]"

# ---------------------------------------------------------------------------
# Stage 2 — runtime: slim image, no compilers, no build artefacts.
# ---------------------------------------------------------------------------
FROM python:3.12-slim

LABEL org.opencontainers.image.source="https://github.com/IxMxAMAR/raggity"
LABEL org.opencontainers.image.description="Local-first RAG answered by Claude — server image"
LABEL org.opencontainers.image.licenses="AGPL-3.0-or-later"

WORKDIR /app

# Install the pre-built wheels from the builder stage
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels raggity[server,openai,qdrant] && \
    rm -rf /wheels

# Runtime config location (mount raggity.toml here)
VOLUME ["/app/data"]

EXPOSE 8000

CMD ["rag", "serve", "--host", "0.0.0.0", "--port", "8000"]
