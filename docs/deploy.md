# Deploy

raggity ships a pre-built Docker image and a Compose file for one-command deployment with Qdrant.

---

## Docker quick start

```bash
# Clone the repo (or copy compose.yaml + raggity.toml to an empty directory)
git clone https://github.com/IxMxAMAR/raggity
cd raggity

# Copy and edit your config
cp raggity.example.toml raggity.toml
# Edit raggity.toml — set your backend, sources, etc.

# Start raggity + Qdrant
docker compose up
```

The raggity server is available at `http://localhost:8000`.

---

## Pre-built image (GHCR)

Tagged releases are automatically published to GitHub Container Registry:

```bash
docker pull ghcr.io/ixmxamar/raggity:latest

# Pin to a specific release
docker pull ghcr.io/ixmxamar/raggity:0.6.0
```

---

## Environment variables

Pass API keys via the environment or a `.env` file:

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Claude API key (if not using `claude login` subscription) |
| `OPENAI_API_KEY` | OpenAI-compatible backend key |
| `QDRANT_API_KEY` | Qdrant API key (if using a remote Qdrant Cloud instance) |

```bash
ANTHROPIC_API_KEY=sk-ant-... docker compose up
```

Or create a `.env` file and uncomment the `env_file` line in `compose.yaml`.

---

## Volumes

| Volume | Purpose |
|---|---|
| `./raggity.toml` | Config file — mounted read-only into `/app/raggity.toml` |
| `raggity_data` | Index data (LanceDB / embeddings) persisted across restarts |
| `qdrant_storage` | Qdrant vector store data |

---

## Optional Ollama sidecar

Run raggity against a local Ollama model instead of Claude or OpenAI:

```bash
docker compose --profile ollama up
```

Pull the model inside the running container:

```bash
docker compose exec ollama ollama pull llama3.1
```

Configure `raggity.toml`:

```toml
[generation]
backend = "ollama"
model = "llama3.1"
# base_url defaults to http://ollama:11434/v1 inside compose
```

---

## OpenTelemetry observability

raggity supports OpenTelemetry tracing and metrics export via the `otel` extra:

```bash
pip install raggity[otel]
```

Configure the OTLP endpoint:

```toml
[observability]
otel_endpoint = "http://localhost:4317"   # gRPC OTLP collector
```

With a local collector (e.g., OpenTelemetry Collector or Jaeger):

```bash
# Example: Jaeger all-in-one with OTLP gRPC
docker run -p 4317:4317 -p 16686:16686 jaegertracing/all-in-one:latest
```

Then set `otel_endpoint = "http://localhost:4317"` in `raggity.toml` and start raggity. Traces appear in the Jaeger UI at `http://localhost:16686`.

In Docker Compose, you can add a collector as an additional service and set the endpoint via environment variable.

---

## GitHub Actions CI / CD

raggity's own CI uses GitHub Actions:

- **Tests** — `.github/workflows/tests.yml` — runs the full test suite on Python 3.11 and 3.12 on every push and PR.
- **Publish** — `.github/workflows/publish.yml` — publishes to PyPI via OIDC Trusted Publishing on GitHub Release.
- **Docker** — `.github/workflows/docker.yml` — builds and pushes the GHCR image on Release.
- **Docs** — `.github/workflows/docs.yml` — builds and deploys this documentation site to GitHub Pages on every push to `main`.
