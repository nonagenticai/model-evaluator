# Model Evaluator

Validates new LoRA-adapted models before deployment by running inference against ground-truth
data. Port 8154.

| | |
|---|---|
| **Stack** | Python 3.14 · FastAPI · SQLAlchemy · Pydantic |
| **Database** | PostgreSQL 18 |

## What Model Evaluator Does

When a training trigger or a manual API call starts an evaluation, this service receives a model
ID, schema ID, and an inference endpoint URL. It creates a pending evaluation record in
PostgreSQL, then runs an asynchronous background pipeline: fetch ground-truth samples, load the
LoRA adapter, run inference on every sample, compute metrics, and gate the result for deployment.

The pipeline pulls validated samples from the data-collector service, filtered by schema ID. It
then calls an adapter-manager sidecar to hot-swap the candidate LoRA adapter into the running
inference process. With the adapter loaded, the evaluator sends each ground-truth sample through
the inference endpoint using the OpenAI-compatible chat completion API, optionally constraining
output with a GBNF grammar. For each sample it records whether the output is valid JSON, whether
it matches the expected JSON schema, whether it exactly matches the ground-truth response, and
how long inference took.

After all samples are processed, the metrics service computes aggregate rates (JSON parse rate,
schema validity rate, exact match rate) and combines them into a weighted composite score. Two
scoring profiles are available: the default weights exact match at 0.50, schema validity at 0.35,
and JSON parsing at 0.15; the structural profile shifts weight toward schema validity (0.60) for
schemas where outputs are large or variable. If the composite score meets the threshold (default
0.85), the evaluation is auto-approved and the model-registry is notified to promote the model.
If it falls short, the evaluation enters a "completed" state where a reviewer can manually approve
or reject it.

Per-sample results are stored alongside the aggregate metrics. All state transitions are persisted
to PostgreSQL. The readiness probe checks the database and upstream health, reporting degraded
status when upstreams are unreachable without failing the probe entirely.

## Features

- **Async evaluation pipeline** — inference runs in the background; callers poll for status
- **LoRA adapter hot-swap** — loads the candidate adapter via an adapter-manager sidecar before
  evaluation
- **Composite scoring with profiles** — combines JSON parse rate, schema validity, and exact
  match with configurable weights
- **Auto-approval gate** — approves models above the threshold and notifies the model-registry
- **Manual review workflow** — below-threshold evaluations can be approved or rejected with a
  documented reason
- **GBNF grammar constraints** — optional grammars match production decoding conditions
- **Per-sample result storage** — individual predictions, expected values, and validity flags
- **System prompt injection** — uses the routing rule's production system prompt
- **Degraded readiness reporting** — checks the database and upstreams concurrently
- **OpenTelemetry tracing** — instruments FastAPI and HTTPX with graceful degradation

## Quick Start

With Docker Compose (service + PostgreSQL 18):

```bash
docker compose up --build
curl -s http://localhost:8154/health/ready
```

With `uv` and services you already run:

```bash
uv sync
MODEL_EVALUATOR_ENVIRONMENT=development \
MODEL_EVALUATOR_DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5456/model_evaluator \
MODEL_EVALUATOR_DATA_COLLECTOR_URL=http://localhost:8151 \
MODEL_EVALUATOR_MODEL_REGISTRY_URL=http://localhost:8155 \
  uv run uvicorn main:app --host 0.0.0.0 --port 8154 --app-dir src
```

Migrations run automatically to `head` on startup.

## API Endpoints

### Evaluations `/api/evaluations`

| Method | Path | Description |
|--------|------|-------------|
| POST | `/` | Start a new evaluation (runs asynchronously) |
| GET | `/` | List evaluations with schema/status filters and pagination |
| GET | `/{eval_id}` | Evaluation details including metrics and per-sample results |
| POST | `/{eval_id}/approve` | Manually approve a completed evaluation |
| POST | `/{eval_id}/reject` | Manually reject a completed evaluation (reason required) |

### Health

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service info and endpoint listing |
| GET | `/health` | Liveness probe |
| GET | `/health/live` | Lightweight liveness probe |
| GET | `/health/ready` | Readiness probe (database and upstream health) |

## Configuration

All variables are prefixed with `MODEL_EVALUATOR_`.

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_EVALUATOR_SERVICE_NAME` | `model-evaluator` | Service name for telemetry |
| `MODEL_EVALUATOR_PORT` | `8154` | HTTP listen port |
| `MODEL_EVALUATOR_DATABASE_URL` | `""` | PostgreSQL DSN (`postgresql+psycopg://…`); required in production |
| `MODEL_EVALUATOR_DATA_COLLECTOR_URL` | `http://data-collector.example.com:8151` | Data-collector URL |
| `MODEL_EVALUATOR_MODEL_REGISTRY_URL` | `http://model-registry.example.com:8155` | Model-registry URL |
| `MODEL_EVALUATOR_ADAPTER_MANAGER_URL` | `http://vllm.example.com:8001` | Adapter-manager sidecar URL |
| `MODEL_EVALUATOR_DEFAULT_THRESHOLD` | `0.85` | Composite score threshold for auto-approval (0–1) |
| `MODEL_EVALUATOR_INFERENCE_ENDPOINT_URL` | `http://fg-entity-spotter.example.com:8080` | Default inference endpoint for health checks |
| `MODEL_EVALUATOR_INFERENCE_TIMEOUT` | `30.0` | Inference request timeout in seconds |
| `MODEL_EVALUATOR_OTLP_TRACES_ENDPOINT` | `http://localhost:4318/v1/traces` | OTLP HTTP trace endpoint |
| `MODEL_EVALUATOR_ENVIRONMENT` | `production` | `production` enables the required-field check |
| `MODEL_EVALUATOR_LOG_LEVEL` | `INFO` | Root log level |
| `MODEL_EVALUATOR_LOG_FORMAT` | `json` | Log format (`json` or `console`) |

## Architecture

```
src/
├── main.py                  # FastAPI app, lifespan, health endpoints
├── settings.py              # Pydantic BaseSettings (MODEL_EVALUATOR_ prefix)
├── db.py                    # async SQLAlchemy engine and session management
├── logging_config.py        # structlog configuration
├── telemetry.py             # OpenTelemetry setup with graceful degradation
├── models/
│   ├── db.py                # SQLAlchemy ORM (EvaluationORM)
│   └── domain.py            # Pydantic request/response schemas and enums
├── routers/
│   └── evaluations.py       # /api/evaluations CRUD, approve, reject
└── services/
    ├── evaluation_service.py  # background evaluation pipeline
    ├── inference_client.py    # OpenAI-compatible inference calls
    ├── metrics_service.py     # composite scoring
    ├── collector_client.py    # data-collector client
    ├── registry_client.py     # model-registry client
    └── adapter_client.py      # adapter-manager client
```

## Testing

The suite requires PostgreSQL. With no DSN set it starts an ephemeral `postgres:18` container
(fail-closed when Docker is unavailable). Set `TEST_DATABASE_URL` or `DATABASE_URL` to use an
existing server.

```bash
docker run -d --name me-testpg -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=model_evaluator -p 5456:5432 postgres:18-alpine
TEST_DATABASE_URL="postgresql+psycopg://postgres:postgres@localhost:5456/model_evaluator" \
  uv run pytest
```

Lint and format:

```bash
uv run ruff check .
uv run ruff format --check .
```

## CI

`.github/workflows/ci-public.yml` runs on GitHub-hosted runners: betterleaks secret scan, the
private-reference guard, ruff, the full pytest suite against a PostgreSQL 18 service container,
and `pip-audit`. The image build/publish job is off unless the repository variable
`PUBLISH_IMAGE=true` is set.

## Docker

```bash
docker build -t model-evaluator .
docker run --rm -p 8154:8154 \
  -e MODEL_EVALUATOR_ENVIRONMENT=development \
  -e MODEL_EVALUATOR_DATABASE_URL=postgresql+psycopg://postgres:postgres@host.docker.internal:5456/model_evaluator \
  model-evaluator
```

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Licensing

There is no licence gate in this service: it needs no licence key, token, or network call to
build, test or run. Every feature is available by default.
