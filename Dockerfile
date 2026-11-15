ARG BASE_REGISTRY=docker.io
FROM ${BASE_REGISTRY}/library/python:3.14-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.10.10 /uv /uvx /usr/local/bin/

WORKDIR /app

COPY pyproject.toml uv.lock* ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

COPY src/ src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# spaCy NER model used by the eval-set sampler/redactor (Plan 3 Lane A).
# uv-managed venv has no `pip`; spacy download shells out to pip. Use
# `uv pip install` against the model wheel (published to PyPI) instead.
RUN uv pip install --python /app/.venv/bin/python en_core_web_sm


FROM ${BASE_REGISTRY}/library/python:3.14-slim AS runtime

RUN groupadd --gid 1000 app && \
    useradd --uid 1000 --gid app --shell /bin/bash --create-home app

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY src/ src/
COPY alembic.ini ./alembic.ini
COPY alembic/ ./alembic/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src

USER app

EXPOSE 8154

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8154", "--workers", "2"]
