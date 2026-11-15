"""Shared fixtures for model-evaluator tests.

DB-backed tests require PostgreSQL via TEST_DATABASE_URL (or DATABASE_URL; CI exports
DATABASE_URL). There is NO host:5432 default -- it silently hits whatever runs on that
port on a dev box. If neither is set, an ephemeral docker postgres:18 is started for the
session (tests/ephemeral_postgres.py); without docker the session stops (fail closed).
If the database is unreachable the tests will fail loudly with the real
SQLAlchemy error -- do not silently skip, it hides real infra issues.
"""

import os

# Main module calls get_settings() at import time; ensure the production
# validator does not fire during tests by forcing a non-production env.
# The URLs here are also used by readiness-probe tests that assert on
# substrings like "data-collector" / "inference-router" in the probe URL.
os.environ.setdefault("MODEL_EVALUATOR_ENVIRONMENT", "test")
os.environ.setdefault(
    "MODEL_EVALUATOR_DATA_COLLECTOR_URL",
    "http://data-collector.example.com:8151",
)
os.environ.setdefault(
    "MODEL_EVALUATOR_INFERENCE_ENDPOINT_URL",
    "http://inference-router.example.com:8150",
)

from collections.abc import AsyncGenerator  # noqa: E402
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from db import Base, get_session
from main import app
from models.db import EvaluationORM  # noqa: F401 -- register model on Base.metadata
from routers.evaluations import _get_eval_service
from services.adapter_client import AdapterClient
from services.collector_client import CollectorClient
from services.eval_service import EvalService
from services.inference_client import InferenceClient
from services.metrics_service import MetricsService
from services.registry_client import RegistryClient
from tests.ephemeral_postgres import (
    resolve_test_database_url,
    start_session_database,
    stop_session_database,
)

# ---------------------------------------------------------------------------
# Engine & session fixtures
# ---------------------------------------------------------------------------


def pytest_configure(config: pytest.Config) -> None:
    """Start the ONE session database in the xdist controller (no-op in workers), before
    workers spawn, so they inherit the exported DSN instead of starting their own."""
    start_session_database(
        env_vars=("TEST_DATABASE_URL", "DATABASE_URL"),
        export_var="TEST_DATABASE_URL",
        dbname="model_evaluator_test",
        config=config,
    )


def pytest_unconfigure(config: pytest.Config) -> None:
    """Controller only: remove the session container by its own ID."""
    stop_session_database(config)


@pytest.fixture(scope="session")
def test_database_url(request: pytest.FixtureRequest) -> str:
    """TEST_DATABASE_URL / DATABASE_URL, else an ephemeral docker postgres:18 for this
    session, else the session stops (fail closed)."""
    return resolve_test_database_url(
        env_vars=("TEST_DATABASE_URL", "DATABASE_URL"),
        export_var="TEST_DATABASE_URL",
        dbname="model_evaluator_test",
        request=request,
    )


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture()
async def engine(test_database_url: str) -> AsyncGenerator[Any]:
    """Create a fresh PostgreSQL engine for each test.

    Uses DROP SCHEMA CASCADE to ensure stale composite types (from ARRAY
    columns) and lingering connections from previous CI runs are cleaned up.
    If the database is unreachable the underlying SQLAlchemy error is
    allowed to propagate so CI shows the real infra issue instead of
    silently skipping tests.
    """
    eng = create_async_engine(test_database_url, echo=False)
    async with eng.begin() as conn:
        await conn.execute(
            text(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                "WHERE datname = current_database() AND pid != pg_backend_pid() "
                "AND backend_type = 'client backend'"
            )
        )
        await conn.execute(text("DROP SCHEMA public CASCADE"))
        await conn.execute(text("CREATE SCHEMA public"))
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    async with eng.begin() as conn:
        await conn.execute(text("DROP SCHEMA public CASCADE"))
        await conn.execute(text("CREATE SCHEMA public"))
    await eng.dispose()


@pytest.fixture()
async def session(engine) -> AsyncGenerator[AsyncSession]:
    """Yield a fresh session per test with a clean database."""
    session_factory = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    async with session_factory() as sess:
        yield sess


# ---------------------------------------------------------------------------
# Mock service fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_collector_client() -> CollectorClient:
    """Return a mock CollectorClient."""
    client = AsyncMock(spec=CollectorClient)
    client.get_validated_samples = AsyncMock(return_value=[])
    client.health_check = AsyncMock(return_value=True)
    client.close = AsyncMock()
    return client


@pytest.fixture()
def mock_inference_client() -> InferenceClient:
    """Return a mock InferenceClient."""
    client = AsyncMock(spec=InferenceClient)
    client.complete = AsyncMock(
        return_value={"choices": [{"message": {"content": '{"result": "test"}'}}]}
    )
    client.health_check = AsyncMock(return_value=True)
    client.close = AsyncMock()
    return client


@pytest.fixture()
def mock_registry_client() -> RegistryClient:
    """Return a mock RegistryClient."""
    client = AsyncMock(spec=RegistryClient)
    client.update_model_status = AsyncMock(return_value={"status": "ok"})
    client.report_evaluation = AsyncMock(return_value={"status": "ok"})
    client.health_check = AsyncMock(return_value=True)
    return client


@pytest.fixture()
def metrics_service() -> MetricsService:
    """Return a real MetricsService (it is stateless)."""
    return MetricsService()


@pytest.fixture()
def mock_adapter_client() -> AdapterClient:
    """Return a mock AdapterClient."""
    client = AsyncMock(spec=AdapterClient)
    client.load_adapter = AsyncMock(return_value={"status": "loaded"})
    client.get_loaded_adapters = AsyncMock(return_value={"adapters": []})
    client.health_check = AsyncMock(return_value=True)
    return client


@pytest.fixture()
def eval_service(
    mock_collector_client: CollectorClient,
    mock_inference_client: InferenceClient,
    metrics_service: MetricsService,
    mock_registry_client: RegistryClient,
) -> EvalService:
    """Return an EvalService with mocked external clients."""
    return EvalService(
        collector_client=mock_collector_client,
        inference_client=mock_inference_client,
        metrics_service=metrics_service,
        registry_client=mock_registry_client,
    )


@pytest.fixture()
def eval_service_with_adapter(
    mock_collector_client: CollectorClient,
    mock_inference_client: InferenceClient,
    metrics_service: MetricsService,
    mock_registry_client: RegistryClient,
    mock_adapter_client: AdapterClient,
) -> EvalService:
    """Return an EvalService with mocked external clients including adapter."""
    return EvalService(
        collector_client=mock_collector_client,
        inference_client=mock_inference_client,
        metrics_service=metrics_service,
        registry_client=mock_registry_client,
        adapter_client=mock_adapter_client,
    )


@pytest.fixture()
async def client(
    session: AsyncSession,
    eval_service: EvalService,
) -> AsyncGenerator[AsyncClient]:
    """HTTPX async client with the FastAPI app, using the test session and mocked services."""

    async def _override_session() -> AsyncGenerator[AsyncSession]:
        yield session

    def _override_eval_service() -> EvalService:
        return eval_service

    app.dependency_overrides[get_session] = _override_session
    app.dependency_overrides[_get_eval_service] = _override_eval_service

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac

    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EVAL_PAYLOAD: dict[str, Any] = {
    "model_id": str(uuid4()),
    "schema_id": "ks_EntitySpotterAgent",
    "endpoint": "http://fg-entity-spotter.example.com:8080",
    "threshold": 0.85,
    "max_samples": 50,
    "json_schema": {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "type": {"type": "string"},
                    },
                    "required": ["text", "type"],
                },
            }
        },
        "required": ["entities"],
    },
    "gbnf_grammar": None,
}


@pytest.fixture()
def eval_payload() -> dict[str, Any]:
    """Return a valid evaluation creation payload."""
    return {**EVAL_PAYLOAD, "model_id": str(uuid4())}


GROUND_TRUTH_SAMPLES: list[dict[str, Any]] = [
    {
        "id": str(uuid4()),
        "messages": [
            {"role": "system", "content": "You are an entity spotter."},
            {"role": "user", "content": "Find entities in: Apple announced new iPhones."},
        ],
        "response_content": '{"entities": [{"text": "Apple", "type": "ORG"}]}',
    },
    {
        "id": str(uuid4()),
        "messages": [
            {"role": "system", "content": "You are an entity spotter."},
            {"role": "user", "content": "Find entities in: Google hired John."},
        ],
        "response_content": '{"entities": [{"text": "Google", "type": "ORG"}, {"text": "John", "type": "PERSON"}]}',
    },
    {
        "id": str(uuid4()),
        "messages": [
            {"role": "system", "content": "You are an entity spotter."},
            {"role": "user", "content": "Find entities in: No entities here."},
        ],
        "response_content": '{"entities": []}',
    },
]


@pytest.fixture()
def ground_truth_samples() -> list[dict[str, Any]]:
    """Return sample ground truth data for evaluation tests."""
    return [s.copy() for s in GROUND_TRUTH_SAMPLES]
