"""FastAPI application with async lifespan for the model-evaluator service."""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, cast

import httpx
import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from db import dispose_engine, get_engine_for_check, init_db
from logging_config import configure_logging
from models.domain import HealthResponse, LivenessResponse, ReadinessResponse
from routers.evaluations import router as evaluations_router
from routers.evaluations import set_eval_service
from settings import get_settings
from telemetry import setup_telemetry

logger = structlog.stdlib.get_logger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None]:
    """Manage startup and shutdown resources."""
    settings = get_settings()

    # Logging
    configure_logging(settings)
    logger.info("starting", service=settings.service_name, port=settings.port)

    # Telemetry (graceful degradation)
    setup_telemetry(settings)

    # Instrument FastAPI with OpenTelemetry (best-effort)
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(application)
    except Exception:
        logger.warning("fastapi_otel_instrumentation_failed", exc_info=True)

    # Database
    await init_db(settings)
    logger.info("database_initialised", url=settings.database_url.split("@")[-1])

    # Shared HTTP clients -- reuse TCP connections across requests
    from services.adapter_client import AdapterClient
    from services.collector_client import CollectorClient
    from services.eval_service import EvalService
    from services.inference_client import InferenceClient
    from services.metrics_service import MetricsService
    from services.registry_client import RegistryClient

    inference_http = httpx.AsyncClient(timeout=settings.inference_timeout)
    collector_http = httpx.AsyncClient(timeout=settings.inference_timeout)
    registry_http = httpx.AsyncClient(timeout=settings.inference_timeout)
    adapter_http = httpx.AsyncClient(timeout=settings.inference_timeout)

    inference_client = InferenceClient(
        timeout=settings.inference_timeout,
        http_client=inference_http,
    )
    collector_client = CollectorClient(
        base_url=settings.data_collector_url,
        timeout=settings.inference_timeout,
        http_client=collector_http,
    )
    registry_client = RegistryClient(
        base_url=settings.model_registry_url,
        timeout=settings.inference_timeout,
        http_client=registry_http,
    )
    adapter_client = AdapterClient(
        base_url=settings.adapter_manager_url,
        http_client=adapter_http,
    )

    set_eval_service(
        EvalService(
            collector_client=collector_client,
            inference_client=inference_client,
            metrics_service=MetricsService(),
            registry_client=registry_client,
            adapter_client=adapter_client,
            settings=settings,
        )
    )
    logger.info("shared_http_clients_created")

    yield

    # Shutdown
    logger.info("shutting_down", service=settings.service_name)
    await inference_client.close()
    await collector_client.close()
    # RegistryClient and AdapterClient don't own their shared http_client --
    # we close the underlying httpx.AsyncClient instances directly here.
    await registry_http.aclose()
    await adapter_http.aclose()
    set_eval_service(None)
    logger.info("shared_http_clients_closed")
    await dispose_engine()


_settings = get_settings()

app = FastAPI(
    title="Model Evaluator",
    description="Validates new models before deployment by running inference on actual Pi hardware",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware -- must be added before routers
app.add_middleware(
    cast(Any, CORSMiddleware),
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(evaluations_router)


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["health"],
    summary="Health check",
    description="Liveness probe that always returns OK if the process is up.",
)
async def health() -> HealthResponse:
    """Liveness probe -- always returns OK if the process is up."""
    return HealthResponse(status="ok", service="model-evaluator", version="0.1.0")


@app.get(
    "/health/live",
    response_model=LivenessResponse,
    tags=["health"],
    summary="Liveness probe",
    description="Kubernetes liveness probe. Always returns 200 if the process is alive.",
)
async def liveness() -> LivenessResponse:
    """Kubernetes liveness probe. Always returns 200 if the process is alive."""
    return LivenessResponse()


async def _check_upstream(url: str, timeout: float = 2.0) -> bool:
    """Check an upstream service health endpoint with a short timeout.

    Non-blocking: returns False on any error instead of raising.
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url)
            return response.status_code == 200
    except Exception:
        logger.debug("upstream_health_check_failed", url=url, exc_info=True)
        return False


@app.get(
    "/health/ready",
    response_model=ReadinessResponse,
    tags=["health"],
    summary="Readiness probe",
    description="Checks database and upstream service health. Returns 200 even when upstreams are down to avoid failing deploys, but reports degraded status.",
)
async def readiness() -> ReadinessResponse:
    """Readiness probe -- checks database and upstream service health.

    The database connection is the primary readiness gate. Upstream services
    (inference endpoint, data-collector) are checked concurrently with a 2s
    timeout but do not block readiness -- they report degraded status instead.
    Still returns 200 if the database is up to avoid failing deploys for
    transient upstream issues.
    """
    settings = get_settings()

    # Run all checks concurrently
    db_ok, inference_ok, collector_ok = await asyncio.gather(
        get_engine_for_check(),
        _check_upstream(f"{settings.inference_endpoint_url.rstrip('/')}/health"),
        _check_upstream(f"{settings.data_collector_url.rstrip('/')}/health"),
    )

    # Determine overall status: ready if DB is up, degraded if upstreams are down
    if not db_ok:
        status = "not_ready"
    elif not inference_ok or not collector_ok:
        status = "degraded"
    else:
        status = "ready"

    return ReadinessResponse(
        status=status,
        database="connected" if db_ok else "disconnected",
        inference="ok" if inference_ok else "error",
        collector="ok" if collector_ok else "error",
    )


@app.get(
    "/",
    tags=["health"],
    summary="Root",
    description="Root endpoint returning service metadata and documentation links.",
)
async def root() -> dict[str, str]:
    """Root endpoint with service metadata."""
    return {
        "service": _settings.service_name,
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler for unhandled exceptions."""
    from fastapi import HTTPException

    if isinstance(exc, HTTPException):
        raise exc

    logger.error("Unhandled error: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "Internal server error",
                "type": "server_error",
                "code": 500,
            }
        },
    )
