"""OpenTelemetry setup with graceful degradation.

Tests connection to OTLP endpoint before enabling, falls back gracefully
if Grafana agent is unavailable, and respects OTEL_SDK_DISABLED env var.
Follows karma_service telemetry patterns.
"""

import logging
import os
import socket
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_initialized: bool = False


def _test_connection(host: str, port: int, timeout: float = 2.0) -> bool:
    """Test if OTLP endpoint is reachable."""
    try:
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.close()
        return True
    except (TimeoutError, ConnectionRefusedError, OSError):  # fmt: skip
        return False


def setup_telemetry(settings: Any = None, service_name: str = "model-evaluator") -> bool:
    """Initialize OpenTelemetry with graceful degradation.

    Accepts a *Settings* instance (with ``service_name`` and
    ``otlp_traces_endpoint`` attributes) **or** falls back to environment
    variables / defaults.

    Args:
        settings: Optional Settings object carrying service configuration.
        service_name: Fallback service name when *settings* is ``None``.

    Returns:
        True if OTEL was successfully initialized, False otherwise.
    """
    global _initialized

    if _initialized:
        return True

    if os.getenv("OTEL_SDK_DISABLED", "false").lower() == "true":
        logger.info("OTEL SDK disabled via environment variable")
        return False

    # Extract configuration from settings when available
    if settings is not None and hasattr(settings, "service_name"):
        service_name = settings.service_name

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as e:
        logger.warning("OTEL packages not available: %s", e)
        return False

    resource = Resource.create({SERVICE_NAME: service_name})
    provider = TracerProvider(resource=resource)

    if settings is not None and hasattr(settings, "otlp_traces_endpoint"):
        otlp_endpoint = settings.otlp_traces_endpoint
    else:
        otlp_endpoint = os.getenv("MODEL_EVALUATOR_OTLP_TRACES_ENDPOINT", "")

    if not otlp_endpoint:
        logger.info(
            "OTLP traces endpoint unset (settings.otlp_traces_endpoint / "
            "MODEL_EVALUATOR_OTLP_TRACES_ENDPOINT); tracing disabled."
        )
        return False

    try:
        parsed = urlparse(otlp_endpoint.replace("/v1/traces", ""))
        host = parsed.hostname or "alloy"
        port = parsed.port or 4318

        if _test_connection(host, port):
            exporter = OTLPSpanExporter(endpoint=otlp_endpoint, timeout=5)
            processor = BatchSpanProcessor(
                exporter,
                max_queue_size=2048,
                schedule_delay_millis=5000,
                max_export_batch_size=512,
                export_timeout_millis=30000,
            )
            provider.add_span_processor(processor)
            logger.info("OTEL exporter configured: %s", otlp_endpoint)
        else:
            logger.warning(
                "OTLP endpoint not reachable: %s:%s, tracing disabled",
                host,
                port,
            )
    except Exception as e:
        logger.warning("Failed to configure OTEL exporter: %s", e)

    trace.set_tracer_provider(provider)

    # Note: FastAPI instrumentation is handled in main.py lifespan where the
    # app instance is available.

    try:
        HTTPXClientInstrumentor().instrument()
        logger.info("HTTPX instrumentation enabled")
    except Exception as e:
        logger.warning("HTTPX instrumentation failed: %s", e)

    _initialized = True
    return True
