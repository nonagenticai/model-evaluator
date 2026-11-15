"""Structured logging with OpenTelemetry trace correlation.

Follows karma_service patterns: structlog with JSON output in production,
console renderer in development, and optional OTEL trace context injection.
"""

import logging
import os
import sys
import types
from typing import TYPE_CHECKING, Any

import structlog

# Check if OTEL is available
OTEL_AVAILABLE = False
if TYPE_CHECKING:
    trace: types.ModuleType | None = None
else:
    try:
        from opentelemetry import trace

        OTEL_AVAILABLE = True
    except ImportError:
        trace = None


def add_trace_context(
    logger: Any,
    _method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Inject OpenTelemetry trace context into structured logs."""
    if OTEL_AVAILABLE and trace:
        span = trace.get_current_span()
        if span and span.is_recording():
            ctx = span.get_span_context()
            if ctx.is_valid:
                event_dict["trace_id"] = format(ctx.trace_id, "032x")
                event_dict["span_id"] = format(ctx.span_id, "016x")
    return event_dict


def add_service_context(
    logger: Any,
    _method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Add service metadata to logs."""
    event_dict["service"] = os.getenv("OTEL_SERVICE_NAME", "model-evaluator")
    event_dict["environment"] = os.getenv("MODEL_EVALUATOR_ENVIRONMENT", "production")
    return event_dict


_configured: bool = False


def configure_logging(settings: Any = None) -> None:
    """Configure structured JSON logging with trace context.

    Args:
        settings: Optional Settings instance (uses ``log_format`` and ``log_level``
            attributes).  When *None*, values are read from environment variables.
    """
    global _configured

    if _configured:
        return

    if settings is not None and hasattr(settings, "log_format"):
        log_format = settings.log_format.lower()
    else:
        log_format = os.getenv("MODEL_EVALUATOR_LOG_FORMAT", "json").lower()

    if settings is not None and hasattr(settings, "log_level"):
        log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    else:
        log_level = getattr(
            logging,
            os.getenv("MODEL_EVALUATOR_LOG_LEVEL", "INFO").upper(),
        )

    processors: list[Any] = []
    if log_format == "json":
        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            add_service_context,
            add_trace_context,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ]
    else:
        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            add_service_context,
            add_trace_context,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=log_level)

    # Suppress noisy loggers
    for logger_name in ("httpx", "httpcore", "uvicorn.access", "urllib3"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a module-level structured logger.

    Args:
        name: Logger name (defaults to 'model_evaluator').

    Returns:
        Configured structlog BoundLogger instance.
    """
    configure_logging()
    logger: Any = structlog.get_logger(name or "model_evaluator")
    return logger
