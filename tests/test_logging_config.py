"""Tests for the logging_config module.

Covers:
- add_trace_context: injects OTEL trace/span IDs when available
- add_service_context: injects service metadata
- configure_logging: JSON and console modes, idempotency
- get_logger: returns configured logger
"""

import os
from unittest.mock import MagicMock, patch


class TestAddServiceContext:
    """Tests for add_service_context."""

    def test_adds_default_service_name(self) -> None:
        """Should add service=model-evaluator by default."""
        from logging_config import add_service_context

        event_dict: dict = {}
        result = add_service_context(None, "info", event_dict)
        assert result["service"] == "model-evaluator"

    def test_uses_env_service_name(self) -> None:
        """Should use OTEL_SERVICE_NAME env var if set."""
        from logging_config import add_service_context

        with patch.dict(os.environ, {"OTEL_SERVICE_NAME": "custom-svc"}):
            event_dict: dict = {}
            result = add_service_context(None, "info", event_dict)
            assert result["service"] == "custom-svc"

    def test_adds_environment(self) -> None:
        """Should add environment field."""
        from logging_config import add_service_context

        with patch.dict(os.environ, {"MODEL_EVALUATOR_ENVIRONMENT": "staging"}):
            event_dict: dict = {}
            result = add_service_context(None, "info", event_dict)
            assert result["environment"] == "staging"


class TestAddTraceContext:
    """Tests for add_trace_context."""

    def test_no_trace_when_otel_unavailable(self) -> None:
        """Should return event_dict unchanged when OTEL is not available."""
        from logging_config import add_trace_context

        with patch("logging_config.OTEL_AVAILABLE", False):
            event_dict: dict = {"event": "test"}
            result = add_trace_context(None, "info", event_dict)
            assert "trace_id" not in result
            assert "span_id" not in result

    def test_adds_trace_when_span_active(self) -> None:
        """Should inject trace/span IDs when a valid span is recording."""
        from logging_config import add_trace_context

        mock_ctx = MagicMock()
        mock_ctx.is_valid = True
        mock_ctx.trace_id = 0x1234567890ABCDEF1234567890ABCDEF
        mock_ctx.span_id = 0xFEDCBA0987654321

        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        mock_span.get_span_context.return_value = mock_ctx

        mock_trace = MagicMock()
        mock_trace.get_current_span.return_value = mock_span

        with (
            patch("logging_config.OTEL_AVAILABLE", True),
            patch("logging_config.trace", mock_trace),
        ):
            event_dict: dict = {"event": "test"}
            result = add_trace_context(None, "info", event_dict)
            assert "trace_id" in result
            assert "span_id" in result

    def test_no_trace_when_span_not_recording(self) -> None:
        """Should not inject trace data when span is not recording."""
        from logging_config import add_trace_context

        mock_span = MagicMock()
        mock_span.is_recording.return_value = False

        mock_trace = MagicMock()
        mock_trace.get_current_span.return_value = mock_span

        with (
            patch("logging_config.OTEL_AVAILABLE", True),
            patch("logging_config.trace", mock_trace),
        ):
            event_dict: dict = {"event": "test"}
            result = add_trace_context(None, "info", event_dict)
            assert "trace_id" not in result


class TestConfigureLogging:
    """Tests for configure_logging."""

    def test_configure_logging_json_format(self) -> None:
        """Should configure structlog with JSON renderer."""
        import logging_config

        # Reset the configured flag
        orig = logging_config._configured
        try:
            logging_config._configured = False

            with (
                patch.dict(os.environ, {"MODEL_EVALUATOR_LOG_FORMAT": "json"}),
                patch("structlog.configure") as mock_configure,
                patch("logging.basicConfig"),
            ):
                logging_config.configure_logging()
                mock_configure.assert_called_once()
        finally:
            logging_config._configured = orig

    def test_configure_logging_console_format(self) -> None:
        """Should configure structlog with console renderer."""
        import logging_config

        orig = logging_config._configured
        try:
            logging_config._configured = False

            with (
                patch.dict(os.environ, {"MODEL_EVALUATOR_LOG_FORMAT": "console"}),
                patch("structlog.configure") as mock_configure,
                patch("logging.basicConfig"),
            ):
                logging_config.configure_logging()
                mock_configure.assert_called_once()
        finally:
            logging_config._configured = orig

    def test_configure_logging_idempotent(self) -> None:
        """Should only configure once (idempotent)."""
        import logging_config

        orig = logging_config._configured
        try:
            logging_config._configured = True

            with patch("structlog.configure") as mock_configure:
                logging_config.configure_logging()
                mock_configure.assert_not_called()
        finally:
            logging_config._configured = orig

    def test_configure_logging_with_settings(self) -> None:
        """Should read log_level from settings object."""
        import logging

        import logging_config

        mock_settings = MagicMock()
        mock_settings.log_format = "json"
        mock_settings.log_level = "DEBUG"

        orig = logging_config._configured
        try:
            logging_config._configured = False

            with patch("structlog.configure"), patch("logging.basicConfig") as mock_basic:
                logging_config.configure_logging(settings=mock_settings)
                mock_basic.assert_called_once()
                # Verify the level was used
                call_kwargs = mock_basic.call_args
                assert call_kwargs[1]["level"] == logging.DEBUG
        finally:
            logging_config._configured = orig


class TestGetLogger:
    """Tests for get_logger."""

    def test_get_logger_returns_logger(self) -> None:
        """Should return a structlog BoundLogger."""
        from logging_config import get_logger

        with patch("logging_config.configure_logging"):
            logger = get_logger("test_module")
            assert logger is not None

    def test_get_logger_default_name(self) -> None:
        """Should use 'model_evaluator' when no name given."""
        from logging_config import get_logger

        with patch("logging_config.configure_logging"), patch("structlog.get_logger") as mock_get:
            get_logger()
            mock_get.assert_called_once_with("model_evaluator")
