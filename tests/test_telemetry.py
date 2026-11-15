"""Tests for the telemetry module.

Covers:
- _test_connection: socket-level connectivity checks
- setup_telemetry: OTEL_SDK_DISABLED, idempotent initialization,
  import errors, connection success/failure, settings integration,
  HTTPX instrumentation failure
"""

import os
import socket
from unittest.mock import MagicMock, patch

import telemetry


class TestTestConnection:
    """Tests for _test_connection helper."""

    def test_connection_success(self) -> None:
        """Should return True when socket connects successfully."""
        mock_sock = MagicMock()
        with patch.object(socket, "create_connection", return_value=mock_sock) as mock_create:
            result = telemetry._test_connection("localhost", 4318)
            assert result is True
            mock_create.assert_called_once_with(("localhost", 4318), timeout=2.0)
            mock_sock.close.assert_called_once()

    def test_connection_timeout(self) -> None:
        """Should return False on TimeoutError."""
        with patch.object(socket, "create_connection", side_effect=TimeoutError):
            result = telemetry._test_connection("localhost", 4318)
            assert result is False

    def test_connection_refused(self) -> None:
        """Should return False on ConnectionRefusedError."""
        with patch.object(socket, "create_connection", side_effect=ConnectionRefusedError):
            result = telemetry._test_connection("localhost", 4318)
            assert result is False

    def test_connection_os_error(self) -> None:
        """Should return False on OSError."""
        with patch.object(socket, "create_connection", side_effect=OSError):
            result = telemetry._test_connection("localhost", 4318)
            assert result is False

    def test_connection_custom_timeout(self) -> None:
        """Should pass custom timeout to create_connection."""
        mock_sock = MagicMock()
        with patch.object(socket, "create_connection", return_value=mock_sock) as mock_create:
            telemetry._test_connection("myhost", 9090, timeout=5.0)
            mock_create.assert_called_once_with(("myhost", 9090), timeout=5.0)


class TestSetupTelemetry:
    """Tests for setup_telemetry."""

    def setup_method(self) -> None:
        """Reset module state before each test."""
        telemetry._initialized = False

    def teardown_method(self) -> None:
        """Reset module state after each test."""
        telemetry._initialized = False

    def test_otel_disabled(self) -> None:
        """Should return False when OTEL_SDK_DISABLED is 'true'."""
        with patch.dict(os.environ, {"OTEL_SDK_DISABLED": "true"}):
            result = telemetry.setup_telemetry()
            assert result is False
            assert telemetry._initialized is False

    def test_otel_disabled_case_insensitive(self) -> None:
        """Should handle OTEL_SDK_DISABLED='True' (mixed case)."""
        with patch.dict(os.environ, {"OTEL_SDK_DISABLED": "True"}):
            result = telemetry.setup_telemetry()
            assert result is False

    def test_already_initialized(self) -> None:
        """Should return True immediately when already initialized."""
        telemetry._initialized = True
        result = telemetry.setup_telemetry()
        assert result is True

    def test_import_error(self) -> None:
        """Should return False when opentelemetry packages are unavailable."""
        original_import = (
            __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__
        )  # noqa: E501

        def mock_import(name, *args, **kwargs):
            if name.startswith("opentelemetry"):
                raise ImportError(f"No module named '{name}'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = telemetry.setup_telemetry()
            assert result is False
            assert telemetry._initialized is False

    def test_connection_success(self) -> None:
        """Should return True and set _initialized when connection succeeds."""
        mock_trace = MagicMock()
        mock_exporter_cls = MagicMock()
        mock_instrumentor_cls = MagicMock()
        mock_resource_cls = MagicMock()
        mock_provider_cls = MagicMock()
        mock_processor_cls = MagicMock()
        mock_service_name = MagicMock()

        modules = {
            "opentelemetry": MagicMock(trace=mock_trace),
            "opentelemetry.trace": mock_trace,
            "opentelemetry.exporter.otlp.proto.http.trace_exporter": MagicMock(
                OTLPSpanExporter=mock_exporter_cls
            ),
            "opentelemetry.instrumentation.httpx": MagicMock(
                HTTPXClientInstrumentor=mock_instrumentor_cls
            ),
            "opentelemetry.sdk.resources": MagicMock(
                SERVICE_NAME=mock_service_name,
                Resource=mock_resource_cls,
            ),
            "opentelemetry.sdk.trace": MagicMock(TracerProvider=mock_provider_cls),
            "opentelemetry.sdk.trace.export": MagicMock(BatchSpanProcessor=mock_processor_cls),
        }

        with (
            patch.dict("sys.modules", modules),
            patch.object(socket, "create_connection", return_value=MagicMock()),
            patch.dict(
                os.environ,
                {"MODEL_EVALUATOR_OTLP_TRACES_ENDPOINT": "http://alloy:4318/v1/traces"},
                clear=False,
            ),
        ):
            # Remove OTEL_SDK_DISABLED if set
            os.environ.pop("OTEL_SDK_DISABLED", None)
            result = telemetry.setup_telemetry()
            assert result is True
            assert telemetry._initialized is True
            mock_trace.set_tracer_provider.assert_called_once()

    def test_connection_failure_still_returns_true(self) -> None:
        """Should still return True when endpoint is unreachable (tracing disabled)."""
        mock_trace = MagicMock()
        mock_instrumentor_cls = MagicMock()
        mock_resource_cls = MagicMock()
        mock_provider_cls = MagicMock()
        mock_service_name = MagicMock()

        modules = {
            "opentelemetry": MagicMock(trace=mock_trace),
            "opentelemetry.trace": mock_trace,
            "opentelemetry.exporter.otlp.proto.http.trace_exporter": MagicMock(
                OTLPSpanExporter=MagicMock()
            ),
            "opentelemetry.instrumentation.httpx": MagicMock(
                HTTPXClientInstrumentor=mock_instrumentor_cls
            ),
            "opentelemetry.sdk.resources": MagicMock(
                SERVICE_NAME=mock_service_name,
                Resource=mock_resource_cls,
            ),
            "opentelemetry.sdk.trace": MagicMock(TracerProvider=mock_provider_cls),
            "opentelemetry.sdk.trace.export": MagicMock(BatchSpanProcessor=MagicMock()),
        }

        with (
            patch.dict("sys.modules", modules),
            patch.object(socket, "create_connection", side_effect=ConnectionRefusedError),
            patch.dict(
                os.environ,
                {"MODEL_EVALUATOR_OTLP_TRACES_ENDPOINT": "http://alloy:4318/v1/traces"},
                clear=False,
            ),
        ):
            os.environ.pop("OTEL_SDK_DISABLED", None)
            result = telemetry.setup_telemetry()
            assert result is True
            assert telemetry._initialized is True
            # Tracer provider is still set even without exporter
            mock_trace.set_tracer_provider.assert_called_once()

    def test_unset_endpoint_returns_false(self) -> None:
        """Should return False when MODEL_EVALUATOR_OTLP_TRACES_ENDPOINT is unset."""
        import telemetry

        telemetry._initialized = False
        with patch.dict(os.environ, {"MODEL_EVALUATOR_OTLP_TRACES_ENDPOINT": ""}, clear=False):
            os.environ.pop("OTEL_SDK_DISABLED", None)
            result = telemetry.setup_telemetry()
        assert result is False
        assert telemetry._initialized is False

    def test_with_settings(self) -> None:
        """Should use service_name and otlp_traces_endpoint from settings."""
        mock_trace = MagicMock()
        mock_exporter_cls = MagicMock()
        mock_instrumentor_cls = MagicMock()
        mock_resource_cls = MagicMock()
        mock_provider_cls = MagicMock()
        mock_processor_cls = MagicMock()
        mock_service_name_const = MagicMock()

        modules = {
            "opentelemetry": MagicMock(trace=mock_trace),
            "opentelemetry.trace": mock_trace,
            "opentelemetry.exporter.otlp.proto.http.trace_exporter": MagicMock(
                OTLPSpanExporter=mock_exporter_cls
            ),
            "opentelemetry.instrumentation.httpx": MagicMock(
                HTTPXClientInstrumentor=mock_instrumentor_cls
            ),
            "opentelemetry.sdk.resources": MagicMock(
                SERVICE_NAME=mock_service_name_const,
                Resource=mock_resource_cls,
            ),
            "opentelemetry.sdk.trace": MagicMock(TracerProvider=mock_provider_cls),
            "opentelemetry.sdk.trace.export": MagicMock(BatchSpanProcessor=mock_processor_cls),
        }

        mock_settings = MagicMock()
        mock_settings.service_name = "my-custom-service"
        mock_settings.otlp_traces_endpoint = "http://localhost:4318/v1/traces"

        with (
            patch.dict("sys.modules", modules),
            patch.object(socket, "create_connection", return_value=MagicMock()),
            patch.dict(os.environ, {}, clear=False),
        ):
            os.environ.pop("OTEL_SDK_DISABLED", None)
            result = telemetry.setup_telemetry(settings=mock_settings)
            assert result is True

            # Verify service_name from settings was used in Resource.create
            mock_resource_cls.create.assert_called_once_with(
                {mock_service_name_const: "my-custom-service"}
            )

            # Verify endpoint from settings was used in OTLPSpanExporter
            mock_exporter_cls.assert_called_once_with(
                endpoint="http://localhost:4318/v1/traces", timeout=5
            )

    def test_httpx_instrument_failure(self) -> None:
        """Should still return True when HTTPXClientInstrumentor.instrument raises."""
        mock_trace = MagicMock()
        mock_instrumentor = MagicMock()
        mock_instrumentor_cls = MagicMock(return_value=mock_instrumentor)
        mock_instrumentor.instrument.side_effect = RuntimeError("instrument failed")

        mock_resource_cls = MagicMock()
        mock_provider_cls = MagicMock()
        mock_service_name = MagicMock()

        modules = {
            "opentelemetry": MagicMock(trace=mock_trace),
            "opentelemetry.trace": mock_trace,
            "opentelemetry.exporter.otlp.proto.http.trace_exporter": MagicMock(
                OTLPSpanExporter=MagicMock()
            ),
            "opentelemetry.instrumentation.httpx": MagicMock(
                HTTPXClientInstrumentor=mock_instrumentor_cls
            ),
            "opentelemetry.sdk.resources": MagicMock(
                SERVICE_NAME=mock_service_name,
                Resource=mock_resource_cls,
            ),
            "opentelemetry.sdk.trace": MagicMock(TracerProvider=mock_provider_cls),
            "opentelemetry.sdk.trace.export": MagicMock(BatchSpanProcessor=MagicMock()),
        }

        with (
            patch.dict("sys.modules", modules),
            patch.object(socket, "create_connection", side_effect=ConnectionRefusedError),
            patch.dict(
                os.environ,
                {"MODEL_EVALUATOR_OTLP_TRACES_ENDPOINT": "http://alloy:4318/v1/traces"},
                clear=False,
            ),
        ):
            os.environ.pop("OTEL_SDK_DISABLED", None)
            result = telemetry.setup_telemetry()
            assert result is True
            assert telemetry._initialized is True

    def test_exporter_config_exception(self) -> None:
        """Should still return True when endpoint config raises a generic exception."""
        mock_trace = MagicMock()
        mock_instrumentor_cls = MagicMock()
        mock_resource_cls = MagicMock()
        mock_provider_cls = MagicMock()
        mock_service_name = MagicMock()

        modules = {
            "opentelemetry": MagicMock(trace=mock_trace),
            "opentelemetry.trace": mock_trace,
            "opentelemetry.exporter.otlp.proto.http.trace_exporter": MagicMock(
                OTLPSpanExporter=MagicMock()
            ),
            "opentelemetry.instrumentation.httpx": MagicMock(
                HTTPXClientInstrumentor=mock_instrumentor_cls
            ),
            "opentelemetry.sdk.resources": MagicMock(
                SERVICE_NAME=mock_service_name,
                Resource=mock_resource_cls,
            ),
            "opentelemetry.sdk.trace": MagicMock(TracerProvider=mock_provider_cls),
            "opentelemetry.sdk.trace.export": MagicMock(BatchSpanProcessor=MagicMock()),
        }

        mock_settings = MagicMock()
        mock_settings.service_name = "test-svc"
        # Make the endpoint attribute raise when .replace is called
        mock_settings.otlp_traces_endpoint = MagicMock()
        mock_settings.otlp_traces_endpoint.replace.side_effect = RuntimeError("bad endpoint")

        with (
            patch.dict("sys.modules", modules),
            patch.dict(os.environ, {}, clear=False),
        ):
            os.environ.pop("OTEL_SDK_DISABLED", None)
            result = telemetry.setup_telemetry(settings=mock_settings)
            assert result is True
            assert telemetry._initialized is True
