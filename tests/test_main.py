"""Tests for the main application module.

Covers:
- lifespan: startup and shutdown behaviour
- health endpoint
- liveness endpoint
- readiness endpoint
- root endpoint
- CORS middleware
- global exception handler
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from main import app


class TestLifespan:
    """Tests for the lifespan context manager."""

    async def test_lifespan_startup_and_shutdown(self) -> None:
        """lifespan should call init_db on startup and dispose_engine on shutdown."""
        with (
            patch("main.get_settings") as mock_settings,
            patch("main.configure_logging") as mock_logging,
            patch("main.setup_telemetry") as mock_telemetry,
            patch("main.init_db", new_callable=AsyncMock) as mock_init_db,
            patch("main.dispose_engine", new_callable=AsyncMock) as mock_dispose,
        ):
            settings = MagicMock()
            settings.service_name = "model-evaluator"
            settings.port = 8154
            settings.database_url = "sqlite:///:memory:"
            mock_settings.return_value = settings

            # Simulate the lifespan (startup + shutdown)
            from main import lifespan

            async with lifespan(app):
                # After startup, init_db should have been called
                mock_init_db.assert_called_once_with(settings)
                mock_logging.assert_called_once_with(settings)
                mock_telemetry.assert_called_once_with(settings)

            # After shutdown, dispose_engine should have been called
            mock_dispose.assert_awaited_once()

    async def test_lifespan_handles_otel_instrumentation_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """lifespan must continue even if FastAPI OTEL instrumentation fails.

        Instead of the fragile ``patch('builtins.__import__')`` trick, monkeypatch
        ``FastAPIInstrumentor.instrument_app`` directly to raise.
        """
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        def boom(app_arg: object) -> None:
            raise RuntimeError("simulated OTEL instrumentation failure")

        monkeypatch.setattr(FastAPIInstrumentor, "instrument_app", boom)

        with (
            patch("main.get_settings") as mock_settings,
            patch("main.configure_logging"),
            patch("main.setup_telemetry"),
            patch("main.init_db", new_callable=AsyncMock),
            patch("main.dispose_engine", new_callable=AsyncMock),
        ):
            settings = MagicMock()
            settings.service_name = "model-evaluator"
            settings.port = 8154
            settings.database_url = "sqlite:///:memory:"
            mock_settings.return_value = settings

            from main import lifespan

            # Should NOT raise -- instrumentation failure is caught and logged
            async with lifespan(app):
                pass


class TestHealthEndpoint:
    """Tests for GET /health."""

    async def test_health_returns_ok(self) -> None:
        """Health endpoint should return service info."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
            response = await ac.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "model-evaluator"
        assert data["version"] == "0.1.0"


class TestReadinessEndpoint:
    """Tests for GET /health/ready.

    The readiness probe checks:
    - Database connection (primary gate)
    - Inference endpoint health (non-blocking, 2s timeout)
    - Data-collector health (non-blocking, 2s timeout)

    Returns 200 even when upstreams are down (to avoid failing deploys),
    but reports degraded status.
    """

    async def test_readiness_all_healthy(self) -> None:
        """Should return ready with all services ok when everything is up."""
        with (
            patch("main.get_engine_for_check", new_callable=AsyncMock, return_value=True),
            patch("main._check_upstream", new_callable=AsyncMock, return_value=True),
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
                response = await ac.get("/health/ready")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ready"
            assert data["database"] == "connected"
            assert data["inference"] == "ok"
            assert data["collector"] == "ok"

    async def test_readiness_when_db_connected_upstreams_healthy(self) -> None:
        """Should return ready when database and all upstreams are connected."""

        async def _mock_upstream(url: str, timeout: float = 2.0) -> bool:
            return True

        with (
            patch("main.get_engine_for_check", new_callable=AsyncMock, return_value=True),
            patch("main._check_upstream", side_effect=_mock_upstream),
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
                response = await ac.get("/health/ready")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ready"
            assert data["database"] == "connected"
            assert data["inference"] == "ok"
            assert data["collector"] == "ok"

    async def test_readiness_when_db_disconnected(self) -> None:
        """Should return not_ready when database is disconnected, regardless of upstreams."""
        with (
            patch("main.get_engine_for_check", new_callable=AsyncMock, return_value=False),
            patch("main._check_upstream", new_callable=AsyncMock, return_value=True),
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
                response = await ac.get("/health/ready")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "not_ready"
            assert data["database"] == "disconnected"

    async def test_readiness_when_inference_down(self) -> None:
        """Should return 200 with degraded status when inference endpoint is down."""

        async def _mock_upstream(url: str, timeout: float = 2.0) -> bool:
            # Inference endpoint is down, collector is up
            return "/health" in url and "data-collector" in url

        with (
            patch("main.get_engine_for_check", new_callable=AsyncMock, return_value=True),
            patch("main._check_upstream", side_effect=_mock_upstream),
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
                response = await ac.get("/health/ready")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "degraded"
            assert data["database"] == "connected"
            assert data["inference"] == "error"
            assert data["collector"] == "ok"

    async def test_readiness_when_collector_down(self) -> None:
        """Should return 200 with degraded status when data-collector is down."""

        async def _mock_upstream(url: str, timeout: float = 2.0) -> bool:
            # Collector is down, inference endpoint is up
            return "/health" in url and "inference-router" in url

        with (
            patch("main.get_engine_for_check", new_callable=AsyncMock, return_value=True),
            patch("main._check_upstream", side_effect=_mock_upstream),
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
                response = await ac.get("/health/ready")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "degraded"
            assert data["database"] == "connected"
            assert data["inference"] == "ok"
            assert data["collector"] == "error"

    async def test_readiness_when_both_upstreams_down(self) -> None:
        """Should return 200 with degraded status when both upstreams are down."""
        with (
            patch("main.get_engine_for_check", new_callable=AsyncMock, return_value=True),
            patch("main._check_upstream", new_callable=AsyncMock, return_value=False),
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
                response = await ac.get("/health/ready")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "degraded"
            assert data["database"] == "connected"
            assert data["inference"] == "error"
            assert data["collector"] == "error"

    async def test_readiness_when_everything_down(self) -> None:
        """Should return 200 with not_ready when DB and all upstreams are down."""
        with (
            patch("main.get_engine_for_check", new_callable=AsyncMock, return_value=False),
            patch("main._check_upstream", new_callable=AsyncMock, return_value=False),
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
                response = await ac.get("/health/ready")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "not_ready"
            assert data["database"] == "disconnected"
            assert data["inference"] == "error"
            assert data["collector"] == "error"

    async def test_readiness_response_includes_all_fields(self) -> None:
        """Readiness response should always include database, inference, and collector fields."""
        with (
            patch("main.get_engine_for_check", new_callable=AsyncMock, return_value=True),
            patch("main._check_upstream", new_callable=AsyncMock, return_value=True),
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
                response = await ac.get("/health/ready")

            data = response.json()
            assert "status" in data
            assert "database" in data
            assert "inference" in data
            assert "collector" in data


class TestLivenessEndpoint:
    """Tests for GET /health/live."""

    async def test_liveness_endpoint(self) -> None:
        """Liveness endpoint should return 200 with status=alive."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
            response = await ac.get("/health/live")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"


class TestRootEndpoint:
    """Tests for GET /."""

    async def test_root_endpoint(self) -> None:
        """Root endpoint should return 200 with service info."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
            response = await ac.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "model-evaluator"
        assert data["version"] == "0.1.0"
        assert data["docs"] == "/docs"
        assert data["health"] == "/health"


class TestCorsMiddleware:
    """Tests for CORS middleware."""

    async def test_cors_headers(self) -> None:
        """OPTIONS request should return CORS headers."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
            response = await ac.options(
                "/health",
                headers={
                    "Origin": "http://localhost:3000",
                    "Access-Control-Request-Method": "GET",
                },
            )

        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers
        assert response.headers["access-control-allow-origin"] == "*"


class TestGlobalExceptionHandler:
    """Tests for the global exception handler."""

    async def test_unhandled_exception_returns_500(self) -> None:
        """Unhandled exceptions should return a structured 500 JSON response."""
        # Call the handler function directly with a non-HTTP exception
        from main import global_exception_handler

        request = MagicMock()
        exc = RuntimeError("unexpected boom")

        response = await global_exception_handler(request, exc)

        assert response.status_code == 500
        import json

        body = response.body
        assert isinstance(body, bytes)
        data = json.loads(body.decode("utf-8"))
        assert data["error"]["message"] == "Internal server error"
        assert data["error"]["type"] == "server_error"
        assert data["error"]["code"] == 500

    async def test_http_exception_is_reraised(self) -> None:
        """HTTPException should be re-raised, not caught by global handler."""
        import pytest
        from fastapi import HTTPException

        from main import global_exception_handler

        request = MagicMock()
        exc = HTTPException(status_code=404, detail="Not found")

        with pytest.raises(HTTPException) as exc_info:
            await global_exception_handler(request, exc)
        assert exc_info.value.status_code == 404
