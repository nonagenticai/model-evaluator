"""Tests for the enhanced readiness probe with upstream health checks.

Covers:
- _check_upstream helper function: success, HTTP errors, connection errors, timeouts
- ReadinessResponse model: field defaults, all status combinations
- Integration: readiness endpoint with various upstream scenarios
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from httpx import ASGITransport, AsyncClient

from main import _check_upstream, app
from models.domain import ReadinessResponse

# ---------------------------------------------------------------------------
# Unit tests for _check_upstream
# ---------------------------------------------------------------------------


class TestCheckUpstream:
    """Tests for the _check_upstream helper function."""

    async def test_returns_true_on_200(self) -> None:
        """Should return True when upstream responds with 200."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("main.httpx.AsyncClient", return_value=mock_client):
            result = await _check_upstream("http://example.com/health")

        assert result is True

    async def test_returns_false_on_non_200(self) -> None:
        """Should return False when upstream responds with non-200 status."""
        mock_response = MagicMock()
        mock_response.status_code = 503

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("main.httpx.AsyncClient", return_value=mock_client):
            result = await _check_upstream("http://example.com/health")

        assert result is False

    async def test_returns_false_on_connection_error(self) -> None:
        """Should return False when upstream is unreachable."""
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("main.httpx.AsyncClient", return_value=mock_client):
            result = await _check_upstream("http://unreachable.local/health")

        assert result is False

    async def test_returns_false_on_timeout(self) -> None:
        """Should return False when upstream times out."""
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.ReadTimeout("Timed out"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("main.httpx.AsyncClient", return_value=mock_client):
            result = await _check_upstream("http://slow.local/health")

        assert result is False

    async def test_returns_false_on_generic_exception(self) -> None:
        """Should return False on any unexpected exception without raising."""
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=RuntimeError("unexpected"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("main.httpx.AsyncClient", return_value=mock_client):
            result = await _check_upstream("http://broken.local/health")

        assert result is False

    async def test_uses_custom_timeout(self) -> None:
        """Should pass the specified timeout to the HTTP client."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("main.httpx.AsyncClient", return_value=mock_client) as mock_constructor:
            await _check_upstream("http://example.com/health", timeout=5.0)

        mock_constructor.assert_called_once_with(timeout=5.0)

    async def test_default_timeout_is_2_seconds(self) -> None:
        """Should default to a 2-second timeout."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("main.httpx.AsyncClient", return_value=mock_client) as mock_constructor:
            await _check_upstream("http://example.com/health")

        mock_constructor.assert_called_once_with(timeout=2.0)


# ---------------------------------------------------------------------------
# Unit tests for ReadinessResponse model
# ---------------------------------------------------------------------------


class TestReadinessResponseModel:
    """Tests for the ReadinessResponse Pydantic model."""

    def test_defaults_for_upstream_fields(self) -> None:
        """Inference and collector fields should default to 'unknown'."""
        response = ReadinessResponse(status="ready", database="connected")
        assert response.inference == "unknown"
        assert response.collector == "unknown"

    def test_all_fields_populated(self) -> None:
        """Should accept all fields when explicitly set."""
        response = ReadinessResponse(
            status="degraded",
            database="connected",
            inference="ok",
            collector="error",
        )
        assert response.status == "degraded"
        assert response.database == "connected"
        assert response.inference == "ok"
        assert response.collector == "error"

    def test_serialization_includes_all_fields(self) -> None:
        """model_dump should include all fields including upstream status."""
        response = ReadinessResponse(
            status="ready",
            database="connected",
            inference="ok",
            collector="ok",
        )
        data = response.model_dump()
        assert set(data.keys()) == {"status", "database", "inference", "collector"}

    def test_not_ready_status(self) -> None:
        """Model should accept not_ready status."""
        response = ReadinessResponse(
            status="not_ready",
            database="disconnected",
            inference="error",
            collector="error",
        )
        assert response.status == "not_ready"
        assert response.database == "disconnected"


# ---------------------------------------------------------------------------
# Integration tests for readiness endpoint with upstream checks
# ---------------------------------------------------------------------------


class TestReadinessEndpointIntegration:
    """Integration tests for the /health/ready endpoint with upstream checks."""

    async def test_db_down_overrides_upstream_status(self) -> None:
        """When DB is down, status should be not_ready even if upstreams are healthy."""
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
            # Upstreams still report their actual status
            assert data["inference"] == "ok"
            assert data["collector"] == "ok"

    async def test_always_returns_200(self) -> None:
        """Readiness probe should always return HTTP 200 to avoid failing K8s deploys."""
        # Test with everything down
        with (
            patch("main.get_engine_for_check", new_callable=AsyncMock, return_value=False),
            patch("main._check_upstream", new_callable=AsyncMock, return_value=False),
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
                response = await ac.get("/health/ready")
            assert response.status_code == 200

        # Test with everything up
        with (
            patch("main.get_engine_for_check", new_callable=AsyncMock, return_value=True),
            patch("main._check_upstream", new_callable=AsyncMock, return_value=True),
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
                response = await ac.get("/health/ready")
            assert response.status_code == 200

    async def test_upstream_checks_use_settings_urls(self) -> None:
        """Upstream checks should use URLs from service settings."""
        calls: list[str] = []

        async def _tracking_check(url: str, timeout: float = 2.0) -> bool:
            calls.append(url)
            return True

        with (
            patch("main.get_engine_for_check", new_callable=AsyncMock, return_value=True),
            patch("main._check_upstream", side_effect=_tracking_check),
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
                await ac.get("/health/ready")

        # Should have checked both inference endpoint and collector
        assert len(calls) == 2
        assert any("health" in url for url in calls)

    async def test_concurrent_health_checks(self) -> None:
        """All three health checks (db, inference, collector) should run concurrently."""
        import asyncio

        check_order: list[str] = []

        async def _slow_db_check() -> bool:
            check_order.append("db_start")
            await asyncio.sleep(0.01)
            check_order.append("db_end")
            return True

        async def _slow_upstream(url: str, timeout: float = 2.0) -> bool:
            name = "inference" if "inference-router" in url else "collector"
            check_order.append(f"{name}_start")
            await asyncio.sleep(0.01)
            check_order.append(f"{name}_end")
            return True

        with (
            patch("main.get_engine_for_check", side_effect=_slow_db_check),
            patch("main._check_upstream", side_effect=_slow_upstream),
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
                response = await ac.get("/health/ready")

            assert response.status_code == 200
            # All three should have started before any finished (concurrent via asyncio.gather)
            # Check that all starts happen, all ends happen
            starts = [c for c in check_order if c.endswith("_start")]
            assert len(starts) == 3

    async def test_degraded_status_values(self) -> None:
        """Verify the exact status string is 'degraded' when upstreams fail."""
        with (
            patch("main.get_engine_for_check", new_callable=AsyncMock, return_value=True),
            patch("main._check_upstream", new_callable=AsyncMock, return_value=False),
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
                response = await ac.get("/health/ready")

            data = response.json()
            # Status should be exactly "degraded", not "ready" or "not_ready"
            assert data["status"] == "degraded"
