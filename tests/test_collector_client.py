"""Tests for the CollectorClient.

Covers:
- __init__: sets base_url, timeout, accepts shared http_client
- get_validated_samples: fetches samples with correct URL/params, handles errors
- health_check: returns True/False based on service availability
- close: closes the underlying httpx.AsyncClient when owned

Uses httpx.MockTransport for realistic HTTP testing without real network calls.
"""

import json

import httpx
import pytest

from services.collector_client import CollectorClient


def _make_transport(
    *,
    status_code: int = 200,
    json_data: list | dict | None = None,
    raise_error: Exception | None = None,
) -> httpx.MockTransport:
    """Build a MockTransport that returns a fixed response."""

    def handler(request: httpx.Request) -> httpx.Response:
        if raise_error:
            raise raise_error
        body = json.dumps(json_data if json_data is not None else []).encode()
        return httpx.Response(
            status_code, content=body, headers={"content-type": "application/json"}
        )

    return httpx.MockTransport(handler)


class TestCollectorClientInit:
    """Tests for CollectorClient initialization."""

    def test_strips_trailing_slash(self) -> None:
        """base_url should have trailing slash removed."""
        client = CollectorClient(base_url="http://localhost:8151/", timeout=10.0)
        assert client._base_url == "http://localhost:8151"

    def test_default_timeout(self) -> None:
        """Should use 30s default timeout."""
        client = CollectorClient(base_url="http://localhost:8151")
        assert client._timeout == 30.0

    def test_accepts_shared_http_client(self) -> None:
        """Should store the provided httpx.AsyncClient and not own it."""
        transport = _make_transport()
        shared_client = httpx.AsyncClient(transport=transport)
        client = CollectorClient(base_url="http://collector:8151", http_client=shared_client)
        assert client._client is shared_client
        assert client._owns_client is False

    def test_creates_default_client_when_none_provided(self) -> None:
        """Should create its own httpx.AsyncClient when none is passed."""
        client = CollectorClient(base_url="http://collector:8151", timeout=15.0)
        assert isinstance(client._client, httpx.AsyncClient)
        assert client._owns_client is True


class TestGetValidatedSamples:
    """Tests for CollectorClient.get_validated_samples."""

    async def test_fetches_samples_successfully(self) -> None:
        """Should call GET with correct URL and params, returning parsed JSON."""
        samples_data = [
            {
                "id": "1",
                "messages": [{"role": "user", "content": "hello"}],
                "response_content": "{}",
            },
            {
                "id": "2",
                "messages": [{"role": "user", "content": "world"}],
                "response_content": "{}",
            },
        ]
        captured_requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_requests.append(request)
            return httpx.Response(
                200,
                content=json.dumps(samples_data).encode(),
                headers={"content-type": "application/json"},
            )

        transport = httpx.MockTransport(handler)
        http_client = httpx.AsyncClient(transport=transport)
        client = CollectorClient(base_url="http://collector:8151", http_client=http_client)

        samples = await client.get_validated_samples("test_schema", limit=50)

        assert len(samples) == 2
        assert samples[0]["id"] == "1"
        assert samples[1]["id"] == "2"

        # Verify correct URL and params
        assert len(captured_requests) == 1
        req = captured_requests[0]
        assert str(req.url).startswith("http://collector:8151/api/samples")
        assert req.url.params["schema_id"] == "test_schema"
        assert req.url.params["limit"] == "50"
        assert req.url.params["validation_passed"] == "true"

    async def test_raises_on_http_error(self) -> None:
        """Should propagate HTTP errors from the server."""
        transport = _make_transport(status_code=500)
        http_client = httpx.AsyncClient(transport=transport)
        client = CollectorClient(base_url="http://collector:8151", http_client=http_client)

        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await client.get_validated_samples("test_schema")
        assert exc_info.value.response.status_code == 500

    async def test_default_limit(self) -> None:
        """Should use default limit of 100 when not specified."""
        captured_requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_requests.append(request)
            return httpx.Response(200, content=b"[]", headers={"content-type": "application/json"})

        transport = httpx.MockTransport(handler)
        http_client = httpx.AsyncClient(transport=transport)
        client = CollectorClient(base_url="http://collector:8151", http_client=http_client)

        await client.get_validated_samples("test_schema")

        assert captured_requests[0].url.params["limit"] == "100"

    async def test_shared_client_reused_across_calls(self) -> None:
        """The same httpx.AsyncClient should be used for multiple requests."""
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(200, content=b"[]", headers={"content-type": "application/json"})

        transport = httpx.MockTransport(handler)
        http_client = httpx.AsyncClient(transport=transport)
        client = CollectorClient(base_url="http://collector:8151", http_client=http_client)

        await client.get_validated_samples("schema_a")
        await client.get_validated_samples("schema_b")

        assert call_count == 2

    async def test_returns_empty_list_when_no_samples(self) -> None:
        """Should return an empty list when the server returns []."""
        transport = _make_transport(json_data=[])
        http_client = httpx.AsyncClient(transport=transport)
        client = CollectorClient(base_url="http://collector:8151", http_client=http_client)

        samples = await client.get_validated_samples("empty_schema")
        assert samples == []

    async def test_passes_job_id_when_provided(self) -> None:
        """Should include job_id in query params when provided."""
        captured_requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_requests.append(request)
            return httpx.Response(200, content=b"[]", headers={"content-type": "application/json"})

        transport = httpx.MockTransport(handler)
        http_client = httpx.AsyncClient(transport=transport)
        client = CollectorClient(base_url="http://collector:8151", http_client=http_client)

        await client.get_validated_samples("test_schema", job_id="job-123")

        assert captured_requests[0].url.params["job_id"] == "job-123"

    async def test_passes_inference_id_when_provided(self) -> None:
        """Should include inference_id in query params when provided."""
        captured_requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_requests.append(request)
            return httpx.Response(200, content=b"[]", headers={"content-type": "application/json"})

        transport = httpx.MockTransport(handler)
        http_client = httpx.AsyncClient(transport=transport)
        client = CollectorClient(base_url="http://collector:8151", http_client=http_client)

        await client.get_validated_samples("test_schema", inference_id="inf-456")

        assert captured_requests[0].url.params["inference_id"] == "inf-456"

    async def test_passes_both_job_and_inference_id(self) -> None:
        """Should include both job_id and inference_id when both provided."""
        captured_requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_requests.append(request)
            return httpx.Response(200, content=b"[]", headers={"content-type": "application/json"})

        transport = httpx.MockTransport(handler)
        http_client = httpx.AsyncClient(transport=transport)
        client = CollectorClient(base_url="http://collector:8151", http_client=http_client)

        await client.get_validated_samples("test_schema", job_id="job-789", inference_id="inf-012")

        params = captured_requests[0].url.params
        assert params["job_id"] == "job-789"
        assert params["inference_id"] == "inf-012"
        assert params["schema_id"] == "test_schema"
        assert params["validation_passed"] == "true"

    async def test_omits_job_and_inference_id_when_none(self) -> None:
        """Should NOT include job_id or inference_id when they are None (default)."""
        captured_requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_requests.append(request)
            return httpx.Response(200, content=b"[]", headers={"content-type": "application/json"})

        transport = httpx.MockTransport(handler)
        http_client = httpx.AsyncClient(transport=transport)
        client = CollectorClient(base_url="http://collector:8151", http_client=http_client)

        await client.get_validated_samples("test_schema")

        params = captured_requests[0].url.params
        assert "job_id" not in params
        assert "inference_id" not in params


class TestCollectorHealthCheck:
    """Tests for CollectorClient.health_check."""

    async def test_health_check_returns_true(self) -> None:
        """Should return True on 200 response."""
        transport = _make_transport(status_code=200)
        http_client = httpx.AsyncClient(transport=transport)
        client = CollectorClient(base_url="http://collector:8151", http_client=http_client)

        result = await client.health_check()
        assert result is True

    async def test_health_check_returns_false_on_error(self) -> None:
        """Should return False when service is unreachable."""
        transport = _make_transport(raise_error=httpx.ConnectError("Connection refused"))
        http_client = httpx.AsyncClient(transport=transport)
        client = CollectorClient(base_url="http://collector:8151", http_client=http_client)

        result = await client.health_check()
        assert result is False

    async def test_health_check_returns_false_on_non_200(self) -> None:
        """Should return False on non-200 status."""
        transport = _make_transport(status_code=503)
        http_client = httpx.AsyncClient(transport=transport)
        client = CollectorClient(base_url="http://collector:8151", http_client=http_client)

        result = await client.health_check()
        assert result is False

    async def test_health_check_hits_correct_url(self) -> None:
        """Should call GET /health on the base_url."""
        captured_requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_requests.append(request)
            return httpx.Response(200)

        transport = httpx.MockTransport(handler)
        http_client = httpx.AsyncClient(transport=transport)
        client = CollectorClient(base_url="http://collector:8151", http_client=http_client)

        await client.health_check()

        assert len(captured_requests) == 1
        assert str(captured_requests[0].url) == "http://collector:8151/health"


class TestCollectorClose:
    """Tests for CollectorClient.close."""

    async def test_close_closes_owned_client(self) -> None:
        """close() should close the client when it was auto-created."""
        client = CollectorClient(base_url="http://collector:8151")
        assert client._owns_client is True

        # Should not raise
        await client.close()

    async def test_close_skips_shared_client(self) -> None:
        """close() should not close an injected shared client."""
        transport = _make_transport()
        shared_client = httpx.AsyncClient(transport=transport)
        client = CollectorClient(base_url="http://collector:8151", http_client=shared_client)

        assert client._owns_client is False
        # Should not raise, and shared_client should still be usable
        await client.close()

        # Verify shared client is still open by making a request
        response = await shared_client.get("http://collector:8151/test")
        assert response.status_code == 200
        await shared_client.aclose()
