"""Tests for the AdapterClient.

Covers:
- __init__: sets base_url and timeout
- load_adapter: POST request with correct payload and URL
- get_loaded_adapters: GET request to list adapters
- health_check: returns True/False based on service availability

Uses httpx.MockTransport for realistic HTTP testing without real network calls.
"""

import json

import httpx
import pytest

from services.adapter_client import AdapterClient


class TestAdapterClientInit:
    """Tests for AdapterClient initialization."""

    def test_strips_trailing_slash(self) -> None:
        """base_url should have trailing slash removed."""
        client = AdapterClient(base_url="http://localhost:8001/")
        assert client._base_url == "http://localhost:8001"

    def test_default_timeout(self) -> None:
        """Should use 60s default timeout."""
        client = AdapterClient(base_url="http://localhost:8001")
        assert client._timeout == 60.0

    def test_custom_timeout(self) -> None:
        """Should accept custom timeout."""
        client = AdapterClient(base_url="http://localhost:8001", timeout=120.0)
        assert client._timeout == 120.0


class TestLoadAdapter:
    """Tests for AdapterClient.load_adapter."""

    async def test_load_adapter_success(self) -> None:
        """Should POST to /api/adapters/{schema_id}/load with adapter_path payload."""
        captured_requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_requests.append(request)
            return httpx.Response(
                200,
                content=json.dumps({"status": "loaded"}).encode(),
                headers={"content-type": "application/json"},
            )

        transport = httpx.MockTransport(handler)
        client = AdapterClient(base_url="http://vllm:8001")

        from unittest.mock import patch

        with patch(
            "services.adapter_client.httpx.AsyncClient",
            return_value=httpx.AsyncClient(transport=transport),
        ):
            result = await client.load_adapter("ks_EntitySpotterAgent", "s3://bucket/adapter")

        assert result == {"status": "loaded"}
        assert len(captured_requests) == 1

        req = captured_requests[0]
        assert "/api/adapters/ks_EntitySpotterAgent/load" in str(req.url)
        payload = json.loads(req.content)
        assert payload["adapter_path"] == "s3://bucket/adapter"

    async def test_load_adapter_raises_on_error(self) -> None:
        """Should propagate HTTP errors after retries are exhausted."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, content=b'{"error": "internal"}')

        transport = httpx.MockTransport(handler)
        client = AdapterClient(base_url="http://vllm:8001")

        from unittest.mock import patch

        # Save real class before patching to avoid recursion
        real_async_client = httpx.AsyncClient

        with patch(
            "services.adapter_client.httpx.AsyncClient",
            side_effect=lambda **kwargs: real_async_client(transport=transport),
        ):
            with patch("services.adapter_client._LOAD_RETRY_DELAY", 0):
                with pytest.raises(httpx.HTTPStatusError) as exc_info:
                    await client.load_adapter("test", "s3://bucket/adapter")
                assert exc_info.value.response.status_code == 500

    async def test_load_adapter_4xx_no_retry_real_http(self) -> None:
        """Through the REAL retry loop, a 4xx response must raise immediately
        with zero retries (not 3). Verified by counting the transport calls."""
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(400, content=b'{"error": "bad adapter path"}')

        shared = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        client = AdapterClient(base_url="http://vllm:8001", http_client=shared)

        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await client.load_adapter("schema_bad", "s3://bucket/bad-adapter")
        assert exc_info.value.response.status_code == 400
        # 4xx must not retry -- exactly one attempt
        assert call_count == 1
        await shared.aclose()

    async def test_load_adapter_correct_url_construction(self) -> None:
        """Should construct the correct URL from base_url + schema_id."""
        captured_urls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_urls.append(str(request.url))
            return httpx.Response(
                200,
                content=json.dumps({"status": "loaded"}).encode(),
                headers={"content-type": "application/json"},
            )

        transport = httpx.MockTransport(handler)
        client = AdapterClient(base_url="http://vllm:8001/")  # trailing slash

        from unittest.mock import patch

        with patch(
            "services.adapter_client.httpx.AsyncClient",
            return_value=httpx.AsyncClient(transport=transport),
        ):
            await client.load_adapter("my_schema", "/path/to/adapter")

        assert captured_urls[0] == "http://vllm:8001/api/adapters/my_schema/load"


class TestGetLoadedAdapters:
    """Tests for AdapterClient.get_loaded_adapters."""

    async def test_get_loaded_adapters_success(self) -> None:
        """Should GET the list of loaded adapters from /api/adapters."""
        adapters_data = {"adapters": [{"name": "test", "path": "/adapters/test"}]}
        captured_requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_requests.append(request)
            return httpx.Response(
                200,
                content=json.dumps(adapters_data).encode(),
                headers={"content-type": "application/json"},
            )

        transport = httpx.MockTransport(handler)
        client = AdapterClient(base_url="http://vllm:8001")

        from unittest.mock import patch

        with patch(
            "services.adapter_client.httpx.AsyncClient",
            return_value=httpx.AsyncClient(transport=transport),
        ):
            result = await client.get_loaded_adapters()

        assert result == adapters_data
        assert "/api/adapters" in str(captured_requests[0].url)

    async def test_get_loaded_adapters_raises_on_error(self) -> None:
        """Should propagate HTTP errors."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, content=b'{"error": "internal"}')

        transport = httpx.MockTransport(handler)
        client = AdapterClient(base_url="http://vllm:8001")

        from unittest.mock import patch

        with patch(
            "services.adapter_client.httpx.AsyncClient",
            return_value=httpx.AsyncClient(transport=transport),
        ):
            with pytest.raises(httpx.HTTPStatusError):
                await client.get_loaded_adapters()


class TestAdapterHealthCheck:
    """Tests for AdapterClient.health_check."""

    async def test_health_check_returns_true(self) -> None:
        """Should return True on 200 response."""
        captured_requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_requests.append(request)
            return httpx.Response(200)

        transport = httpx.MockTransport(handler)
        client = AdapterClient(base_url="http://vllm:8001")

        from unittest.mock import patch

        with patch(
            "services.adapter_client.httpx.AsyncClient",
            return_value=httpx.AsyncClient(transport=transport),
        ):
            result = await client.health_check()

        assert result is True
        assert str(captured_requests[0].url) == "http://vllm:8001/health"

    async def test_health_check_returns_false_on_error(self) -> None:
        """Should return False when service is unreachable."""

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Connection refused")

        transport = httpx.MockTransport(handler)
        client = AdapterClient(base_url="http://vllm:8001")

        from unittest.mock import patch

        with patch(
            "services.adapter_client.httpx.AsyncClient",
            return_value=httpx.AsyncClient(transport=transport),
        ):
            result = await client.health_check()

        assert result is False

    async def test_health_check_returns_false_on_non_200(self) -> None:
        """Should return False on non-200 status."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503)

        transport = httpx.MockTransport(handler)
        client = AdapterClient(base_url="http://vllm:8001")

        from unittest.mock import patch

        with patch(
            "services.adapter_client.httpx.AsyncClient",
            return_value=httpx.AsyncClient(transport=transport),
        ):
            result = await client.health_check()

        assert result is False


class TestAdapterClientSharedHttpClient:
    """Tests that AdapterClient reuses an injected shared httpx.AsyncClient."""

    async def test_uses_shared_http_client_for_load_and_health(self) -> None:
        """When http_client is injected, all methods must route through it
        (no per-call httpx.AsyncClient constructed). Verified by counting the
        shared transport's invocations across load_adapter + health_check."""
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if request.url.path.endswith("/health"):
                return httpx.Response(200)
            return httpx.Response(
                200,
                content=json.dumps({"status": "loaded"}).encode(),
                headers={"content-type": "application/json"},
            )

        shared = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        client = AdapterClient(base_url="http://vllm:8001", http_client=shared)

        await client.load_adapter("schema_x", "s3://bucket/x")
        assert await client.health_check() is True
        # load_adapter + health_check = 2 requests, all through the shared transport
        assert call_count == 2

        # Shared client remains usable -- caller owns it
        resp = await shared.get("http://vllm:8001/health")
        assert resp.status_code == 200
        await shared.aclose()

    async def test_retry_loop_runtime_error_if_no_error_captured(self) -> None:
        """The runtime guard after the retry loop must be in place (not
        assert, which would be stripped under python -O). We can't easily
        reach this branch without monkeypatching internals, so verify the
        guard exists as a runtime check by ensuring load_adapter doesn't
        use an assert statement by reading the source."""
        import inspect

        from services import adapter_client

        source = inspect.getsource(adapter_client.AdapterClient.load_adapter)
        assert "assert last_error" not in source, (
            "assert last_error would be stripped under python -O"
        )
        assert "RuntimeError" in source, "must use runtime guard instead of assert"
