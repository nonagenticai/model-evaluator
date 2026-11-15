"""Tests for the InferenceClient.

Covers:
- __init__: sets timeout, accepts shared http_client
- complete: sends chat completion via shared client, includes grammar, handles errors
- health_check: returns True/False based on endpoint availability
- close: closes the underlying httpx.AsyncClient when owned

Uses httpx.MockTransport for realistic HTTP testing without real network calls.
"""

import json

import httpx
import pytest

from services.inference_client import InferenceClient


def _make_transport(
    *,
    status_code: int = 200,
    json_data: dict | None = None,
    raise_error: Exception | None = None,
) -> httpx.MockTransport:
    """Build a MockTransport that returns a fixed response."""

    def handler(request: httpx.Request) -> httpx.Response:
        if raise_error:
            raise raise_error
        body = json.dumps(json_data or {}).encode()
        return httpx.Response(
            status_code, content=body, headers={"content-type": "application/json"}
        )

    return httpx.MockTransport(handler)


class TestInferenceClientInit:
    """Tests for InferenceClient initialization."""

    def test_default_timeout(self) -> None:
        """Should use 30s default timeout."""
        client = InferenceClient()
        assert client._timeout == 30.0

    def test_custom_timeout(self) -> None:
        """Should accept custom timeout."""
        client = InferenceClient(timeout=60.0)
        assert client._timeout == 60.0

    def test_accepts_shared_http_client(self) -> None:
        """Should store the provided httpx.AsyncClient."""
        transport = _make_transport()
        shared_client = httpx.AsyncClient(transport=transport)
        client = InferenceClient(http_client=shared_client)
        assert client._client is shared_client
        assert client._owns_client is False

    def test_creates_default_client_when_none_provided(self) -> None:
        """Should create its own httpx.AsyncClient when none is passed."""
        client = InferenceClient(timeout=15.0)
        assert isinstance(client._client, httpx.AsyncClient)
        assert client._owns_client is True


class TestComplete:
    """Tests for InferenceClient.complete."""

    async def test_complete_success(self) -> None:
        """Should POST to /v1/chat/completions and return the response dict."""
        response_data = {"choices": [{"message": {"content": '{"entities": []}'}}]}
        captured_requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_requests.append(request)
            return httpx.Response(
                200,
                content=json.dumps(response_data).encode(),
                headers={"content-type": "application/json"},
            )

        transport = httpx.MockTransport(handler)
        http_client = httpx.AsyncClient(transport=transport)
        client = InferenceClient(timeout=10.0, http_client=http_client)

        result = await client.complete(
            endpoint="http://pi-node:8080",
            messages=[{"role": "user", "content": "test"}],
            model="test-model",
        )

        assert result["choices"][0]["message"]["content"] == '{"entities": []}'

        # Verify correct URL
        assert len(captured_requests) == 1
        assert str(captured_requests[0].url) == "http://pi-node:8080/v1/chat/completions"

    async def test_complete_sends_correct_payload(self) -> None:
        """Should send model, messages, temperature, max_tokens, stream in the payload."""
        captured_payloads: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            captured_payloads.append(payload)
            return httpx.Response(
                200,
                content=json.dumps({"choices": [{"message": {"content": "{}"}}]}).encode(),
                headers={"content-type": "application/json"},
            )

        transport = httpx.MockTransport(handler)
        http_client = httpx.AsyncClient(transport=transport)
        client = InferenceClient(http_client=http_client)

        await client.complete(
            endpoint="http://pi-node:8080",
            messages=[{"role": "user", "content": "test"}],
            model="ks_EntitySpotterAgent",
        )

        payload = captured_payloads[0]
        assert payload["model"] == "ks_EntitySpotterAgent"
        assert payload["messages"] == [{"role": "user", "content": "test"}]
        assert payload["temperature"] == 0.0
        assert payload["max_tokens"] == 2048
        assert payload["stream"] is False
        assert "guided_grammar" not in payload

    async def test_complete_with_grammar(self) -> None:
        """Should include grammar in payload when provided."""
        captured_payloads: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_payloads.append(json.loads(request.content))
            return httpx.Response(
                200,
                content=json.dumps({"choices": [{"message": {"content": "{}"}}]}).encode(),
                headers={"content-type": "application/json"},
            )

        transport = httpx.MockTransport(handler)
        http_client = httpx.AsyncClient(transport=transport)
        client = InferenceClient(http_client=http_client)
        grammar = 'root ::= "{}" ws'

        await client.complete(
            endpoint="http://pi-node:8080",
            messages=[{"role": "user", "content": "test"}],
            grammar=grammar,
            model="test-model",
        )

        assert captured_payloads[0]["guided_grammar"] == grammar

    async def test_complete_strips_trailing_slash(self) -> None:
        """Should strip trailing slash from endpoint URL."""
        captured_requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_requests.append(request)
            return httpx.Response(
                200,
                content=json.dumps({"choices": [{"message": {"content": "{}"}}]}).encode(),
                headers={"content-type": "application/json"},
            )

        transport = httpx.MockTransport(handler)
        http_client = httpx.AsyncClient(transport=transport)
        client = InferenceClient(http_client=http_client)

        await client.complete(
            endpoint="http://pi-node:8080/",
            messages=[{"role": "user", "content": "test"}],
            model="test-model",
        )

        assert str(captured_requests[0].url) == "http://pi-node:8080/v1/chat/completions"

    async def test_complete_raises_on_http_error(self) -> None:
        """Should propagate HTTP errors."""
        transport = _make_transport(status_code=500)
        http_client = httpx.AsyncClient(transport=transport)
        client = InferenceClient(http_client=http_client)

        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await client.complete(
                endpoint="http://pi-node:8080",
                messages=[{"role": "user", "content": "test"}],
                model="test-model",
            )
        assert exc_info.value.response.status_code == 500

    async def test_shared_client_reused_across_calls(self) -> None:
        """The same httpx.AsyncClient instance should be used for all requests."""
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(
                200,
                content=json.dumps({"choices": [{"message": {"content": "{}"}}]}).encode(),
                headers={"content-type": "application/json"},
            )

        transport = httpx.MockTransport(handler)
        http_client = httpx.AsyncClient(transport=transport)
        client = InferenceClient(http_client=http_client)

        await client.complete(
            "http://a:8080", messages=[{"role": "user", "content": "1"}], model="test-model"
        )
        await client.complete(
            "http://b:8080", messages=[{"role": "user", "content": "2"}], model="test-model"
        )

        assert call_count == 2


class TestInferenceHealthCheck:
    """Tests for InferenceClient.health_check."""

    async def test_health_check_returns_true(self) -> None:
        """Should return True on 200 response."""
        captured_requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_requests.append(request)
            return httpx.Response(200)

        transport = httpx.MockTransport(handler)
        http_client = httpx.AsyncClient(transport=transport)
        client = InferenceClient(http_client=http_client)

        result = await client.health_check("http://pi-node:8080")
        assert result is True
        assert str(captured_requests[0].url) == "http://pi-node:8080/health"

    async def test_health_check_returns_false_on_error(self) -> None:
        """Should return False when endpoint is unreachable."""
        transport = _make_transport(raise_error=httpx.ConnectError("Connection refused"))
        http_client = httpx.AsyncClient(transport=transport)
        client = InferenceClient(http_client=http_client)

        result = await client.health_check("http://pi-node:8080")
        assert result is False

    async def test_health_check_strips_trailing_slash(self) -> None:
        """Should handle trailing slash in endpoint URL."""
        captured_requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_requests.append(request)
            return httpx.Response(200)

        transport = httpx.MockTransport(handler)
        http_client = httpx.AsyncClient(transport=transport)
        client = InferenceClient(http_client=http_client)

        await client.health_check("http://pi-node:8080/")
        assert str(captured_requests[0].url) == "http://pi-node:8080/health"


class TestInferenceClose:
    """Tests for InferenceClient.close."""

    async def test_close_closes_owned_client(self) -> None:
        """close() should close the client when auto-created."""
        client = InferenceClient()
        assert client._owns_client is True
        await client.close()

    async def test_close_skips_shared_client(self) -> None:
        """close() should not close an injected shared client."""
        transport = _make_transport(json_data={"choices": [{"message": {"content": "{}"}}]})
        shared_client = httpx.AsyncClient(transport=transport)
        client = InferenceClient(http_client=shared_client)
        assert client._owns_client is False

        await client.close()

        # Shared client should still be usable
        result = await client.complete(
            "http://test:8080", messages=[{"role": "user", "content": "x"}], model="test-model"
        )
        assert "choices" in result
        await shared_client.aclose()
