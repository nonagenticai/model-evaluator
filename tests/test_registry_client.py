"""Tests for the RegistryClient.

Covers:
- __init__: sets base_url and timeout
- update_model_status: PATCH request with correct payload
- get_model: GET request for model info
- get_routing_rule: GET routing rule, 404 handling
- report_evaluation: convenience method that calls update_model_status
- health_check: returns True/False

Uses httpx.MockTransport for realistic HTTP testing without real network calls.
"""

import json
from unittest.mock import patch
from uuid import uuid4

import httpx
import pytest

from models.domain import EvalMetrics
from services.registry_client import RegistryClient


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


class TestRegistryClientInit:
    """Tests for RegistryClient initialization."""

    def test_strips_trailing_slash(self) -> None:
        """base_url should have trailing slash removed."""
        client = RegistryClient(base_url="http://registry:8155/", timeout=10.0)
        assert client._base_url == "http://registry:8155"

    def test_default_timeout(self) -> None:
        """Should use 30s default timeout."""
        client = RegistryClient(base_url="http://registry:8155")
        assert client._timeout == 30.0


class TestUpdateModelStatus:
    """Tests for RegistryClient.update_model_status."""

    async def test_update_status_basic(self) -> None:
        """Should send PATCH to /api/models/{id}/status with status payload."""
        model_id = uuid4()
        captured_requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_requests.append(request)
            return httpx.Response(
                200,
                content=json.dumps({"status": "approved"}).encode(),
                headers={"content-type": "application/json"},
            )

        transport = httpx.MockTransport(handler)
        client = RegistryClient(base_url="http://registry:8155")

        with patch(
            "services.registry_client.httpx.AsyncClient",
            return_value=httpx.AsyncClient(transport=transport),
        ):
            result = await client.update_model_status(model_id, "approved")

        assert result == {"status": "approved"}

        req = captured_requests[0]
        assert req.method == "PATCH"
        assert f"/api/models/{model_id}/status" in str(req.url)
        payload = json.loads(req.content)
        assert payload["status"] == "approved"

    async def test_update_status_with_metrics(self) -> None:
        """Should include eval_score and eval_metrics when provided."""
        model_id = uuid4()
        captured_payloads: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_payloads.append(json.loads(request.content))
            return httpx.Response(
                200,
                content=json.dumps({"status": "approved"}).encode(),
                headers={"content-type": "application/json"},
            )

        transport = httpx.MockTransport(handler)
        client = RegistryClient(base_url="http://registry:8155")
        metrics_dict = {"composite_score": 0.95}

        with patch(
            "services.registry_client.httpx.AsyncClient",
            return_value=httpx.AsyncClient(transport=transport),
        ):
            await client.update_model_status(
                model_id, "approved", eval_score=0.95, eval_metrics=metrics_dict
            )

        payload = captured_payloads[0]
        assert payload["eval_score"] == 0.95
        assert payload["eval_metrics"] == metrics_dict
        assert payload["status"] == "approved"

    async def test_update_status_raises_on_error(self) -> None:
        """Should propagate HTTP errors."""
        transport = _make_transport(status_code=404)
        client = RegistryClient(base_url="http://registry:8155")

        with patch(
            "services.registry_client.httpx.AsyncClient",
            return_value=httpx.AsyncClient(transport=transport),
        ):
            with pytest.raises(httpx.HTTPStatusError):
                await client.update_model_status(uuid4(), "approved")


class TestGetModel:
    """Tests for RegistryClient.get_model."""

    async def test_get_model_success(self) -> None:
        """Should GET model info with correct URL."""
        model_id = uuid4()
        model_data = {"id": str(model_id), "adapter_path": "s3://bucket/adapter"}
        captured_requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_requests.append(request)
            return httpx.Response(
                200,
                content=json.dumps(model_data).encode(),
                headers={"content-type": "application/json"},
            )

        transport = httpx.MockTransport(handler)
        client = RegistryClient(base_url="http://registry:8155")

        with patch(
            "services.registry_client.httpx.AsyncClient",
            return_value=httpx.AsyncClient(transport=transport),
        ):
            result = await client.get_model(model_id)

        assert result["adapter_path"] == "s3://bucket/adapter"
        assert f"/api/models/{model_id}" in str(captured_requests[0].url)

    async def test_get_model_raises_on_404(self) -> None:
        """Should propagate 404 errors."""
        transport = _make_transport(status_code=404)
        client = RegistryClient(base_url="http://registry:8155")

        with patch(
            "services.registry_client.httpx.AsyncClient",
            return_value=httpx.AsyncClient(transport=transport),
        ):
            with pytest.raises(httpx.HTTPStatusError) as exc_info:
                await client.get_model(uuid4())
            assert exc_info.value.response.status_code == 404


class TestGetRoutingRule:
    """Tests for RegistryClient.get_routing_rule."""

    async def test_get_routing_rule_success(self) -> None:
        """Should GET routing rule and return parsed JSON."""
        rule_data = {"schema_id": "ks_test", "system_prompt": "You are a test agent."}

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=json.dumps(rule_data).encode(),
                headers={"content-type": "application/json"},
            )

        transport = httpx.MockTransport(handler)
        client = RegistryClient(base_url="http://registry:8155")

        with patch(
            "services.registry_client.httpx.AsyncClient",
            return_value=httpx.AsyncClient(transport=transport),
        ):
            result = await client.get_routing_rule("ks_test")

        assert result["system_prompt"] == "You are a test agent."

    async def test_get_routing_rule_404_returns_empty(self) -> None:
        """Should return empty dict when routing rule doesn't exist (404)."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, content=b"")

        transport = httpx.MockTransport(handler)
        client = RegistryClient(base_url="http://registry:8155")

        with patch(
            "services.registry_client.httpx.AsyncClient",
            return_value=httpx.AsyncClient(transport=transport),
        ):
            result = await client.get_routing_rule("nonexistent")

        assert result == {}

    async def test_get_routing_rule_raises_on_server_error(self) -> None:
        """Should propagate non-404 HTTP errors."""
        transport = _make_transport(status_code=500)
        client = RegistryClient(base_url="http://registry:8155")

        with patch(
            "services.registry_client.httpx.AsyncClient",
            return_value=httpx.AsyncClient(transport=transport),
        ):
            with pytest.raises(httpx.HTTPStatusError) as exc_info:
                await client.get_routing_rule("test")
            assert exc_info.value.response.status_code == 500


class TestReportEvaluation:
    """Tests for RegistryClient.report_evaluation."""

    async def test_report_approved(self) -> None:
        """Should call update_model_status with 'approved' status and composite score."""
        model_id = uuid4()
        metrics = EvalMetrics(
            json_parse_rate=1.0,
            schema_valid_rate=0.9,
            exact_match_rate=0.8,
            avg_latency_ms=100.0,
            p95_latency_ms=200.0,
            composite_score=0.87,
            total_samples=50,
            passed_samples=40,
            failed_samples=10,
        )
        captured_payloads: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_payloads.append(json.loads(request.content))
            return httpx.Response(
                200,
                content=json.dumps({"status": "approved"}).encode(),
                headers={"content-type": "application/json"},
            )

        transport = httpx.MockTransport(handler)
        client = RegistryClient(base_url="http://registry:8155")

        with patch(
            "services.registry_client.httpx.AsyncClient",
            return_value=httpx.AsyncClient(transport=transport),
        ):
            result = await client.report_evaluation(model_id, metrics, approved=True)

        assert result == {"status": "approved"}
        payload = captured_payloads[0]
        assert payload["status"] == "approved"
        assert payload["eval_score"] == 0.87
        assert "eval_metrics" in payload
        assert payload["eval_metrics"]["composite_score"] == 0.87

    async def test_report_not_approved(self) -> None:
        """Should call update_model_status with 'evaluated' status when not approved."""
        model_id = uuid4()
        metrics = EvalMetrics(
            json_parse_rate=1.0,
            schema_valid_rate=0.5,
            exact_match_rate=0.3,
            avg_latency_ms=200.0,
            p95_latency_ms=400.0,
            composite_score=0.47,
            total_samples=50,
            passed_samples=15,
            failed_samples=35,
        )
        captured_payloads: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_payloads.append(json.loads(request.content))
            return httpx.Response(
                200,
                content=json.dumps({"status": "evaluated"}).encode(),
                headers={"content-type": "application/json"},
            )

        transport = httpx.MockTransport(handler)
        client = RegistryClient(base_url="http://registry:8155")

        with patch(
            "services.registry_client.httpx.AsyncClient",
            return_value=httpx.AsyncClient(transport=transport),
        ):
            result = await client.report_evaluation(model_id, metrics, approved=False)

        assert result == {"status": "evaluated"}
        payload = captured_payloads[0]
        assert payload["status"] == "evaluated"
        assert payload["eval_score"] == 0.47


class TestRegistryHealthCheck:
    """Tests for RegistryClient.health_check."""

    async def test_health_check_returns_true(self) -> None:
        """Should return True on 200 response."""
        transport = _make_transport(status_code=200)
        client = RegistryClient(base_url="http://registry:8155")

        with patch(
            "services.registry_client.httpx.AsyncClient",
            return_value=httpx.AsyncClient(transport=transport),
        ):
            result = await client.health_check()

        assert result is True

    async def test_health_check_returns_false_on_error(self) -> None:
        """Should return False when service is unreachable."""
        transport = _make_transport(raise_error=httpx.ConnectError("Connection refused"))
        client = RegistryClient(base_url="http://registry:8155")

        with patch(
            "services.registry_client.httpx.AsyncClient",
            return_value=httpx.AsyncClient(transport=transport),
        ):
            result = await client.health_check()

        assert result is False

    async def test_health_check_returns_false_on_non_200(self) -> None:
        """Should return False on non-200 status."""
        transport = _make_transport(status_code=503)
        client = RegistryClient(base_url="http://registry:8155")

        with patch(
            "services.registry_client.httpx.AsyncClient",
            return_value=httpx.AsyncClient(transport=transport),
        ):
            result = await client.health_check()

        assert result is False


class TestRegistryClientSharedHttpClient:
    """Tests that RegistryClient reuses an injected shared httpx.AsyncClient."""

    async def test_uses_shared_http_client_for_get_and_patch(self) -> None:
        """When http_client is injected, update_model_status (PATCH),
        get_model (GET), get_routing_rule (GET), and health_check (GET) all
        route through the same transport. Verified by a call counter."""
        call_count = 0
        model_id = uuid4()

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            path = request.url.path
            if request.method == "PATCH" and path.endswith(f"/api/models/{model_id}/status"):
                return httpx.Response(
                    200,
                    content=json.dumps({"status": "approved"}).encode(),
                    headers={"content-type": "application/json"},
                )
            if request.method == "GET" and path.endswith(f"/api/models/{model_id}"):
                return httpx.Response(
                    200,
                    content=json.dumps({"id": str(model_id), "adapter_path": "s3://x"}).encode(),
                    headers={"content-type": "application/json"},
                )
            if request.method == "GET" and path.endswith("/api/routing/ks_x"):
                return httpx.Response(
                    200,
                    content=json.dumps({"system_prompt": "p"}).encode(),
                    headers={"content-type": "application/json"},
                )
            if request.method == "GET" and path.endswith("/health"):
                return httpx.Response(200)
            return httpx.Response(500, content=b'{"error": "unexpected"}')

        shared = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        client = RegistryClient(base_url="http://registry:8155", http_client=shared)

        patch_result = await client.update_model_status(model_id, "approved")
        assert patch_result == {"status": "approved"}

        model = await client.get_model(model_id)
        assert model["adapter_path"] == "s3://x"

        routing = await client.get_routing_rule("ks_x")
        assert routing["system_prompt"] == "p"

        assert await client.health_check() is True

        # 4 requests, all through the shared transport -- no per-call client
        assert call_count == 4

        # Shared client is still the caller's -- they close it
        await shared.aclose()

    async def test_get_routing_rule_404_via_shared_client(self) -> None:
        """404 on get_routing_rule must return {} even when using the shared client."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404)

        shared = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        client = RegistryClient(base_url="http://registry:8155", http_client=shared)

        result = await client.get_routing_rule("nonexistent")
        assert result == {}
        await shared.aclose()
