"""HTTP client for the model-registry service.

Reports evaluation results and status updates back to the model registry
so the deployment pipeline knows whether a model is approved.
"""

from typing import Any
from uuid import UUID

import httpx
import structlog

from models.domain import EvalMetrics

logger = structlog.stdlib.get_logger(__name__)


class RegistryClient:
    """Async client for the model-registry service.

    Accepts an optional shared ``httpx.AsyncClient`` so TCP connections are
    reused across calls. The caller owns the shared client and is
    responsible for closing it (typically in the FastAPI lifespan shutdown).
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client = http_client

    async def update_model_status(
        self,
        model_id: UUID,
        status: str,
        eval_score: float | None = None,
        eval_metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Update the status of a model version in the registry.

        Args:
            model_id: UUID of the model version.
            status: Target lifecycle status (e.g. 'evaluated', 'approved').
            eval_score: Composite evaluation score.
            eval_metrics: Full evaluation metrics dictionary.

        Returns:
            Response body from the model-registry.

        Raises:
            httpx.HTTPStatusError: If the registry returns a non-2xx status.
        """
        url = f"{self._base_url}/api/models/{model_id}/status"
        payload: dict[str, Any] = {"status": status}

        if eval_score is not None:
            payload["eval_score"] = eval_score
        if eval_metrics is not None:
            payload["eval_metrics"] = eval_metrics

        logger.info(
            "updating_model_status",
            model_id=str(model_id),
            status=status,
            eval_score=eval_score,
        )

        response = await self._patch(url, payload)
        response.raise_for_status()

        result = response.json()
        logger.info(
            "model_status_updated",
            model_id=str(model_id),
            status=status,
        )

        return result

    async def get_model(self, model_id: UUID) -> dict[str, Any]:
        """Fetch a model version record from the registry.

        Args:
            model_id: UUID of the model version.

        Returns:
            Model version dict with adapter_path, status, etc.

        Raises:
            httpx.HTTPStatusError: If the registry returns a non-2xx status.
        """
        url = f"{self._base_url}/api/models/{model_id}"
        response = await self._get(url)
        response.raise_for_status()
        return response.json()

    async def get_routing_rule(self, schema_id: str) -> dict[str, Any]:
        """Fetch the routing rule for a schema.

        Args:
            schema_id: Schema identifier.

        Returns:
            Routing rule dict with json_schema, system_prompt, etc.
        """
        url = f"{self._base_url}/api/routing/{schema_id}"
        response = await self._get(url)
        if response.status_code == 404:
            return {}
        response.raise_for_status()
        return response.json()

    async def report_evaluation(
        self,
        model_id: UUID,
        metrics: EvalMetrics,
        approved: bool,
    ) -> dict[str, Any]:
        """Report evaluation results and update model status accordingly.

        This is a convenience method that converts metrics to the registry's
        expected format and sets the appropriate status.

        Args:
            model_id: UUID of the model version.
            metrics: Computed evaluation metrics.
            approved: Whether the model passed the gate.

        Returns:
            Response body from the model-registry.
        """
        target_status = "approved" if approved else "evaluated"
        metrics_dict = metrics.model_dump()

        return await self.update_model_status(
            model_id=model_id,
            status=target_status,
            eval_score=metrics.composite_score,
            eval_metrics=metrics_dict,
        )

    async def health_check(self) -> bool:
        """Check whether the model-registry service is reachable."""
        try:
            response = await self._get(f"{self._base_url}/health", timeout=5.0)
            return response.status_code == 200
        except Exception:
            logger.debug("health_check_failed", exc_info=True)
            return False

    async def _get(self, url: str, timeout: float | None = None) -> httpx.Response:
        """GET via the shared client when provided, otherwise per-call client."""
        effective_timeout = self._timeout if timeout is None else timeout
        if self._client is not None:
            return await self._client.get(url, timeout=effective_timeout)
        async with httpx.AsyncClient(timeout=effective_timeout) as client:
            return await client.get(url)

    async def _patch(self, url: str, json: dict[str, Any]) -> httpx.Response:
        """PATCH via the shared client when provided, otherwise per-call client."""
        if self._client is not None:
            return await self._client.patch(url, json=json, timeout=self._timeout)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            return await client.patch(url, json=json)
