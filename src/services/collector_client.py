"""HTTP client for the data-collector service.

Fetches validated ground truth samples that are used as the evaluation
dataset for model validation.
"""

from typing import Any

import httpx
import structlog

logger = structlog.stdlib.get_logger(__name__)


class CollectorClient:
    """Async client for the data-collector service.

    Accepts a shared ``httpx.AsyncClient`` so that TCP connections are reused
    across requests instead of creating a fresh connection per call.
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client = http_client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = http_client is None

    async def get_validated_samples(
        self,
        schema_id: str,
        limit: int = 100,
        job_id: str | None = None,
        inference_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch validated ground truth samples from data-collector.

        Only retrieves samples where ``validation_passed=True`` so they can
        serve as reliable ground truth for evaluation.

        Args:
            schema_id: Schema identifier to fetch samples for.
            limit: Maximum number of samples to retrieve.
            job_id: Optional job identifier to narrow the sample set.
            inference_id: Optional inference identifier to narrow the sample set.

        Returns:
            List of sample dictionaries, each containing at minimum:
            - id: sample UUID
            - messages: input messages array
            - response_content: expected output string
            - inference_id: inference identifier (when available)

        Raises:
            httpx.HTTPStatusError: If the data-collector returns a non-2xx status.
            httpx.ConnectError: If the data-collector is unreachable.
        """
        url = f"{self._base_url}/api/samples"
        params: dict[str, Any] = {
            "schema_id": schema_id,
            "validation_passed": True,
            "limit": limit,
        }
        if job_id is not None:
            params["job_id"] = job_id
        if inference_id is not None:
            params["inference_id"] = inference_id

        logger.info(
            "fetching_ground_truth",
            schema_id=schema_id,
            limit=limit,
            url=url,
        )

        response = await self._client.get(url, params=params)
        response.raise_for_status()

        samples = response.json()
        logger.info(
            "ground_truth_fetched",
            schema_id=schema_id,
            count=len(samples),
        )

        return samples

    async def health_check(self) -> bool:
        """Check whether the data-collector service is reachable."""
        try:
            response = await self._client.get(f"{self._base_url}/health")
            return response.status_code == 200
        except Exception:
            logger.debug("health_check_failed", exc_info=True)
            return False

    async def close(self) -> None:
        """Close the underlying HTTP client if we own it."""
        if self._owns_client:
            await self._client.aclose()
