"""HTTP client for the adapter-manager sidecar.

Loads and unloads LoRA adapters in the RTX server before evaluation
so that inference runs against the correct fine-tuned model.

Includes retry logic for transient failures (MinIO outages, network blips).
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import structlog

logger = structlog.stdlib.get_logger(__name__)

# Retry config for adapter loading (covers short MinIO blips)
_LOAD_MAX_RETRIES = 3
_LOAD_RETRY_DELAY = 10  # seconds between retries


class AdapterClient:
    """Async client for the adapter-manager sidecar (port 8001 on vllm pod).

    Accepts an optional shared ``httpx.AsyncClient`` so TCP connections are
    reused across calls -- important for the retry loop and for batched
    evaluation runs. The caller owns the shared client and is responsible
    for closing it (typically in the FastAPI lifespan shutdown).
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 60.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client = http_client

    async def load_adapter(self, schema_id: str, adapter_path: str) -> dict[str, Any]:
        """Load a LoRA adapter into the RTX server.

        Retries on 5xx errors and connection failures to handle transient
        MinIO outages that prevent the adapter-manager from downloading adapters.

        Args:
            schema_id: Schema identifier (used as the adapter name).
            adapter_path: MinIO path to the PEFT adapter directory.

        Returns:
            Response body from adapter-manager.

        Raises:
            httpx.HTTPStatusError: If the adapter-manager returns a non-2xx status
                after all retries are exhausted.
            httpx.ConnectError / httpx.ReadError: If the adapter-manager is
                unreachable after all retries.
        """
        url = f"{self._base_url}/api/adapters/{schema_id}/load"
        payload = {"adapter_path": adapter_path}

        logger.info(
            "loading_adapter",
            schema_id=schema_id,
            adapter_path=adapter_path,
        )

        last_error: Exception | None = None
        for attempt in range(1, _LOAD_MAX_RETRIES + 1):
            try:
                response = await self._post(url, payload)
                response.raise_for_status()

                result = response.json()
                logger.info(
                    "adapter_loaded",
                    schema_id=schema_id,
                    status=result.get("status"),
                )
                return result
            except (httpx.HTTPStatusError, httpx.ConnectError, httpx.ReadError) as exc:
                last_error = exc
                # Only retry on 5xx (server) errors or connection issues
                if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code < 500:
                    raise
                if attempt < _LOAD_MAX_RETRIES:
                    logger.warning(
                        "adapter_load_retrying",
                        schema_id=schema_id,
                        attempt=attempt,
                        max_attempts=_LOAD_MAX_RETRIES,
                        error=str(exc),
                    )
                    await asyncio.sleep(_LOAD_RETRY_DELAY)

        # All retries exhausted: last_error is guaranteed set because the only
        # way to exit the loop without returning is to hit the except branch.
        if last_error is None:
            raise RuntimeError("adapter_load retry loop exited without raising or returning")
        logger.error(
            "adapter_load_retries_exhausted",
            schema_id=schema_id,
            attempts=_LOAD_MAX_RETRIES,
            error=str(last_error),
        )
        raise last_error

    async def get_loaded_adapters(self) -> dict[str, Any]:
        """List all currently loaded adapters.

        Returns:
            Response body with loaded adapter info.
        """
        url = f"{self._base_url}/api/adapters"
        response = await self._get(url)
        response.raise_for_status()
        return response.json()

    async def health_check(self) -> bool:
        """Check if adapter-manager is reachable."""
        try:
            response = await self._get(f"{self._base_url}/health")
            return response.status_code == 200
        except Exception:
            logger.debug("adapter_manager_health_check_failed", exc_info=True)
            return False

    async def _post(self, url: str, json: dict[str, Any]) -> httpx.Response:
        """POST via the shared client when provided, otherwise per-call client."""
        if self._client is not None:
            return await self._client.post(url, json=json, timeout=self._timeout)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            return await client.post(url, json=json)

    async def _get(self, url: str) -> httpx.Response:
        """GET via the shared client when provided, otherwise per-call client."""
        if self._client is not None:
            return await self._client.get(url, timeout=self._timeout)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            return await client.get(url)
