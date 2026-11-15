"""HTTP client for OpenAI-compatible chat completion endpoints (vLLM).

Sends chat completion requests with optional GBNF grammar constraints
(routed via vLLM's guided-decoding `guided_grammar` field) to guarantee
syntactically valid JSON output from the model.
"""

from typing import Any

import httpx
import structlog

logger = structlog.stdlib.get_logger(__name__)


class InferenceClient:
    """Async client for vLLM /v1/chat/completions with optional GBNF grammar.

    Accepts a shared ``httpx.AsyncClient`` so that TCP connections are reused
    across requests -- critical in tight evaluation loops that may call
    inference hundreds of times.
    """

    def __init__(
        self,
        timeout: float = 30.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._timeout = timeout
        self._client = http_client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = http_client is None

    async def complete(
        self,
        endpoint: str,
        messages: list[dict[str, str]],
        grammar: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        *,
        model: str,
    ) -> dict[str, Any]:
        """Send a chat completion request to a vLLM endpoint.

        Args:
            endpoint: Base URL of the vLLM server
                (e.g. ``http://vllm.example.com:8000``).
            messages: Chat messages array in OpenAI format.
            grammar: Optional GBNF grammar string for constrained decoding
                (passed to vLLM as ``guided_grammar``).
            temperature: Sampling temperature (0.0 = greedy).
            max_tokens: Maximum tokens to generate.
            model: Model identifier (e.g. schema_id) -- required, no default.
                For LoRA adapters, this is the adapter name registered with
                vLLM's ``/v1/load_lora_adapter`` endpoint.

        Returns:
            Full OpenAI-compatible response dict with ``choices``.

        Raises:
            httpx.HTTPStatusError: If the endpoint returns a non-2xx status.
            httpx.ConnectError: If the endpoint is unreachable.
            httpx.ReadTimeout: If inference exceeds the timeout.
        """
        url = f"{endpoint.rstrip('/')}/v1/chat/completions"

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        if grammar is not None:
            payload["guided_grammar"] = grammar

        logger.debug(
            "inference_request",
            endpoint=endpoint,
            message_count=len(messages),
            has_grammar=grammar is not None,
        )

        response = await self._client.post(url, json=payload)
        response.raise_for_status()

        result = response.json()
        logger.debug(
            "inference_response",
            endpoint=endpoint,
            choices_count=len(result.get("choices", [])),
        )

        return result

    async def health_check(self, endpoint: str) -> bool:
        """Check whether a vLLM endpoint is reachable.

        Attempts to hit the ``/health`` endpoint exposed by vLLM's OpenAI
        API server.
        """
        try:
            url = f"{endpoint.rstrip('/')}/health"
            response = await self._client.get(url)
            return response.status_code == 200
        except Exception:
            logger.debug("health_check_failed", exc_info=True)
            return False

    async def close(self) -> None:
        """Close the underlying HTTP client if we own it."""
        if self._owns_client:
            await self._client.aclose()
