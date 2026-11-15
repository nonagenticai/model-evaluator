"""Tests for EvalService adapter loading and system prompt injection paths.

Covers:
- run_evaluation with adapter_client (load adapter before inference)
- run_evaluation when adapter has no adapter_path
- run_evaluation when adapter loading fails (connection error, 4xx, retry exhaustion)
- run_evaluation with system prompt injection from routing rule
- run_evaluation when routing rule fetch fails

RegistryClient HTTP tests live in test_registry_client.py -- do not
duplicate them here with MagicMock patterns; use httpx.MockTransport there.
"""

from typing import Any, cast
from unittest.mock import AsyncMock
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from models.domain import EvalCreateRequest, EvalStatus
from services.eval_service import EvalService


class TestAdapterLoadingPath:
    """Tests for the adapter loading branch in run_evaluation."""

    async def test_adapter_loaded_before_inference(
        self,
        session: AsyncSession,
        eval_service_with_adapter: EvalService,
        eval_payload: dict[str, Any],
        ground_truth_samples: list[dict[str, Any]],
    ) -> None:
        """Should load adapter before running inference when adapter_client is present."""
        svc = eval_service_with_adapter

        cast(Any, svc.collector_client).get_validated_samples = AsyncMock(
            return_value=ground_truth_samples
        )
        cast(Any, svc.registry_client).get_model = AsyncMock(
            return_value={"adapter_path": "s3://bucket/my-adapter"}
        )
        cast(Any, svc.registry_client).get_routing_rule = AsyncMock(
            return_value={"system_prompt": "You are a test agent."}
        )

        # Return exact matches
        async def mock_complete(
            endpoint: str, messages: list, grammar: str | None = None, **kw: Any
        ) -> dict:
            for sample in ground_truth_samples:
                if sample["messages"] == messages or (
                    len(messages) > 0
                    and messages[-1].get("content") == sample["messages"][-1].get("content")
                ):
                    return {"choices": [{"message": {"content": sample["response_content"]}}]}
            return {"choices": [{"message": {"content": '{"entities": []}'}}]}

        cast(Any, svc.inference_client).complete = AsyncMock(side_effect=mock_complete)

        request = EvalCreateRequest(**eval_payload)
        evaluation = await svc.create_evaluation(session, request)
        await svc.run_evaluation(session, evaluation.id)

        result = await svc.get(session, evaluation.id)
        assert result is not None
        assert result.status in (EvalStatus.APPROVED, EvalStatus.COMPLETED)

        # Verify adapter was loaded
        cast(Any, svc.adapter_client).load_adapter.assert_called_once_with(
            eval_payload["schema_id"], "s3://bucket/my-adapter"
        )

    async def test_no_adapter_path_continues(
        self,
        session: AsyncSession,
        eval_service_with_adapter: EvalService,
        eval_payload: dict[str, Any],
        ground_truth_samples: list[dict[str, Any]],
    ) -> None:
        """Should continue evaluation when model has no adapter_path."""
        svc = eval_service_with_adapter

        cast(Any, svc.collector_client).get_validated_samples = AsyncMock(
            return_value=ground_truth_samples
        )
        # No adapter_path in model info
        cast(Any, svc.registry_client).get_model = AsyncMock(return_value={"status": "trained"})
        cast(Any, svc.registry_client).get_routing_rule = AsyncMock(return_value={})
        cast(Any, svc.inference_client).complete = AsyncMock(
            return_value={"choices": [{"message": {"content": '{"entities": []}'}}]}
        )

        request = EvalCreateRequest(**eval_payload)
        evaluation = await svc.create_evaluation(session, request)
        await svc.run_evaluation(session, evaluation.id)

        result = await svc.get(session, evaluation.id)
        assert result is not None
        # Should still complete (not fail)
        assert result.status in (EvalStatus.APPROVED, EvalStatus.COMPLETED)
        # load_adapter should NOT have been called
        cast(Any, svc.adapter_client).load_adapter.assert_not_called()

    async def test_adapter_loading_failure_fails_evaluation(
        self,
        session: AsyncSession,
        eval_service_with_adapter: EvalService,
        eval_payload: dict[str, Any],
        ground_truth_samples: list[dict[str, Any]],
    ) -> None:
        """Should fail evaluation when adapter loading raises a transport error."""
        import httpx

        svc = eval_service_with_adapter

        cast(Any, svc.collector_client).get_validated_samples = AsyncMock(
            return_value=ground_truth_samples
        )
        cast(Any, svc.registry_client).get_model = AsyncMock(
            return_value={"adapter_path": "s3://bucket/bad-adapter"}
        )
        # AdapterClient.load_adapter only raises httpx exceptions in production;
        # exercise the narrowed httpx.HTTPError branch in eval_service.
        cast(Any, svc.adapter_client).load_adapter = AsyncMock(
            side_effect=httpx.ConnectError("Adapter manager unreachable")
        )

        request = EvalCreateRequest(**eval_payload)
        evaluation = await svc.create_evaluation(session, request)
        await svc.run_evaluation(session, evaluation.id)

        result = await svc.get(session, evaluation.id)
        assert result is not None
        assert result.status == EvalStatus.FAILED
        assert result.error_message == "Failed to load adapter before evaluation"
        assert result.completed_at is not None
        # Adapter must have been called exactly once with the expected args
        cast(Any, svc.adapter_client).load_adapter.assert_called_once_with(
            eval_payload["schema_id"], "s3://bucket/bad-adapter"
        )

    async def test_adapter_load_retry_exhausted_marks_failed(
        self,
        session: AsyncSession,
        eval_service_with_adapter: EvalService,
        eval_payload: dict[str, Any],
        ground_truth_samples: list[dict[str, Any]],
    ) -> None:
        """When the real adapter_client retries and still fails, the evaluation
        ends in FAILED with the exact error message from eval_service."""
        import httpx

        svc = eval_service_with_adapter

        cast(Any, svc.collector_client).get_validated_samples = AsyncMock(
            return_value=ground_truth_samples
        )
        cast(Any, svc.registry_client).get_model = AsyncMock(
            return_value={"adapter_path": "s3://bucket/flaky-adapter"}
        )
        # Simulate the retry loop exhausting: every attempt raises ConnectError.
        # Patch the full adapter_client.load_adapter so we don't have to wait 30s
        # for the real 3x10s retry delay.
        cast(Any, svc.adapter_client).load_adapter = AsyncMock(
            side_effect=httpx.ConnectError("MinIO unreachable")
        )

        request = EvalCreateRequest(**eval_payload)
        evaluation = await svc.create_evaluation(session, request)
        await svc.run_evaluation(session, evaluation.id)

        result = await svc.get(session, evaluation.id)
        assert result is not None
        assert result.status == EvalStatus.FAILED
        assert result.error_message == "Failed to load adapter before evaluation"
        assert result.completed_at is not None

    async def test_adapter_load_4xx_no_retry_still_fails_eval(
        self,
        session: AsyncSession,
        eval_service_with_adapter: EvalService,
        eval_payload: dict[str, Any],
        ground_truth_samples: list[dict[str, Any]],
    ) -> None:
        """A 4xx error from adapter-manager should NOT retry but must still
        mark the evaluation FAILED."""
        import httpx

        svc = eval_service_with_adapter

        cast(Any, svc.collector_client).get_validated_samples = AsyncMock(
            return_value=ground_truth_samples
        )
        cast(Any, svc.registry_client).get_model = AsyncMock(
            return_value={"adapter_path": "s3://bucket/bad-adapter"}
        )
        cast(Any, svc.adapter_client).load_adapter = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "Bad Request",
                request=httpx.Request("POST", "http://adapter-manager:8001/api/adapters/x/load"),
                response=httpx.Response(400),
            )
        )

        request = EvalCreateRequest(**eval_payload)
        evaluation = await svc.create_evaluation(session, request)
        await svc.run_evaluation(session, evaluation.id)

        result = await svc.get(session, evaluation.id)
        assert result is not None
        assert result.status == EvalStatus.FAILED
        assert result.error_message == "Failed to load adapter before evaluation"


class TestSystemPromptInjection:
    """Tests for system prompt injection from routing rules."""

    async def test_system_prompt_prepended_to_messages(
        self,
        session: AsyncSession,
        eval_service_with_adapter: EvalService,
        eval_payload: dict[str, Any],
    ) -> None:
        """Should prepend system prompt when routing rule provides one and messages lack it."""
        svc = eval_service_with_adapter

        # Samples WITHOUT system prompt
        samples_no_system = [
            {
                "id": str(uuid4()),
                "messages": [
                    {"role": "user", "content": "Find entities in: Apple announced iPhones."},
                ],
                "response_content": '{"entities": [{"text": "Apple", "type": "ORG"}]}',
            },
        ]

        cast(Any, svc.collector_client).get_validated_samples = AsyncMock(
            return_value=samples_no_system
        )
        cast(Any, svc.registry_client).get_model = AsyncMock(
            return_value={"adapter_path": "s3://bucket/adapter"}
        )
        cast(Any, svc.registry_client).get_routing_rule = AsyncMock(
            return_value={"system_prompt": "You are an entity extraction agent."}
        )

        captured_messages: list[list[dict]] = []

        async def mock_complete(
            endpoint: str, messages: list, grammar: str | None = None, **kw: Any
        ) -> dict:
            captured_messages.append(messages)
            return {
                "choices": [
                    {"message": {"content": '{"entities": [{"text": "Apple", "type": "ORG"}]}'}}
                ]
            }

        cast(Any, svc.inference_client).complete = AsyncMock(side_effect=mock_complete)

        request = EvalCreateRequest(**eval_payload)
        evaluation = await svc.create_evaluation(session, request)
        await svc.run_evaluation(session, evaluation.id)

        # Verify system prompt was prepended
        assert len(captured_messages) == 1
        assert captured_messages[0][0]["role"] == "system"
        assert captured_messages[0][0]["content"] == "You are an entity extraction agent."

    async def test_routing_rule_fetch_failure_continues(
        self,
        session: AsyncSession,
        eval_service_with_adapter: EvalService,
        eval_payload: dict[str, Any],
        ground_truth_samples: list[dict[str, Any]],
    ) -> None:
        """Should continue evaluation when routing rule fetch fails with httpx error.

        Exercises the narrowed httpx.HTTPError branch in eval_service: a 5xx
        from the registry must NOT abort the run -- the pipeline simply
        proceeds without a system prompt.
        """
        import httpx

        svc = eval_service_with_adapter

        cast(Any, svc.collector_client).get_validated_samples = AsyncMock(
            return_value=ground_truth_samples
        )
        cast(Any, svc.registry_client).get_model = AsyncMock(
            return_value={"adapter_path": "s3://bucket/adapter"}
        )
        cast(Any, svc.registry_client).get_routing_rule = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "Server Error",
                request=httpx.Request("GET", "http://registry/api/routing/x"),
                response=httpx.Response(500),
            )
        )

        captured_messages: list[list[dict[str, Any]]] = []

        async def mock_complete(
            endpoint: str, messages: list, grammar: str | None = None, **kw: Any
        ) -> dict[str, Any]:
            captured_messages.append(messages)
            return {"choices": [{"message": {"content": '{"entities": []}'}}]}

        cast(Any, svc.inference_client).complete = AsyncMock(side_effect=mock_complete)

        request = EvalCreateRequest(**eval_payload)
        evaluation = await svc.create_evaluation(session, request)
        await svc.run_evaluation(session, evaluation.id)

        result = await svc.get(session, evaluation.id)
        assert result is not None
        # Should still complete even if routing rule fetch failed
        assert result.status in (EvalStatus.APPROVED, EvalStatus.COMPLETED)
        assert result.metrics is not None
        # Inference still called for every sample
        assert len(captured_messages) == len(ground_truth_samples)
        # No system prompt was injected because routing failed -- the very
        # first message must be whatever the sample originally had (no role:
        # "system" prepended by us).
        for original_sample, sent in zip(ground_truth_samples, captured_messages, strict=True):
            assert sent[0] == original_sample["messages"][0]
