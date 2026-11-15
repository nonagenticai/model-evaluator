"""Extended tests for EvalService covering remaining uncovered branches.

Covers:
- list_evaluations with status filter
- run_evaluation with non-existent eval_id
- _evaluate_sample schema validation
- _validate_schema with invalid JSON and invalid schema
- _try_parse_json with various inputs
- registry notification failure paths
- approve/reject for non-existent evaluations
- _update_status for non-existent row
"""

from typing import Any, cast
from unittest.mock import AsyncMock
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from models.domain import EvalCreateRequest, EvalMetrics, EvalStatus
from services.eval_service import EvalService


class TestListEvaluationsExtended:
    """Extended tests for list_evaluations."""

    async def test_list_with_status_filter(
        self,
        session: AsyncSession,
        eval_service: EvalService,
        eval_payload: dict[str, Any],
    ) -> None:
        """Should filter evaluations by status and verify result data."""
        request = EvalCreateRequest(**eval_payload)
        created = await eval_service.create_evaluation(session, request)

        # Filter by pending status
        evals, total = await eval_service.list_evaluations(session, status="pending")
        assert total >= 1
        for e in evals:
            assert e.status == EvalStatus.PENDING
            assert e.model_id is not None
            assert e.schema_id is not None

        # Verify our created eval is in the results
        found = [e for e in evals if e.id == created.id]
        assert len(found) == 1
        assert found[0].schema_id == eval_payload["schema_id"]

        # Filter by a status with no results
        evals, total = await eval_service.list_evaluations(session, status="approved")
        assert total == 0
        assert evals == []

    async def test_list_with_both_filters(
        self,
        session: AsyncSession,
        eval_service: EvalService,
        eval_payload: dict[str, Any],
    ) -> None:
        """Should filter by both schema_id and status."""
        request = EvalCreateRequest(**eval_payload)
        await eval_service.create_evaluation(session, request)

        evals, total = await eval_service.list_evaluations(
            session,
            schema_id=eval_payload["schema_id"],
            status="pending",
        )
        assert total >= 1
        # Every result should match both filters
        for e in evals:
            assert e.schema_id == eval_payload["schema_id"]
            assert e.status == EvalStatus.PENDING

    async def test_list_respects_pagination(
        self,
        session: AsyncSession,
        eval_service: EvalService,
        eval_payload: dict[str, Any],
    ) -> None:
        """Pagination should limit results but return correct total."""
        # Create 3 evaluations
        for _ in range(3):
            payload = {**eval_payload, "model_id": str(uuid4())}
            request = EvalCreateRequest.model_validate(payload)
            await eval_service.create_evaluation(session, request)

        evals, total = await eval_service.list_evaluations(session, limit=2, offset=0)
        assert total >= 3
        assert len(evals) == 2

        evals2, total2 = await eval_service.list_evaluations(session, limit=2, offset=2)
        assert total2 == total
        assert len(evals2) >= 1

        # No overlap
        page1_ids = {e.id for e in evals}
        page2_ids = {e.id for e in evals2}
        assert page1_ids.isdisjoint(page2_ids)

    async def test_list_ordered_by_created_at_desc(
        self,
        session: AsyncSession,
        eval_service: EvalService,
        eval_payload: dict[str, Any],
    ) -> None:
        """List results should be ordered newest first."""
        ids = []
        for _ in range(3):
            payload = {**eval_payload, "model_id": str(uuid4())}
            request = EvalCreateRequest.model_validate(payload)
            created = await eval_service.create_evaluation(session, request)
            ids.append(created.id)

        evals, _ = await eval_service.list_evaluations(session)
        # Most recently created should come first
        if len(evals) >= 2:
            for i in range(len(evals) - 1):
                assert evals[i].created_at >= evals[i + 1].created_at


class TestRunEvaluationExtended:
    """Extended tests for run_evaluation edge cases."""

    async def test_run_evaluation_nonexistent(
        self,
        session: AsyncSession,
        eval_service: EvalService,
    ) -> None:
        """Should return early when evaluation doesn't exist, without touching anything.

        Verifies the early-return path: no row is created, none of the external
        clients are called, and the operation completes silently (logged-only).
        """
        from sqlalchemy import func, select

        from models.db import EvaluationORM

        bogus_id = uuid4()

        before_count = await session.execute(select(func.count()).select_from(EvaluationORM))
        rows_before = before_count.scalar_one()

        await eval_service.run_evaluation(session, bogus_id)

        after_count = await session.execute(select(func.count()).select_from(EvaluationORM))
        rows_after = after_count.scalar_one()
        assert rows_after == rows_before

        # External services must not have been called -- the function returns
        # before reaching any of them.
        cast(Any, eval_service.collector_client).get_validated_samples.assert_not_called()
        cast(Any, eval_service.inference_client).complete.assert_not_called()
        cast(Any, eval_service.registry_client).report_evaluation.assert_not_called()

    async def test_registry_notification_failure_on_auto_approve(
        self,
        session: AsyncSession,
        eval_service: EvalService,
        eval_payload: dict[str, Any],
        ground_truth_samples: list[dict[str, Any]],
    ) -> None:
        """Auto-approve must persist even when the registry call raises httpx.HTTPError.

        Exercises the narrowed httpx.HTTPError branch in eval_service.run_evaluation
        where the auto-approve notification fails after the row has already been
        committed as APPROVED.
        """
        import httpx

        cast(Any, eval_service.collector_client).get_validated_samples = AsyncMock(
            return_value=ground_truth_samples
        )

        # Return exact matches for all samples
        async def mock_complete(
            endpoint: str, messages: list, grammar: str | None = None, **kw: Any
        ) -> dict[str, Any]:
            for sample in ground_truth_samples:
                if sample["messages"] == messages:
                    return {"choices": [{"message": {"content": sample["response_content"]}}]}
            return {"choices": [{"message": {"content": '{"entities": []}'}}]}

        cast(Any, eval_service.inference_client).complete = AsyncMock(side_effect=mock_complete)

        # Make registry fail with a real httpx exception (subclass of HTTPError)
        report_mock = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "Service Unavailable",
                request=httpx.Request("PATCH", "http://registry/api/models/x/status"),
                response=httpx.Response(503),
            )
        )
        cast(Any, eval_service.registry_client).report_evaluation = report_mock

        request = EvalCreateRequest(**eval_payload)
        evaluation = await eval_service.create_evaluation(session, request)
        await eval_service.run_evaluation(session, evaluation.id)

        result = await eval_service.get(session, evaluation.id)
        assert result is not None
        # Should still be approved despite registry failure
        assert result.status == EvalStatus.APPROVED
        assert result.auto_approved is True
        assert result.metrics is not None
        # Composite must clear the threshold (otherwise we'd be on the hold path).
        assert result.metrics.composite_score >= result.threshold
        assert result.metrics.total_samples == len(ground_truth_samples)
        assert result.error_message is None
        assert result.completed_at is not None
        # Registry was actually called (and raised) before we declared success
        report_mock.assert_called_once()
        call_kwargs = report_mock.call_args.kwargs
        assert call_kwargs["model_id"] == evaluation.model_id
        assert call_kwargs["approved"] is True

    async def test_registry_notification_failure_on_hold(
        self,
        session: AsyncSession,
        eval_service: EvalService,
        eval_payload: dict[str, Any],
        ground_truth_samples: list[dict[str, Any]],
    ) -> None:
        """Hold-for-review must persist even when registry call raises httpx.HTTPError.

        Exercises the narrowed httpx.HTTPError branch in the COMPLETED (held)
        gate path -- a registry hiccup must not roll back the held row.
        """
        import httpx

        cast(Any, eval_service.collector_client).get_validated_samples = AsyncMock(
            return_value=ground_truth_samples
        )
        cast(Any, eval_service.inference_client).complete = AsyncMock(
            return_value={
                "choices": [
                    {"message": {"content": '{"entities": [{"text": "Wrong", "type": "WRONG"}]}'}}
                ]
            }
        )

        # Make registry fail with a network-side httpx exception
        report_mock = AsyncMock(side_effect=httpx.ConnectError("Registry down"))
        cast(Any, eval_service.registry_client).report_evaluation = report_mock

        request = EvalCreateRequest(**eval_payload)
        evaluation = await eval_service.create_evaluation(session, request)
        await eval_service.run_evaluation(session, evaluation.id)

        result = await eval_service.get(session, evaluation.id)
        assert result is not None
        assert result.status == EvalStatus.COMPLETED
        assert result.auto_approved is False
        assert result.metrics is not None
        # Composite must be BELOW threshold (otherwise we'd be on the approve path).
        assert result.metrics.composite_score < result.threshold
        assert result.completed_at is not None
        # Registry was called once with approved=False before raising
        report_mock.assert_called_once()
        call_kwargs = report_mock.call_args.kwargs
        assert call_kwargs["model_id"] == evaluation.model_id
        assert call_kwargs["approved"] is False


class TestApproveRejectExtended:
    """Extended tests for approve/reject edge cases."""

    async def test_approve_nonexistent_returns_none(
        self,
        session: AsyncSession,
        eval_service: EvalService,
    ) -> None:
        """approve() should return None for non-existent eval."""
        result = await eval_service.approve(session, uuid4(), review_notes="test")
        assert result is None

    async def test_reject_nonexistent_returns_none(
        self,
        session: AsyncSession,
        eval_service: EvalService,
    ) -> None:
        """reject() should return None for non-existent eval."""
        result = await eval_service.reject(session, uuid4(), review_notes="test")
        assert result is None

    async def test_approve_wrong_state_pending(
        self,
        session: AsyncSession,
        eval_service: EvalService,
        eval_payload: dict[str, Any],
    ) -> None:
        """approve() should return None for pending evaluation, leaving DB unchanged."""
        request = EvalCreateRequest(**eval_payload)
        evaluation = await eval_service.create_evaluation(session, request)

        result = await eval_service.approve(session, evaluation.id, review_notes="test")
        assert result is None

        # Verify DB state unchanged
        from sqlalchemy import select

        from models.db import EvaluationORM

        db_result = await session.execute(
            select(EvaluationORM).where(EvaluationORM.id == evaluation.id)
        )
        row = db_result.scalar_one()
        assert row.status == "pending"

    async def test_approve_wrong_state_running(
        self,
        session: AsyncSession,
        eval_service: EvalService,
        eval_payload: dict[str, Any],
    ) -> None:
        """approve() should return None for running evaluation."""
        request = EvalCreateRequest(**eval_payload)
        evaluation = await eval_service.create_evaluation(session, request)
        await eval_service._update_status(session, evaluation.id, EvalStatus.RUNNING)

        result = await eval_service.approve(session, evaluation.id, review_notes="test")
        assert result is None

    async def test_approve_wrong_state_failed(
        self,
        session: AsyncSession,
        eval_service: EvalService,
        eval_payload: dict[str, Any],
    ) -> None:
        """approve() should return None for failed evaluation."""
        request = EvalCreateRequest(**eval_payload)
        evaluation = await eval_service.create_evaluation(session, request)
        await eval_service._update_status(
            session,
            evaluation.id,
            EvalStatus.FAILED,
            error_message="test error",
        )

        result = await eval_service.approve(session, evaluation.id, review_notes="test")
        assert result is None

    async def test_approve_wrong_state_already_approved(
        self,
        session: AsyncSession,
        eval_service: EvalService,
        eval_payload: dict[str, Any],
    ) -> None:
        """approve() should return None for already-approved evaluation."""
        request = EvalCreateRequest(**eval_payload)
        evaluation = await eval_service.create_evaluation(session, request)
        await eval_service._update_status(
            session,
            evaluation.id,
            EvalStatus.APPROVED,
            auto_approved=True,
        )

        result = await eval_service.approve(session, evaluation.id, review_notes="test")
        assert result is None

    async def test_reject_wrong_state(
        self,
        session: AsyncSession,
        eval_service: EvalService,
        eval_payload: dict[str, Any],
    ) -> None:
        """reject() should return None for pending evaluation."""
        request = EvalCreateRequest(**eval_payload)
        evaluation = await eval_service.create_evaluation(session, request)

        result = await eval_service.reject(session, evaluation.id, review_notes="test")
        assert result is None

    async def test_reject_wrong_state_already_rejected(
        self,
        session: AsyncSession,
        eval_service: EvalService,
        eval_payload: dict[str, Any],
    ) -> None:
        """reject() should return None for already-rejected evaluation."""
        request = EvalCreateRequest(**eval_payload)
        evaluation = await eval_service.create_evaluation(session, request)
        await eval_service._update_status(session, evaluation.id, EvalStatus.COMPLETED)
        await eval_service._update_status(
            session,
            evaluation.id,
            EvalStatus.REJECTED,
            review_notes="already rejected",
        )

        result = await eval_service.reject(session, evaluation.id, review_notes="double reject")
        assert result is None

    async def test_approve_with_registry_failure(
        self,
        session: AsyncSession,
        eval_service: EvalService,
        eval_payload: dict[str, Any],
    ) -> None:
        """Manual approve must persist even when the registry call raises httpx.HTTPError.

        Exercises the narrowed httpx.HTTPError branch in EvalService.approve.
        """
        import httpx

        request = EvalCreateRequest(**eval_payload)
        evaluation = await eval_service.create_evaluation(session, request)

        # Set to COMPLETED with metrics
        metrics = EvalMetrics(
            json_parse_rate=1.0,
            schema_valid_rate=0.8,
            exact_match_rate=0.6,
            avg_latency_ms=150.0,
            p95_latency_ms=250.0,
            composite_score=0.73,
            total_samples=50,
            passed_samples=30,
            failed_samples=20,
        )
        await eval_service._update_status(
            session,
            evaluation.id,
            EvalStatus.COMPLETED,
            metrics=metrics,
        )

        # Registry returns 500 -- a real httpx.HTTPStatusError
        report_mock = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "Internal Server Error",
                request=httpx.Request("PATCH", "http://registry/api/models/x/status"),
                response=httpx.Response(500),
            )
        )
        cast(Any, eval_service.registry_client).report_evaluation = report_mock

        result = await eval_service.approve(
            session, evaluation.id, review_notes="Approved despite registry issue"
        )
        assert result is not None
        assert result.status == EvalStatus.APPROVED
        assert result.review_notes == "Approved despite registry issue"
        assert result.metrics is not None
        assert result.metrics.composite_score == 0.73
        # Registry call was actually attempted with the persisted metrics
        report_mock.assert_called_once()
        call_kwargs = report_mock.call_args.kwargs
        assert call_kwargs["model_id"] == evaluation.model_id
        assert call_kwargs["approved"] is True
        assert call_kwargs["metrics"].composite_score == 0.73

        # Verify DB state
        from sqlalchemy import select

        from models.db import EvaluationORM

        db_result = await session.execute(
            select(EvaluationORM).where(EvaluationORM.id == evaluation.id)
        )
        row = db_result.scalar_one()
        assert row.status == "approved"
        assert row.review_notes == "Approved despite registry issue"


class TestUpdateStatusExtended:
    """Tests for _update_status edge cases."""

    async def test_update_status_nonexistent(
        self,
        session: AsyncSession,
        eval_service: EvalService,
    ) -> None:
        """_update_status should handle non-existent eval gracefully."""
        # Should not raise
        await eval_service._update_status(session, uuid4(), EvalStatus.RUNNING)


class TestValidationHelpers:
    """Tests for static helper methods."""

    def test_try_parse_json_valid(self) -> None:
        """Should return True for valid JSON."""
        assert EvalService._try_parse_json('{"key": "value"}') is True

    def test_try_parse_json_invalid(self) -> None:
        """Should return False for invalid JSON."""
        assert EvalService._try_parse_json("not json") is False

    def test_try_parse_json_none_type(self) -> None:
        """Should return False for None."""
        assert EvalService._try_parse_json(None) is False

    def test_validate_schema_valid(self) -> None:
        """Should return True for valid JSON matching schema."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        assert EvalService._validate_schema('{"name": "test"}', schema) is True

    def test_validate_schema_invalid_json(self) -> None:
        """Should return False for invalid JSON."""
        schema = {"type": "object"}
        assert EvalService._validate_schema("not json", schema) is False

    def test_validate_schema_schema_mismatch(self) -> None:
        """Should return False when JSON doesn't match schema."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        assert EvalService._validate_schema('{"age": 25}', schema) is False


class TestScoringProfileFallbacks:
    """Tests for scoring_profile fallbacks (invalid value + missing attribute)."""

    async def test_invalid_scoring_profile_falls_back_and_warns(
        self,
        session: AsyncSession,
        eval_service: EvalService,
        eval_payload: dict[str, Any],
        ground_truth_samples: list[dict[str, Any]],
    ) -> None:
        """If an evaluation row stores an unknown scoring_profile, run_evaluation
        should use the default weights AND emit an invalid_scoring_profile_fallback
        warning with the requested value."""
        import structlog.testing

        from models.db import EvaluationORM

        cast(Any, eval_service.collector_client).get_validated_samples = AsyncMock(
            return_value=ground_truth_samples
        )
        cast(Any, eval_service.inference_client).complete = AsyncMock(
            return_value={"choices": [{"message": {"content": '{"entities": []}'}}]}
        )

        # Create evaluation then stamp scoring_profile to an invalid value
        # directly on the ORM row (bypassing Pydantic validation).
        request = EvalCreateRequest(**eval_payload)
        evaluation = await eval_service.create_evaluation(session, request)
        from sqlalchemy import select

        row_result = await session.execute(
            select(EvaluationORM).where(EvaluationORM.id == evaluation.id)
        )
        row = row_result.scalar_one()
        row.scoring_profile = "bogus-profile"
        await session.commit()

        with structlog.testing.capture_logs() as captured:
            await eval_service.run_evaluation(session, evaluation.id)

        result = await eval_service.get(session, evaluation.id)
        assert result is not None
        assert result.metrics is not None
        # Fallback path used -- invalid profile kept in row, but metrics
        # computed with default weights (0.15/0.35/0.50).
        assert result.scoring_profile == "bogus-profile"
        expected_composite = (
            0.15 * result.metrics.json_parse_rate
            + 0.35 * result.metrics.schema_valid_rate
            + 0.50 * result.metrics.exact_match_rate
        )
        assert abs(result.metrics.composite_score - expected_composite) < 1e-9

        assert any(
            log["event"] == "invalid_scoring_profile_fallback"
            and log.get("requested") == "bogus-profile"
            and log.get("fallback") == "default"
            for log in captured
        ), f"expected invalid_scoring_profile_fallback log; got: {captured}"

    def test_row_to_domain_missing_scoring_profile_attribute_warns(self) -> None:
        """_row_to_domain should fall back to 'default' AND log a warning when
        the ORM row has no scoring_profile attribute at all (legacy migration)."""
        from datetime import UTC, datetime

        import structlog.testing

        from services.eval_service import _row_to_domain
        from util._uuid7 import uuid7

        class LegacyRow:
            """Fake legacy row missing scoring_profile entirely."""

            def __init__(self) -> None:
                self.id = uuid7()
                self.model_id = uuid4()
                self.schema_id = "ks_legacy"
                self.status = "pending"
                self.endpoint = "http://legacy:8080"
                self.metrics = None
                self.threshold = 0.85
                self.max_samples = 100
                self.auto_approved = False
                self.review_notes = None
                self.sample_results = []
                self.json_schema = None
                self.gbnf_grammar = None
                self.created_at = datetime.now(UTC)
                self.started_at = None
                self.completed_at = None
                self.error_message = None

        legacy = LegacyRow()
        assert not hasattr(legacy, "scoring_profile")

        with structlog.testing.capture_logs() as captured:
            domain = _row_to_domain(cast(Any, legacy))

        assert domain.scoring_profile == "default"
        assert any(log["event"] == "scoring_profile_column_missing_on_row" for log in captured), (
            f"expected scoring_profile_column_missing_on_row log; got: {captured}"
        )

    async def test_row_to_domain_empty_scoring_profile_uses_default_silently(
        self,
        session: AsyncSession,
        eval_service: EvalService,
        eval_payload: dict[str, Any],
    ) -> None:
        """Rows with scoring_profile='' (empty string, column is NOT NULL)
        should silently fall back to 'default' via the ``or "default"`` branch
        without emitting the missing-attribute warning."""
        import structlog.testing

        from models.db import EvaluationORM

        request = EvalCreateRequest(**eval_payload)
        evaluation = await eval_service.create_evaluation(session, request)
        from sqlalchemy import select

        row_result = await session.execute(
            select(EvaluationORM).where(EvaluationORM.id == evaluation.id)
        )
        row = row_result.scalar_one()
        row.scoring_profile = ""  # empty string triggers `or "default"` fallback
        await session.commit()

        with structlog.testing.capture_logs() as captured:
            stored = await eval_service.get(session, evaluation.id)

        assert stored is not None
        assert stored.scoring_profile == "default"
        # No warning should be emitted -- empty string is a supported state;
        # only the completely-missing-attribute case warns.
        assert not any(log["event"] == "scoring_profile_column_missing_on_row" for log in captured)


class TestInferenceFailurePerSample:
    """Tests for per-sample inference failure handling."""

    async def test_inference_failure_recorded_per_sample(
        self,
        session: AsyncSession,
        eval_service: EvalService,
        eval_payload: dict[str, Any],
        ground_truth_samples: list[dict[str, Any]],
    ) -> None:
        """If inference fails on exactly one of three samples, the pipeline
        must complete and report json_parse_rate == 2/3 with the failing
        sample_result containing the error string."""
        import httpx

        cast(Any, eval_service.collector_client).get_validated_samples = AsyncMock(
            return_value=ground_truth_samples
        )

        call_count = 0
        err_message = "simulated read timeout"

        async def flaky_complete(
            endpoint: str, messages: list, grammar: str | None = None, **kw: Any
        ) -> dict:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise httpx.ReadTimeout(err_message)
            # Return a payload that scores exact_match for the matching sample
            for sample in ground_truth_samples:
                if sample["messages"] == messages:
                    return {"choices": [{"message": {"content": sample["response_content"]}}]}
            return {"choices": [{"message": {"content": '{"entities": []}'}}]}

        cast(Any, eval_service.inference_client).complete = AsyncMock(side_effect=flaky_complete)

        request = EvalCreateRequest(**eval_payload)
        evaluation = await eval_service.create_evaluation(session, request)
        await eval_service.run_evaluation(session, evaluation.id)

        result = await eval_service.get(session, evaluation.id)
        assert result is not None
        assert result.metrics is not None
        # Two of three samples parsed + schema + exact match; one failed
        assert result.metrics.total_samples == 3
        assert result.metrics.json_parse_rate == 2 / 3
        assert result.metrics.schema_valid_rate == 2 / 3
        assert result.metrics.exact_match_rate == 2 / 3
        # The failing sample_result must contain the error message
        failing = [s for s in result.sample_results if s.get("error")]
        assert len(failing) == 1
        assert err_message in failing[0]["error"]

    async def test_inference_returns_malformed_shape_keyerror(
        self,
        session: AsyncSession,
        eval_service: EvalService,
        eval_payload: dict[str, Any],
        ground_truth_samples: list[dict[str, Any]],
    ) -> None:
        """Inference returning a dict missing the 'choices' key must hit the
        narrowed KeyError branch in _evaluate_sample, NOT crash the pipeline."""
        cast(Any, eval_service.collector_client).get_validated_samples = AsyncMock(
            return_value=ground_truth_samples
        )
        # Missing "choices" key entirely -- raises KeyError on response["choices"]
        cast(Any, eval_service.inference_client).complete = AsyncMock(
            return_value={"error": "model not loaded"}
        )

        request = EvalCreateRequest(**eval_payload)
        evaluation = await eval_service.create_evaluation(session, request)
        await eval_service.run_evaluation(session, evaluation.id)

        result = await eval_service.get(session, evaluation.id)
        assert result is not None
        # Pipeline must complete (not crash) even though every sample failed
        assert result.status in (EvalStatus.COMPLETED, EvalStatus.FAILED)
        # Every sample is recorded as a failed inference with a KeyError message
        assert len(result.sample_results) == len(ground_truth_samples)
        for sr in result.sample_results:
            assert sr["json_valid"] is False
            assert sr["schema_valid"] is False
            assert sr["exact_match"] is False
            assert "error" in sr
            assert "choices" in sr["error"]  # KeyError('choices') message

    async def test_inference_returns_empty_choices_indexerror(
        self,
        session: AsyncSession,
        eval_service: EvalService,
        eval_payload: dict[str, Any],
        ground_truth_samples: list[dict[str, Any]],
    ) -> None:
        """Inference returning {'choices': []} must hit the narrowed IndexError
        branch in _evaluate_sample (response['choices'][0] is out of range)."""
        cast(Any, eval_service.collector_client).get_validated_samples = AsyncMock(
            return_value=ground_truth_samples
        )
        cast(Any, eval_service.inference_client).complete = AsyncMock(return_value={"choices": []})

        request = EvalCreateRequest(**eval_payload)
        evaluation = await eval_service.create_evaluation(session, request)
        await eval_service.run_evaluation(session, evaluation.id)

        result = await eval_service.get(session, evaluation.id)
        assert result is not None
        assert len(result.sample_results) == len(ground_truth_samples)
        for sr in result.sample_results:
            assert sr["json_valid"] is False
            assert "error" in sr
            assert "index" in sr["error"].lower() or "range" in sr["error"].lower()

    async def test_inference_returns_none_typeerror(
        self,
        session: AsyncSession,
        eval_service: EvalService,
        eval_payload: dict[str, Any],
        ground_truth_samples: list[dict[str, Any]],
    ) -> None:
        """Inference returning None must hit the narrowed TypeError branch in
        _evaluate_sample (None is not subscriptable)."""
        cast(Any, eval_service.collector_client).get_validated_samples = AsyncMock(
            return_value=ground_truth_samples
        )
        cast(Any, eval_service.inference_client).complete = AsyncMock(return_value=None)

        request = EvalCreateRequest(**eval_payload)
        evaluation = await eval_service.create_evaluation(session, request)
        await eval_service.run_evaluation(session, evaluation.id)

        result = await eval_service.get(session, evaluation.id)
        assert result is not None
        assert len(result.sample_results) == len(ground_truth_samples)
        for sr in result.sample_results:
            assert sr["json_valid"] is False
            assert "error" in sr
            # TypeError raised by None["choices"]
            assert "subscriptable" in sr["error"] or "NoneType" in sr["error"]

    async def test_inference_http_status_error_per_sample(
        self,
        session: AsyncSession,
        eval_service: EvalService,
        eval_payload: dict[str, Any],
        ground_truth_samples: list[dict[str, Any]],
    ) -> None:
        """Inference raising httpx.HTTPStatusError must hit the narrowed
        httpx.HTTPError branch and be recorded per-sample."""
        import httpx

        cast(Any, eval_service.collector_client).get_validated_samples = AsyncMock(
            return_value=ground_truth_samples
        )
        cast(Any, eval_service.inference_client).complete = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "Bad Gateway",
                request=httpx.Request("POST", "http://pi-node:8080/v1/chat/completions"),
                response=httpx.Response(502),
            )
        )

        request = EvalCreateRequest(**eval_payload)
        evaluation = await eval_service.create_evaluation(session, request)
        await eval_service.run_evaluation(session, evaluation.id)

        result = await eval_service.get(session, evaluation.id)
        assert result is not None
        assert len(result.sample_results) == len(ground_truth_samples)
        # All samples failed -> metrics should be all zero
        assert result.metrics is not None
        assert result.metrics.json_parse_rate == 0.0
        assert result.metrics.schema_valid_rate == 0.0
        assert result.metrics.exact_match_rate == 0.0
        for sr in result.sample_results:
            assert "error" in sr
            assert "Bad Gateway" in sr["error"]


class TestRealHttpWireShape:
    """End-to-end tests of run_evaluation against real httpx clients backed by
    MockTransport. These exercise the actual JSON wire shape sent to the Pi
    inference endpoint and the registry, not just the python-level method
    contract -- so we catch payload regressions that AsyncMock cannot.
    """

    async def test_run_evaluation_with_real_inference_transport(
        self,
        session: AsyncSession,
        eval_service: EvalService,
        eval_payload: dict[str, Any],
        ground_truth_samples: list[dict[str, Any]],
    ) -> None:
        """Replace eval_service.inference_client with a REAL InferenceClient
        backed by httpx.MockTransport so we test the full chat-completion
        request shape end-to-end."""
        import json as _json

        import httpx

        from services.inference_client import InferenceClient

        captured_requests: list[httpx.Request] = []

        # Index ground truth by content of the LAST user message so we can
        # echo back the matching response.
        by_user_content: dict[str, str] = {}
        for sample in ground_truth_samples:
            user_msg = next(
                (m for m in sample["messages"] if m["role"] == "user"),
                None,
            )
            assert user_msg is not None
            by_user_content[user_msg["content"]] = sample["response_content"]

        def handler(request: httpx.Request) -> httpx.Response:
            captured_requests.append(request)
            assert request.method == "POST"
            assert request.url.path == "/v1/chat/completions"
            payload = _json.loads(request.content)
            # Wire-shape contract:
            assert payload["stream"] is False
            assert payload["temperature"] == 0.0
            assert payload["max_tokens"] > 0
            assert isinstance(payload["messages"], list)
            assert payload["messages"][-1]["role"] == "user"
            content_for_user = payload["messages"][-1]["content"]
            response_text = by_user_content.get(content_for_user, '{"entities": []}')
            return httpx.Response(
                200,
                content=_json.dumps(
                    {"choices": [{"message": {"content": response_text}}]}
                ).encode(),
                headers={"content-type": "application/json"},
            )

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http:
            real_inference = InferenceClient(http_client=http)
            eval_service.inference_client = real_inference

            cast(Any, eval_service.collector_client).get_validated_samples = AsyncMock(
                return_value=ground_truth_samples
            )

            request = EvalCreateRequest(**eval_payload)
            evaluation = await eval_service.create_evaluation(session, request)
            await eval_service.run_evaluation(session, evaluation.id)

        result = await eval_service.get(session, evaluation.id)
        assert result is not None
        assert result.metrics is not None
        # All three samples must have round-tripped through the mock transport
        assert len(captured_requests) == len(ground_truth_samples)
        # Every sample matched exactly -> 100% rates -> auto-approved
        assert result.metrics.json_parse_rate == 1.0
        assert result.metrics.schema_valid_rate == 1.0
        assert result.metrics.exact_match_rate == 1.0
        assert result.status == EvalStatus.APPROVED
        assert result.auto_approved is True

    async def test_run_evaluation_with_real_inference_returning_5xx(
        self,
        session: AsyncSession,
        eval_service: EvalService,
        eval_payload: dict[str, Any],
        ground_truth_samples: list[dict[str, Any]],
    ) -> None:
        """A real InferenceClient that gets a 5xx from the Pi must surface as
        an httpx.HTTPStatusError caught by the per-sample handler."""
        import httpx

        from services.inference_client import InferenceClient

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, content=b'{"error":"unavailable"}')

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http:
            eval_service.inference_client = InferenceClient(http_client=http)
            cast(Any, eval_service.collector_client).get_validated_samples = AsyncMock(
                return_value=ground_truth_samples
            )

            request = EvalCreateRequest(**eval_payload)
            evaluation = await eval_service.create_evaluation(session, request)
            await eval_service.run_evaluation(session, evaluation.id)

        result = await eval_service.get(session, evaluation.id)
        assert result is not None
        assert result.metrics is not None
        # Every sample saw a 503 -> all zero rates -> hold for review
        assert result.metrics.json_parse_rate == 0.0
        assert result.metrics.exact_match_rate == 0.0
        assert result.status == EvalStatus.COMPLETED
        for sr in result.sample_results:
            assert "error" in sr
            assert "503" in sr["error"] or "Server error" in sr["error"]
