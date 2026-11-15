"""Edge case tests to improve coverage toward 99.7%+.

Targets the remaining uncovered paths:
- EvaluationORM.__repr__ (models/db.py:53)
- EvalService.approve when evaluation.metrics is None (eval_service.py branch 335->349)
- add_trace_context when ctx.is_valid is False (logging_config.py branch 38->41)
- LivenessResponse default value
- _validate_schema with a SchemaError (bad schema dict)
"""

from typing import Any, cast
from unittest.mock import MagicMock, patch
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from models.db import EvaluationORM
from models.domain import EvalCreateRequest, EvalStatus, LivenessResponse
from services.eval_service import EvalService


class TestEvaluationORMRepr:
    """Cover EvaluationORM.__repr__ (models/db.py line 53)."""

    def test_repr_format(self) -> None:
        """__repr__ should include id, model_id, schema_id, and status."""
        row = EvaluationORM()
        row.id = uuid4()
        row.model_id = uuid4()
        row.schema_id = "ks_EntitySpotterAgent"
        row.status = "pending"

        result = repr(row)

        assert "Evaluation" in result
        assert str(row.id) in result
        assert str(row.model_id) in result
        assert "ks_EntitySpotterAgent" in result
        assert "pending" in result


class TestApproveWithNoneMetrics:
    """Cover the branch in approve() where evaluation.metrics is None.

    When metrics is None, the registry notification should be skipped entirely.
    This covers the eval_service.py branch 335->349 (the else path of
    ``if evaluation.metrics is not None``).
    """

    async def test_approve_completed_no_metrics_skips_registry(
        self,
        session: AsyncSession,
        eval_service: EvalService,
        eval_payload: dict[str, Any],
    ) -> None:
        """Approve a completed evaluation that has no metrics -- registry should not be called."""
        request = EvalCreateRequest(**eval_payload)
        evaluation = await eval_service.create_evaluation(session, request)

        # Set to COMPLETED but without metrics (unusual but possible edge case)
        await eval_service._update_status(
            session,
            evaluation.id,
            EvalStatus.COMPLETED,
        )

        # Reset the mock to ensure we can verify no calls
        cast(Any, eval_service.registry_client).report_evaluation.reset_mock()

        result = await eval_service.approve(
            session, evaluation.id, review_notes="Approved without metrics"
        )

        assert result is not None
        assert result.status == EvalStatus.APPROVED
        assert result.review_notes == "Approved without metrics"

        # Registry should NOT have been called since metrics is None
        cast(Any, eval_service.registry_client).report_evaluation.assert_not_called()


class TestTraceContextInvalidSpan:
    """Cover the logging_config.py branch where ctx.is_valid is False."""

    def test_no_trace_ids_when_context_invalid(self) -> None:
        """Should not inject trace/span IDs when span context is not valid."""
        from logging_config import add_trace_context

        mock_ctx = MagicMock()
        mock_ctx.is_valid = False

        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        mock_span.get_span_context.return_value = mock_ctx

        mock_trace = MagicMock()
        mock_trace.get_current_span.return_value = mock_span

        with (
            patch("logging_config.OTEL_AVAILABLE", True),
            patch("logging_config.trace", mock_trace),
        ):
            event_dict: dict[str, Any] = {"event": "test"}
            result = add_trace_context(None, "info", event_dict)
            assert "trace_id" not in result
            assert "span_id" not in result


class TestLivenessResponseModel:
    """Cover LivenessResponse Pydantic model edge cases."""

    def test_default_status(self) -> None:
        """LivenessResponse should default to status='alive'."""
        resp = LivenessResponse()
        assert resp.status == "alive"

    def test_model_dump(self) -> None:
        """LivenessResponse should dump correctly."""
        resp = LivenessResponse()
        data = resp.model_dump()
        assert data == {"status": "alive"}

    def test_custom_status(self) -> None:
        """LivenessResponse should accept custom status."""
        resp = LivenessResponse(status="custom")
        assert resp.status == "custom"


class TestValidateSchemaError:
    """Cover _validate_schema with a truly invalid schema (SchemaError branch)."""

    def test_validate_schema_with_bad_schema(self) -> None:
        """Should return False when schema itself is invalid (SchemaError)."""
        # A schema with an invalid type value triggers SchemaError
        bad_schema: dict[str, Any] = {"type": "not_a_real_type"}
        result = EvalService._validate_schema('{"key": "value"}', bad_schema)
        assert result is False
