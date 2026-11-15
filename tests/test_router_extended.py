"""Extended tests for the evaluations router.

Covers:
- _get_eval_service factory function
- approve/reject direct-handler conflict cases
- background task in create_evaluation (exception, missing session factory)

Router-level list tests live in test_evaluations_api.py -- do not duplicate.
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import HTTPException
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from models.domain import EvalCreateRequest, EvalMetrics, EvalStatus
from routers.evaluations import (
    approve_evaluation,
    create_evaluation,
    get_evaluation,
    reject_evaluation,
)
from services.eval_service import EvalService


class TestGetEvalServiceFactory:
    """Tests for the _get_eval_service dependency."""

    def test_get_eval_service_creates_service(self) -> None:
        """_get_eval_service should create an EvalService with clients from settings."""
        with patch("routers.evaluations.get_settings") as mock_settings:
            settings = MagicMock()
            settings.data_collector_url = "http://collector:8151"
            settings.model_registry_url = "http://registry:8155"
            settings.inference_timeout = 30.0
            mock_settings.return_value = settings

            from routers.evaluations import _get_eval_service

            service = _get_eval_service()

            assert isinstance(service, EvalService)
            assert service.collector_client is not None
            assert service.inference_client is not None
            assert service.metrics_service is not None
            assert service.registry_client is not None


class TestApproveRejectRouterExtended:
    """Extended router tests for approve/reject flows."""

    async def test_approve_already_approved_returns_conflict(
        self,
        client: AsyncClient,
        eval_payload: dict[str, Any],
        session: AsyncSession,
        eval_service: EvalService,
    ) -> None:
        """Cannot approve an evaluation that is already approved."""
        request = EvalCreateRequest(**eval_payload)
        evaluation = await eval_service.create_evaluation(session, request)

        # Set to approved
        await eval_service._update_status(
            session,
            evaluation.id,
            EvalStatus.APPROVED,
            auto_approved=True,
            metrics=EvalMetrics(
                json_parse_rate=1.0,
                schema_valid_rate=1.0,
                exact_match_rate=1.0,
                avg_latency_ms=100.0,
                p95_latency_ms=200.0,
                composite_score=1.0,
                total_samples=10,
                passed_samples=10,
                failed_samples=0,
            ),
        )

        response = await client.post(
            f"/api/evaluations/{evaluation.id}/approve",
            json={"review_notes": "Double approve"},
        )
        assert response.status_code == 409

    async def test_reject_already_rejected_returns_conflict(
        self,
        client: AsyncClient,
        eval_payload: dict[str, Any],
        session: AsyncSession,
        eval_service: EvalService,
    ) -> None:
        """Cannot reject an evaluation that is already rejected."""
        request = EvalCreateRequest(**eval_payload)
        evaluation = await eval_service.create_evaluation(session, request)

        # Set to COMPLETED first, then reject
        await eval_service._update_status(session, evaluation.id, EvalStatus.COMPLETED)
        await eval_service._update_status(
            session, evaluation.id, EvalStatus.REJECTED, review_notes="bad"
        )

        response = await client.post(
            f"/api/evaluations/{evaluation.id}/reject",
            json={"review_notes": "Double reject"},
        )
        assert response.status_code == 409

    async def test_approve_without_body(
        self,
        client: AsyncClient,
        eval_payload: dict[str, Any],
        session: AsyncSession,
        eval_service: EvalService,
    ) -> None:
        """Approve endpoint should work without a body (review_notes is optional)."""
        request = EvalCreateRequest(**eval_payload)
        evaluation = await eval_service.create_evaluation(session, request)

        await eval_service._update_status(
            session,
            evaluation.id,
            EvalStatus.COMPLETED,
            metrics=EvalMetrics(
                json_parse_rate=1.0,
                schema_valid_rate=0.8,
                exact_match_rate=0.6,
                avg_latency_ms=150.0,
                p95_latency_ms=250.0,
                composite_score=0.73,
                total_samples=50,
                passed_samples=30,
                failed_samples=20,
            ),
        )

        # Post with empty body (no review_notes)
        response = await client.post(f"/api/evaluations/{evaluation.id}/approve")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "approved"


class TestCreateEvaluationBackground:
    """Tests for the background task logic in create_evaluation (lines 69-87).

    Uses direct handler invocation to ensure coverage tracking of the
    inner _run_background coroutine and asyncio.create_task call.
    """

    async def test_background_task_calls_run_evaluation(
        self,
        session: AsyncSession,
        eval_payload: dict[str, Any],
        eval_service: EvalService,
        engine,
    ) -> None:
        """Background task should invoke eval_service.run_evaluation with a fresh session."""
        factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

        with patch.object(eval_service, "run_evaluation", new_callable=AsyncMock) as mock_run:
            with patch("db._session_factory", factory):
                payload = EvalCreateRequest(**eval_payload)
                evaluation = await create_evaluation(
                    payload=payload, session=session, eval_service=eval_service
                )

                # Give the event loop time to execute the background task
                await asyncio.sleep(0.1)

            mock_run.assert_called_once()
            call_args = mock_run.call_args
            assert call_args[0][1] == evaluation.id

    async def test_background_task_no_session_factory(
        self,
        session: AsyncSession,
        eval_payload: dict[str, Any],
        eval_service: EvalService,
    ) -> None:
        """Background task should log error and leave evaluation PENDING when _session_factory is None."""
        import structlog.testing

        with patch("db._session_factory", None):
            payload = EvalCreateRequest(**eval_payload)
            with structlog.testing.capture_logs() as captured:
                evaluation = await create_evaluation(
                    payload=payload, session=session, eval_service=eval_service
                )
                # Let the background task execute — it should handle None gracefully
                await asyncio.sleep(0.1)

        assert evaluation.status == EvalStatus.PENDING
        # The error log should have been emitted with the eval id
        assert any(
            log["event"] == "cannot_run_evaluation_no_db"
            and log.get("eval_id") == str(evaluation.id)
            for log in captured
        ), f"expected cannot_run_evaluation_no_db log; got: {captured}"

        # Row in DB is still PENDING (not touched by failed background task)
        stored = await eval_service.get(session, evaluation.id)
        assert stored is not None
        assert stored.status == EvalStatus.PENDING

    async def test_background_task_logs_and_marks_failed_on_exception(
        self,
        session: AsyncSession,
        eval_payload: dict[str, Any],
        eval_service: EvalService,
        engine: Any,
    ) -> None:
        """If run_evaluation raises, the background task handler must log
        background_evaluation_failed AND mark the evaluation FAILED with the
        exception message in error_message."""
        import structlog.testing

        from routers.evaluations import _background_tasks

        factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

        async def _boom(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("simulated background crash")

        with patch.object(eval_service, "run_evaluation", side_effect=_boom):
            with patch("db._session_factory", factory):
                payload = EvalCreateRequest(**eval_payload)
                with structlog.testing.capture_logs() as captured:
                    evaluation = await create_evaluation(
                        payload=payload, session=session, eval_service=eval_service
                    )
                    # Drain all pending background tasks to completion so the
                    # except branch (logging + FAILED-update) runs before we assert.
                    pending = [t for t in list(_background_tasks) if not t.done()]
                    if pending:
                        await asyncio.gather(*pending, return_exceptions=True)

        # Background error log was emitted with the expected eval id
        assert any(
            log["event"] == "background_evaluation_failed"
            and log.get("eval_id") == str(evaluation.id)
            for log in captured
        ), f"expected background_evaluation_failed log; got: {captured}"

        # Evaluation row is now FAILED with the exception message preserved
        async with factory() as check_session:
            stored = await eval_service.get(check_session, evaluation.id)
        assert stored is not None, "evaluation row disappeared"
        assert stored.status == EvalStatus.FAILED
        assert stored.error_message is not None
        assert "simulated background crash" in stored.error_message
        assert stored.completed_at is not None


class TestGetEvaluationHandler:
    """Tests for the get_evaluation handler (both success and 404 paths)."""

    async def test_get_evaluation_returns_404(
        self,
        session: AsyncSession,
        eval_service: EvalService,
    ) -> None:
        """Should raise 404 HTTPException for nonexistent UUID."""
        fake_id = uuid4()
        with pytest.raises(HTTPException) as exc_info:
            await get_evaluation(eval_id=fake_id, session=session, eval_service=eval_service)
        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "Evaluation not found"

    async def test_get_evaluation_returns_evaluation(
        self,
        session: AsyncSession,
        eval_payload: dict[str, Any],
        eval_service: EvalService,
    ) -> None:
        """Should return Evaluation for a valid ID."""
        payload = EvalCreateRequest(**eval_payload)
        created = await eval_service.create_evaluation(session, payload)

        result = await get_evaluation(
            eval_id=created.id, session=session, eval_service=eval_service
        )
        assert result.id == created.id
        assert result.schema_id == created.schema_id
        assert result.status == EvalStatus.PENDING


class TestApproveRejectConflictDetail:
    """Tests for approve/reject 409 conflict response body detail messages.

    Calls router handler functions directly to verify the HTTPException detail
    includes the current status name.
    """

    async def test_approve_returns_conflict_detail(
        self,
        session: AsyncSession,
        eval_payload: dict[str, Any],
        eval_service: EvalService,
    ) -> None:
        """Approve on pending evaluation should raise 409 with status in detail."""
        payload = EvalCreateRequest(**eval_payload)
        evaluation = await eval_service.create_evaluation(session, payload)

        with pytest.raises(HTTPException) as exc_info:
            await approve_evaluation(
                eval_id=evaluation.id, session=session, eval_service=eval_service
            )
        assert exc_info.value.status_code == 409
        assert "pending" in exc_info.value.detail
        assert "approved" in exc_info.value.detail.lower()

    async def test_approve_nonexistent_returns_404(
        self,
        session: AsyncSession,
        eval_service: EvalService,
    ) -> None:
        """Approve on nonexistent evaluation should raise 404."""
        with pytest.raises(HTTPException) as exc_info:
            await approve_evaluation(eval_id=uuid4(), session=session, eval_service=eval_service)
        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "Evaluation not found"

    async def test_reject_returns_conflict_detail(
        self,
        session: AsyncSession,
        eval_payload: dict[str, Any],
        eval_service: EvalService,
    ) -> None:
        """Reject on pending evaluation should raise 409 with status in detail."""
        payload = EvalCreateRequest(**eval_payload)
        evaluation = await eval_service.create_evaluation(session, payload)

        from models.domain import EvalRejectRequest

        reject_payload = EvalRejectRequest(review_notes="Not good enough")

        with pytest.raises(HTTPException) as exc_info:
            await reject_evaluation(
                eval_id=evaluation.id,
                payload=reject_payload,
                session=session,
                eval_service=eval_service,
            )
        assert exc_info.value.status_code == 409
        assert "pending" in exc_info.value.detail
        assert "rejected" in exc_info.value.detail.lower()

    async def test_reject_nonexistent_returns_404(
        self,
        session: AsyncSession,
        eval_service: EvalService,
    ) -> None:
        """Reject on nonexistent evaluation should raise 404."""
        from models.domain import EvalRejectRequest

        reject_payload = EvalRejectRequest(review_notes="Does not exist")

        with pytest.raises(HTTPException) as exc_info:
            await reject_evaluation(
                eval_id=uuid4(),
                payload=reject_payload,
                session=session,
                eval_service=eval_service,
            )
        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "Evaluation not found"

    async def test_approve_completed_returns_evaluation(
        self,
        session: AsyncSession,
        eval_payload: dict[str, Any],
        eval_service: EvalService,
    ) -> None:
        """Approve on completed evaluation should return the approved Evaluation."""
        payload = EvalCreateRequest(**eval_payload)
        evaluation = await eval_service.create_evaluation(session, payload)

        # Transition to COMPLETED with metrics
        await eval_service._update_status(
            session,
            evaluation.id,
            EvalStatus.COMPLETED,
            metrics=EvalMetrics(
                json_parse_rate=1.0,
                schema_valid_rate=0.9,
                exact_match_rate=0.7,
                avg_latency_ms=120.0,
                p95_latency_ms=200.0,
                composite_score=0.8,
                total_samples=20,
                passed_samples=14,
                failed_samples=6,
            ),
        )

        from models.domain import EvalApproveRequest

        result = await approve_evaluation(
            eval_id=evaluation.id,
            payload=EvalApproveRequest(review_notes="Looks good"),
            session=session,
            eval_service=eval_service,
        )
        assert result.status == EvalStatus.APPROVED
        assert result.id == evaluation.id

    async def test_reject_completed_returns_evaluation(
        self,
        session: AsyncSession,
        eval_payload: dict[str, Any],
        eval_service: EvalService,
    ) -> None:
        """Reject on completed evaluation should return the rejected Evaluation."""
        payload = EvalCreateRequest(**eval_payload)
        evaluation = await eval_service.create_evaluation(session, payload)

        await eval_service._update_status(session, evaluation.id, EvalStatus.COMPLETED)

        from models.domain import EvalRejectRequest

        result = await reject_evaluation(
            eval_id=evaluation.id,
            payload=EvalRejectRequest(review_notes="Needs more work"),
            session=session,
            eval_service=eval_service,
        )
        assert result.status == EvalStatus.REJECTED
        assert result.id == evaluation.id
