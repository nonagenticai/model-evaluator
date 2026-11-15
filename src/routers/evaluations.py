"""Evaluation endpoints for creating, listing, and managing model evaluations."""

import asyncio
from datetime import UTC, datetime
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from db import get_session
from models.domain import (
    EvalApproveRequest,
    EvalCreateRequest,
    EvalListResponse,
    EvalRejectRequest,
    EvalStatus,
    Evaluation,
)
from services.collector_client import CollectorClient
from services.eval_service import EvalService
from services.inference_client import InferenceClient
from services.metrics_service import MetricsService
from services.registry_client import RegistryClient
from settings import get_settings

logger = structlog.stdlib.get_logger(__name__)

router = APIRouter(prefix="/api/evaluations", tags=["evaluations"])


_shared_eval_service: EvalService | None = None

# Strong references to background evaluation tasks so they are not garbage
# collected mid-flight. Completed tasks remove themselves via add_done_callback.
_background_tasks: set[asyncio.Task[None]] = set()


def set_eval_service(service: EvalService | None) -> None:
    """Set the shared EvalService instance (called from lifespan)."""
    global _shared_eval_service  # noqa: PLW0603
    _shared_eval_service = service


def _get_eval_service() -> EvalService:
    """Return the shared EvalService instance.

    The instance is created during application lifespan startup. This
    factory function is used as a FastAPI dependency so it can be
    overridden in tests.
    """
    if _shared_eval_service is not None:
        return _shared_eval_service

    # Fallback: construct ephemeral clients (backwards compatibility)
    settings = get_settings()
    return EvalService(
        collector_client=CollectorClient(
            base_url=settings.data_collector_url,
            timeout=settings.inference_timeout,
        ),
        inference_client=InferenceClient(timeout=settings.inference_timeout),
        metrics_service=MetricsService(),
        registry_client=RegistryClient(
            base_url=settings.model_registry_url,
            timeout=settings.inference_timeout,
        ),
        settings=settings,
    )


@router.post(
    "",
    response_model=Evaluation,
    status_code=201,
    summary="Create evaluation",
    description="Start an asynchronous evaluation for a model version. Creates the record and kicks off the evaluation pipeline in the background. Poll GET /api/evaluations/{eval_id} for status updates.",
)
async def create_evaluation(
    payload: EvalCreateRequest,
    session: AsyncSession = Depends(get_session),
    eval_service: EvalService = Depends(_get_eval_service),
) -> Evaluation:
    """Start an evaluation for a model version.

    Creates the evaluation record and kicks off the evaluation pipeline
    as a background task. The evaluation runs asynchronously; poll
    ``GET /api/evaluations/{eval_id}`` for status updates.
    """
    evaluation = await eval_service.create_evaluation(session, payload)

    # Fire-and-forget: run the evaluation pipeline in the background with a
    # fresh session. Any exception is logged and the evaluation is marked
    # FAILED so callers see the error instead of a stuck RUNNING row.
    async def _run_background() -> None:
        from db import _session_factory

        if _session_factory is None:
            logger.error("cannot_run_evaluation_no_db", eval_id=str(evaluation.id))
            return
        try:
            async with _session_factory() as bg_session:
                await eval_service.run_evaluation(bg_session, evaluation.id)
        except Exception as exc:
            logger.error(
                "background_evaluation_failed",
                eval_id=str(evaluation.id),
                error=str(exc),
                exc_info=True,
            )
            try:
                async with _session_factory() as err_session:
                    await eval_service._update_status(
                        err_session,
                        evaluation.id,
                        EvalStatus.FAILED,
                        error_message=f"Background task crashed: {exc}",
                        completed_at=datetime.now(UTC),
                    )
            except Exception:
                logger.error(
                    "background_evaluation_cleanup_failed",
                    eval_id=str(evaluation.id),
                    exc_info=True,
                )

    task = asyncio.create_task(_run_background())
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    logger.info(
        "evaluation_started",
        eval_id=str(evaluation.id),
        model_id=str(payload.model_id),
        schema_id=payload.schema_id,
    )

    return evaluation


@router.get(
    "",
    response_model=EvalListResponse,
    summary="List evaluations",
    description="List evaluations with optional filtering by schema_id and status, plus limit/offset pagination.",
)
async def list_evaluations(
    schema_id: str | None = Query(default=None),
    status: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_session),
    eval_service: EvalService = Depends(_get_eval_service),
) -> EvalListResponse:
    """List evaluations with optional filtering and pagination."""
    evaluations, total = await eval_service.list_evaluations(
        session,
        schema_id=schema_id,
        status=status,
        limit=limit,
        offset=offset,
    )
    return EvalListResponse(evaluations=evaluations, total=total)


@router.get(
    "/{eval_id}",
    response_model=Evaluation,
    summary="Get evaluation",
    description="Retrieve evaluation details including metrics and per-sample results.",
)
async def get_evaluation(
    eval_id: UUID,
    session: AsyncSession = Depends(get_session),
    eval_service: EvalService = Depends(_get_eval_service),
) -> Evaluation:
    """Get evaluation details including metrics and per-sample results."""
    evaluation = await eval_service.get(session, eval_id)
    if evaluation is None:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    return evaluation


@router.post(
    "/{eval_id}/approve",
    response_model=Evaluation,
    summary="Approve evaluation",
    description="Manually approve a completed evaluation that was held for review. Only evaluations in 'completed' status can be approved.",
)
async def approve_evaluation(
    eval_id: UUID,
    payload: EvalApproveRequest | None = None,
    session: AsyncSession = Depends(get_session),
    eval_service: EvalService = Depends(_get_eval_service),
) -> Evaluation:
    """Manually approve a completed evaluation that was held for review.

    Only evaluations in ``completed`` status can be approved. Evaluations that
    were auto-approved or already approved/rejected cannot be re-approved.
    """
    review_notes = payload.review_notes if payload else None
    evaluation = await eval_service.approve(session, eval_id, review_notes=review_notes)

    if evaluation is None:
        existing = await eval_service.get(session, eval_id)
        if existing is None:
            raise HTTPException(status_code=404, detail="Evaluation not found")
        raise HTTPException(
            status_code=409,
            detail=f"Cannot approve evaluation in '{existing.status}' status. "
            f"Only evaluations in 'completed' status can be manually approved.",
        )

    return evaluation


@router.post(
    "/{eval_id}/reject",
    response_model=Evaluation,
    summary="Reject evaluation",
    description="Manually reject a completed evaluation. Only evaluations in 'completed' status can be rejected. A reason for rejection is required.",
)
async def reject_evaluation(
    eval_id: UUID,
    payload: EvalRejectRequest,
    session: AsyncSession = Depends(get_session),
    eval_service: EvalService = Depends(_get_eval_service),
) -> Evaluation:
    """Manually reject a completed evaluation.

    Only evaluations in ``completed`` status can be rejected. A reason
    for rejection is required.
    """
    evaluation = await eval_service.reject(session, eval_id, review_notes=payload.review_notes)

    if evaluation is None:
        existing = await eval_service.get(session, eval_id)
        if existing is None:
            raise HTTPException(status_code=404, detail="Evaluation not found")
        raise HTTPException(
            status_code=409,
            detail=f"Cannot reject evaluation in '{existing.status}' status. "
            f"Only evaluations in 'completed' status can be rejected.",
        )

    return evaluation
