"""Main evaluation orchestration service.

Coordinates the full evaluation pipeline:
1. Fetch ground truth samples from data-collector
2. Run inference on each sample via llama.cpp on Pi hardware
3. Compute metrics
4. Apply gate logic (auto-approve or hold for manual review)
5. Report results to model-registry
"""

import json
import time
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import httpx
import jsonschema
import structlog
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from models.db import EvaluationORM
from models.domain import EvalCreateRequest, EvalMetrics, EvalStatus, Evaluation
from services.adapter_client import AdapterClient
from services.collector_client import CollectorClient
from services.inference_client import InferenceClient
from services.metrics_service import MetricsService, ScoringProfile
from services.registry_client import RegistryClient
from settings import Settings
from util._uuid7 import uuid7

logger = structlog.stdlib.get_logger(__name__)


class EvalService:
    """Orchestrates model evaluation against ground truth data on Pi hardware."""

    def __init__(
        self,
        collector_client: CollectorClient,
        inference_client: InferenceClient,
        metrics_service: MetricsService,
        registry_client: RegistryClient,
        adapter_client: AdapterClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.collector_client = collector_client
        self.inference_client = inference_client
        self.metrics_service = metrics_service
        self.registry_client = registry_client
        self.adapter_client = adapter_client
        if settings is None:
            from settings import get_settings

            settings = get_settings()
        self._settings = settings

    async def create_evaluation(
        self,
        session: AsyncSession,
        request: EvalCreateRequest,
    ) -> Evaluation:
        """Create a new evaluation record in the database.

        Args:
            session: Database session.
            request: Evaluation creation request with model details.

        Returns:
            The newly created Evaluation domain object.
        """
        eval_id = uuid7()
        now = datetime.now(UTC)
        threshold = (
            request.threshold if request.threshold is not None else self._settings.default_threshold
        )

        row = EvaluationORM(
            id=eval_id,
            model_id=request.model_id,
            schema_id=request.schema_id,
            status=EvalStatus.PENDING,
            endpoint=request.endpoint,
            threshold=threshold,
            max_samples=request.max_samples,
            auto_approved=False,
            sample_results=[],
            json_schema=request.json_schema,
            gbnf_grammar=request.gbnf_grammar,
            scoring_profile=request.scoring_profile,
            created_at=now,
        )

        session.add(row)
        await session.commit()
        await session.refresh(row)

        logger.info(
            "evaluation_created",
            eval_id=str(eval_id),
            model_id=str(request.model_id),
            schema_id=request.schema_id,
            endpoint=request.endpoint,
        )

        return _row_to_domain(row)

    async def get(self, session: AsyncSession, eval_id: UUID) -> Evaluation | None:
        """Retrieve a single evaluation by ID.

        Args:
            session: Database session.
            eval_id: UUID of the evaluation.

        Returns:
            Evaluation domain object, or None if not found.
        """
        stmt = select(EvaluationORM).where(EvaluationORM.id == eval_id)
        result = await session.execute(stmt)
        row = result.scalar_one_or_none()
        if row is None:
            return None
        return _row_to_domain(row)

    async def list_evaluations(
        self,
        session: AsyncSession,
        schema_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[Evaluation], int]:
        """List evaluations with optional filtering and pagination.

        Args:
            session: Database session.
            schema_id: Optional filter by schema identifier.
            status: Optional filter by evaluation status.
            limit: Maximum results per page.
            offset: Number of results to skip.

        Returns:
            Tuple of (list of evaluations, total count).
        """
        stmt = select(EvaluationORM)
        count_stmt = select(func.count()).select_from(EvaluationORM)

        if schema_id is not None:
            stmt = stmt.where(EvaluationORM.schema_id == schema_id)
            count_stmt = count_stmt.where(EvaluationORM.schema_id == schema_id)
        if status is not None:
            stmt = stmt.where(EvaluationORM.status == status)
            count_stmt = count_stmt.where(EvaluationORM.status == status)

        stmt = stmt.order_by(EvaluationORM.created_at.desc())
        stmt = stmt.offset(offset).limit(limit)

        result = await session.execute(stmt)
        rows = result.scalars().all()

        count_result = await session.execute(count_stmt)
        total = count_result.scalar_one()

        return [_row_to_domain(r) for r in rows], total

    async def run_evaluation(self, session: AsyncSession, eval_id: UUID) -> None:
        """Run the full evaluation pipeline for a given evaluation.

        This method:
        1. Fetches ground truth samples from data-collector
        2. Runs inference on each sample against the llama.cpp endpoint
        3. Computes evaluation metrics
        4. Applies gate logic: auto-approve if score >= threshold, else hold
        5. Reports results to model-registry

        Args:
            session: Database session.
            eval_id: UUID of the evaluation to run.
        """
        evaluation = await self.get(session, eval_id)
        if evaluation is None:
            logger.error("evaluation_not_found", eval_id=str(eval_id))
            return

        await self._update_status(
            session, eval_id, EvalStatus.RUNNING, started_at=datetime.now(UTC)
        )

        try:
            # 1. Get ground truth samples from data-collector
            ground_truth = await self.collector_client.get_validated_samples(
                schema_id=evaluation.schema_id,
                limit=evaluation.max_samples,
            )

            if not ground_truth:
                await self._update_status(
                    session,
                    eval_id,
                    EvalStatus.FAILED,
                    error_message="No ground truth samples found for schema",
                )
                return

            logger.info(
                "evaluation_ground_truth_loaded",
                eval_id=str(eval_id),
                sample_count=len(ground_truth),
            )

            # 1b. Load the LoRA adapter before running inference
            if self.adapter_client:
                try:
                    model_info = await self.registry_client.get_model(evaluation.model_id)
                    adapter_path = model_info.get("adapter_path")
                    if adapter_path:
                        await self.adapter_client.load_adapter(evaluation.schema_id, adapter_path)
                        logger.info(
                            "adapter_loaded_for_evaluation",
                            eval_id=str(eval_id),
                            schema_id=evaluation.schema_id,
                            adapter_path=adapter_path,
                        )
                    else:
                        logger.warning(
                            "no_adapter_path_for_model",
                            eval_id=str(eval_id),
                            model_id=str(evaluation.model_id),
                        )
                except httpx.HTTPError:
                    # AdapterClient.load_adapter raises only httpx exceptions
                    # (HTTPStatusError + RequestError subclasses); both are
                    # subclasses of httpx.HTTPError. Anything else propagates
                    # to the top-level pipeline catch.
                    logger.error(
                        "adapter_loading_failed",
                        eval_id=str(eval_id),
                        exc_info=True,
                    )
                    await self._update_status(
                        session,
                        eval_id,
                        EvalStatus.FAILED,
                        error_message="Failed to load adapter before evaluation",
                        completed_at=datetime.now(UTC),
                    )
                    return

            # 1c. Fetch system prompt from routing rule for inference
            system_prompt = None
            try:
                routing = await self.registry_client.get_routing_rule(evaluation.schema_id)
                system_prompt = routing.get("system_prompt")
            except httpx.HTTPError:
                # RegistryClient.get_routing_rule only raises httpx errors
                # on transport/HTTP failures; missing routing rules surface as
                # an empty dict (404 path), not an exception.
                logger.warning(
                    "routing_rule_fetch_failed",
                    eval_id=str(eval_id),
                    exc_info=True,
                )

            # 2. Run inference on each sample
            results: list[dict[str, Any]] = []
            latencies: list[float] = []

            for sample in ground_truth:
                result = await self._evaluate_sample(
                    evaluation, sample, system_prompt=system_prompt
                )
                results.append(result)
                latencies.append(result["latency_ms"])

            # 3. Compute metrics (use scoring_profile from evaluation if stored)
            raw_profile = evaluation.scoring_profile or "default"
            if raw_profile not in ("default", "structural"):
                logger.warning(
                    "invalid_scoring_profile_fallback",
                    eval_id=str(eval_id),
                    requested=raw_profile,
                    fallback="default",
                )
                scoring_profile: ScoringProfile = "default"
            else:
                scoring_profile = raw_profile
            metrics = self.metrics_service.compute(
                results, latencies, scoring_profile=scoring_profile
            )

            logger.info(
                "evaluation_metrics_computed",
                eval_id=str(eval_id),
                composite_score=metrics.composite_score,
                threshold=evaluation.threshold,
            )

            # 4. Gate decision
            now = datetime.now(UTC)

            if metrics.composite_score >= evaluation.threshold:
                await self._update_status(
                    session,
                    eval_id,
                    EvalStatus.APPROVED,
                    metrics=metrics,
                    auto_approved=True,
                    sample_results=results,
                    completed_at=now,
                )

                # 5. Notify model-registry of approval
                try:
                    await self.registry_client.report_evaluation(
                        model_id=evaluation.model_id,
                        metrics=metrics,
                        approved=True,
                    )
                except httpx.HTTPError:
                    # Registry notification is best-effort: a transport or
                    # HTTP error must not undo the auto-approval that we
                    # already persisted to the database.
                    logger.warning(
                        "registry_notification_failed",
                        eval_id=str(eval_id),
                        model_id=str(evaluation.model_id),
                        exc_info=True,
                    )

                logger.info(
                    "evaluation_auto_approved",
                    eval_id=str(eval_id),
                    composite_score=metrics.composite_score,
                )
            else:
                await self._update_status(
                    session,
                    eval_id,
                    EvalStatus.COMPLETED,
                    metrics=metrics,
                    sample_results=results,
                    completed_at=now,
                )

                # Report as evaluated but not approved
                try:
                    await self.registry_client.report_evaluation(
                        model_id=evaluation.model_id,
                        metrics=metrics,
                        approved=False,
                    )
                except httpx.HTTPError:
                    # Same best-effort policy as the approval branch above:
                    # a registry hiccup must not block holding for review.
                    logger.warning(
                        "registry_notification_failed",
                        eval_id=str(eval_id),
                        model_id=str(evaluation.model_id),
                        exc_info=True,
                    )

                logger.info(
                    "evaluation_held_for_review",
                    eval_id=str(eval_id),
                    composite_score=metrics.composite_score,
                    threshold=evaluation.threshold,
                )

        except Exception as exc:
            # Last-resort handler: this is the only place where the pipeline
            # marks an evaluation FAILED for an unexpected error. Anything
            # narrower would risk leaving rows stuck in RUNNING forever, so
            # we deliberately catch broad Exception here and log full traceback
            # via exc_info=True for post-mortem debugging.
            logger.error(
                "evaluation_failed",
                eval_id=str(eval_id),
                error=str(exc),
                exc_info=True,
            )
            await self._update_status(
                session,
                eval_id,
                EvalStatus.FAILED,
                error_message=str(exc),
                completed_at=datetime.now(UTC),
            )

    async def approve(
        self,
        session: AsyncSession,
        eval_id: UUID,
        review_notes: str | None = None,
    ) -> Evaluation | None:
        """Manually approve a completed evaluation that was held for review.

        Args:
            session: Database session.
            eval_id: UUID of the evaluation to approve.
            review_notes: Optional reviewer notes.

        Returns:
            Updated Evaluation, or None if not found or invalid state.
        """
        evaluation = await self.get(session, eval_id)
        if evaluation is None:
            return None

        if evaluation.status != EvalStatus.COMPLETED:
            logger.warning(
                "invalid_approval_state",
                eval_id=str(eval_id),
                current_status=evaluation.status,
            )
            return None

        await self._update_status(
            session,
            eval_id,
            EvalStatus.APPROVED,
            review_notes=review_notes,
        )

        # Notify model-registry
        try:
            if evaluation.metrics is not None:
                await self.registry_client.report_evaluation(
                    model_id=evaluation.model_id,
                    metrics=evaluation.metrics,
                    approved=True,
                )
        except httpx.HTTPError:
            # Manual approval has already been persisted; a registry hiccup
            # must not roll it back. Same best-effort policy as run_evaluation.
            logger.warning(
                "registry_notification_failed",
                eval_id=str(eval_id),
                model_id=str(evaluation.model_id),
                exc_info=True,
            )

        logger.info(
            "evaluation_manually_approved",
            eval_id=str(eval_id),
            review_notes=review_notes,
        )

        return await self.get(session, eval_id)

    async def reject(
        self,
        session: AsyncSession,
        eval_id: UUID,
        review_notes: str,
    ) -> Evaluation | None:
        """Manually reject a completed evaluation.

        Args:
            session: Database session.
            eval_id: UUID of the evaluation to reject.
            review_notes: Reason for rejection (required).

        Returns:
            Updated Evaluation, or None if not found or invalid state.
        """
        evaluation = await self.get(session, eval_id)
        if evaluation is None:
            return None

        if evaluation.status != EvalStatus.COMPLETED:
            logger.warning(
                "invalid_rejection_state",
                eval_id=str(eval_id),
                current_status=evaluation.status,
            )
            return None

        await self._update_status(
            session,
            eval_id,
            EvalStatus.REJECTED,
            review_notes=review_notes,
        )

        logger.info(
            "evaluation_rejected",
            eval_id=str(eval_id),
            review_notes=review_notes,
        )

        return await self.get(session, eval_id)

    async def _evaluate_sample(
        self,
        evaluation: Evaluation,
        sample: dict[str, Any],
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        """Run inference on a single sample and compare to ground truth.

        Args:
            evaluation: The parent evaluation with endpoint and grammar config.
            sample: Ground truth sample with messages and expected response.
            system_prompt: System prompt from routing rule to prepend to messages.

        Returns:
            Result dict with json_valid, schema_valid, exact_match, latency_ms, etc.
        """
        start = time.monotonic()

        try:
            # Inject system prompt if available and not already present
            messages = list(sample["messages"])
            if system_prompt and (not messages or messages[0].get("role") != "system"):
                messages.insert(0, {"role": "system", "content": system_prompt})

            response = await self.inference_client.complete(
                endpoint=evaluation.endpoint,
                messages=messages,
                grammar=evaluation.gbnf_grammar,
                model=evaluation.schema_id,
            )
            latency_ms = (time.monotonic() - start) * 1000

            content = response["choices"][0]["message"]["content"]
        except (httpx.HTTPError, KeyError, IndexError, TypeError) as exc:
            # httpx.HTTPError: inference_client transport / non-2xx (incl.
            #   ReadTimeout) when calling the Pi llama.cpp endpoint.
            # KeyError / IndexError / TypeError: malformed response shape from
            #   the Pi node (missing "choices", empty list, None body, etc.).
            # Any broader exception is treated as a programmer bug and bubbles
            # up to the top-level pipeline catch in run_evaluation.
            latency_ms = (time.monotonic() - start) * 1000
            logger.warning(
                "sample_inference_failed",
                sample_id=sample.get("id", "unknown"),
                error=str(exc),
            )
            return {
                "sample_id": sample.get("id", "unknown"),
                "predicted": "",
                "expected": sample.get("response_content", ""),
                "json_valid": False,
                "schema_valid": False,
                "exact_match": False,
                "latency_ms": latency_ms,
                "error": str(exc),
            }

        # Check JSON parse
        json_valid = self._try_parse_json(content)

        # Check schema validity
        schema_valid = False
        if json_valid and evaluation.json_schema is not None:
            schema_valid = self._validate_schema(content, evaluation.json_schema)

        # Check exact match
        exact_match = content.strip() == sample.get("response_content", "").strip()

        return {
            "sample_id": sample.get("id", "unknown"),
            "predicted": content,
            "expected": sample.get("response_content", ""),
            "json_valid": json_valid,
            "schema_valid": schema_valid,
            "exact_match": exact_match,
            "latency_ms": latency_ms,
        }

    @staticmethod
    def _try_parse_json(content: str | None) -> bool:
        """Attempt to parse content as JSON, returning True on success."""
        if content is None:
            return False
        try:
            json.loads(content)
            return True
        except json.JSONDecodeError, TypeError:
            return False

    @staticmethod
    def _validate_schema(content: str, schema: dict[str, Any]) -> bool:
        """Validate parsed JSON content against a JSON schema.

        Args:
            content: Raw string that should be valid JSON.
            schema: JSON Schema dict to validate against.

        Returns:
            True if valid, False otherwise.
        """
        try:
            parsed = json.loads(content)
            jsonschema.validate(instance=parsed, schema=schema)
            return True
        except json.JSONDecodeError, jsonschema.ValidationError, jsonschema.SchemaError:
            return False

    async def _update_status(
        self,
        session: AsyncSession,
        eval_id: UUID,
        status: EvalStatus,
        metrics: EvalMetrics | None = None,
        auto_approved: bool | None = None,
        review_notes: str | None = None,
        sample_results: list[dict[str, Any]] | None = None,
        error_message: str | None = None,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
    ) -> None:
        """Update evaluation status and associated fields in the database.

        Args:
            session: Database session.
            eval_id: UUID of the evaluation to update.
            status: New status value.
            metrics: Optional computed metrics (set on completion).
            auto_approved: Optional flag if auto-approved.
            review_notes: Optional reviewer notes.
            sample_results: Optional per-sample results.
            error_message: Optional error message (set on failure).
            started_at: Optional start timestamp.
            completed_at: Optional completion timestamp.
        """
        stmt = select(EvaluationORM).where(EvaluationORM.id == eval_id)
        result = await session.execute(stmt)
        row = result.scalar_one_or_none()

        if row is None:
            logger.error("evaluation_not_found_for_update", eval_id=str(eval_id))
            return

        row.status = status.value

        if metrics is not None:
            row.metrics = metrics.model_dump()
        if auto_approved is not None:
            row.auto_approved = auto_approved
        if review_notes is not None:
            row.review_notes = review_notes
        if sample_results is not None:
            row.sample_results = sample_results
        if error_message is not None:
            row.error_message = error_message
        if started_at is not None:
            row.started_at = started_at
        if completed_at is not None:
            row.completed_at = completed_at

        await session.commit()

        logger.debug(
            "evaluation_status_updated",
            eval_id=str(eval_id),
            status=status,
        )


def _row_to_domain(row: EvaluationORM) -> Evaluation:
    """Convert an ORM row to a Pydantic domain model."""
    metrics = None
    if row.metrics is not None:
        metrics = EvalMetrics.model_validate(row.metrics)

    # Legacy rows created before the scoring_profile column existed may not
    # carry the attribute at all. When the attribute is missing entirely we
    # emit a warning so the data gap is visible; rows that have the column
    # but with None/"" are expected and silently default.
    if not hasattr(row, "scoring_profile"):
        logger.warning(
            "scoring_profile_column_missing_on_row",
            eval_id=str(row.id),
            fallback="default",
        )
        scoring_profile = "default"
    else:
        scoring_profile = row.scoring_profile or "default"

    return Evaluation(
        id=row.id,
        model_id=row.model_id,
        schema_id=row.schema_id,
        status=EvalStatus(row.status),
        endpoint=row.endpoint,
        metrics=metrics,
        threshold=row.threshold,
        max_samples=row.max_samples,
        auto_approved=row.auto_approved,
        review_notes=row.review_notes,
        sample_results=row.sample_results or [],
        json_schema=row.json_schema,
        gbnf_grammar=row.gbnf_grammar,
        scoring_profile=scoring_profile,
        created_at=row.created_at,
        started_at=row.started_at,
        completed_at=row.completed_at,
        error_message=row.error_message,
    )
