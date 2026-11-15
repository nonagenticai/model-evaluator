"""Tests for the evaluations API endpoints.

Tests cover:
- Creating evaluations (happy path, defaults, validation errors, DB state)
- Listing evaluations (empty, filtering, pagination, response body)
- Getting evaluation details (existing, nonexistent, full body)
- Manual approve/reject flows (happy path, state guards, double-action, DB verification)
- Health and readiness endpoints (response structure, dynamic values)
- Edge cases (404, invalid state transitions, concurrent modifications)
- Full evaluation pipeline (auto-approve, hold, failures, partial errors)
"""

from typing import Any, cast
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.db import EvaluationORM
from models.domain import EvalCreateRequest, EvalMetrics, EvalStatus
from services.eval_service import EvalService

# ---------------------------------------------------------------------------
# Health endpoints
# ---------------------------------------------------------------------------


class TestHealth:
    """Tests for health check endpoints."""

    async def test_health_returns_ok(self, client: AsyncClient) -> None:
        """Health check should return correct service identity and version."""
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "model-evaluator"
        assert data["version"] == "0.1.0"
        # Verify the response has exactly the expected keys
        assert set(data.keys()) == {"status", "service", "version"}

    async def test_liveness_returns_alive(self, client: AsyncClient) -> None:
        """Liveness probe should confirm the process is alive."""
        response = await client.get("/health/live")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"
        assert set(data.keys()) == {"status"}

    async def test_root_returns_service_metadata(self, client: AsyncClient) -> None:
        """Root endpoint should expose service metadata with doc links."""
        response = await client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "model-evaluator"
        assert data["version"] == "0.1.0"
        assert data["docs"] == "/docs"
        assert data["health"] == "/health"


# ---------------------------------------------------------------------------
# Create evaluation
# ---------------------------------------------------------------------------


class TestCreateEvaluation:
    """Tests for POST /api/evaluations."""

    async def test_create_evaluation_success(
        self,
        client: AsyncClient,
        eval_payload: dict[str, Any],
        session: AsyncSession,
    ) -> None:
        """Created evaluation should be persisted in DB with correct fields."""
        response = await client.post("/api/evaluations", json=eval_payload)
        assert response.status_code == 201
        data = response.json()
        assert data["schema_id"] == eval_payload["schema_id"]
        assert data["endpoint"] == eval_payload["endpoint"]
        assert data["status"] == "pending"
        assert data["threshold"] == eval_payload["threshold"]
        assert data["auto_approved"] is False
        assert data["metrics"] is None
        assert data["review_notes"] is None
        assert data["error_message"] is None
        assert data["started_at"] is None
        assert data["completed_at"] is None
        assert data["scoring_profile"] == "default"
        assert "id" in data
        assert "created_at" in data

        # Verify DB state
        eval_id = UUID(data["id"])
        result = await session.execute(select(EvaluationORM).where(EvaluationORM.id == eval_id))
        row = result.scalar_one_or_none()
        assert row is not None
        assert row.schema_id == eval_payload["schema_id"]
        assert row.status == "pending"
        assert str(row.model_id) == eval_payload["model_id"]

    async def test_create_evaluation_minimal_payload(
        self,
        client: AsyncClient,
    ) -> None:
        """Ensure defaults are applied when optional fields are omitted."""
        payload = {
            "model_id": str(uuid4()),
            "schema_id": "test_schema",
            "endpoint": "http://localhost:8080",
        }
        response = await client.post("/api/evaluations", json=payload)
        assert response.status_code == 201
        data = response.json()
        # Threshold should fall back to settings default (0.85)
        assert data["threshold"] == 0.85
        assert data["max_samples"] == 100
        assert data["gbnf_grammar"] is None
        assert data["json_schema"] is None
        assert data["scoring_profile"] == "default"

    async def test_create_evaluation_with_gbnf_grammar(
        self,
        client: AsyncClient,
    ) -> None:
        payload = {
            "model_id": str(uuid4()),
            "schema_id": "test_schema",
            "endpoint": "http://localhost:8080",
            "gbnf_grammar": 'root ::= "{" ws "entities" ws ":" ws "[" ws "]" ws "}"',
        }
        response = await client.post("/api/evaluations", json=payload)
        assert response.status_code == 201
        data = response.json()
        assert data["gbnf_grammar"] == payload["gbnf_grammar"]

    async def test_create_evaluation_with_scoring_profile(
        self,
        client: AsyncClient,
    ) -> None:
        """Structural scoring profile should be stored on the evaluation."""
        payload = {
            "model_id": str(uuid4()),
            "schema_id": "test_schema",
            "endpoint": "http://localhost:8080",
            "scoring_profile": "structural",
        }
        response = await client.post("/api/evaluations", json=payload)
        assert response.status_code == 201
        data = response.json()
        assert data["scoring_profile"] == "structural"

    async def test_create_evaluation_invalid_threshold(
        self,
        client: AsyncClient,
    ) -> None:
        payload = {
            "model_id": str(uuid4()),
            "schema_id": "test_schema",
            "endpoint": "http://localhost:8080",
            "threshold": 1.5,
        }
        response = await client.post("/api/evaluations", json=payload)
        assert response.status_code == 422
        # Verify error detail mentions threshold constraint
        body = response.json()
        assert "detail" in body

    async def test_create_evaluation_invalid_max_samples(
        self,
        client: AsyncClient,
    ) -> None:
        payload = {
            "model_id": str(uuid4()),
            "schema_id": "test_schema",
            "endpoint": "http://localhost:8080",
            "max_samples": 5,
        }
        response = await client.post("/api/evaluations", json=payload)
        assert response.status_code == 422

    async def test_create_evaluation_missing_required_fields(
        self,
        client: AsyncClient,
    ) -> None:
        """Missing model_id, schema_id, or endpoint should 422."""
        # Missing model_id
        response = await client.post(
            "/api/evaluations",
            json={"schema_id": "s", "endpoint": "http://x:8080"},
        )
        assert response.status_code == 422

        # Missing schema_id
        response = await client.post(
            "/api/evaluations",
            json={"model_id": str(uuid4()), "endpoint": "http://x:8080"},
        )
        assert response.status_code == 422

        # Missing endpoint
        response = await client.post(
            "/api/evaluations",
            json={"model_id": str(uuid4()), "schema_id": "s"},
        )
        assert response.status_code == 422

    async def test_create_evaluation_invalid_model_id(
        self,
        client: AsyncClient,
    ) -> None:
        """Non-UUID model_id should 422."""
        payload = {
            "model_id": "not-a-uuid",
            "schema_id": "test_schema",
            "endpoint": "http://localhost:8080",
        }
        response = await client.post("/api/evaluations", json=payload)
        assert response.status_code == 422

    async def test_create_max_samples_at_boundary(
        self,
        client: AsyncClient,
    ) -> None:
        """max_samples at boundaries (10 and 1000) should be accepted."""
        for max_samples in [10, 1000]:
            payload = {
                "model_id": str(uuid4()),
                "schema_id": "test_schema",
                "endpoint": "http://localhost:8080",
                "max_samples": max_samples,
            }
            response = await client.post("/api/evaluations", json=payload)
            assert response.status_code == 201
            assert response.json()["max_samples"] == max_samples

    async def test_create_max_samples_over_limit(
        self,
        client: AsyncClient,
    ) -> None:
        """max_samples above 1000 should 422."""
        payload = {
            "model_id": str(uuid4()),
            "schema_id": "test_schema",
            "endpoint": "http://localhost:8080",
            "max_samples": 1001,
        }
        response = await client.post("/api/evaluations", json=payload)
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# List evaluations
# ---------------------------------------------------------------------------


class TestListEvaluations:
    """Tests for GET /api/evaluations."""

    async def test_list_empty(self, client: AsyncClient) -> None:
        response = await client.get("/api/evaluations")
        assert response.status_code == 200
        data = response.json()
        assert data["evaluations"] == []
        assert data["total"] == 0
        assert set(data.keys()) == {"evaluations", "total"}

    async def test_list_after_create(
        self,
        client: AsyncClient,
        eval_payload: dict[str, Any],
    ) -> None:
        """Created evaluation should appear in list with correct fields."""
        create_resp = await client.post("/api/evaluations", json=eval_payload)
        created = create_resp.json()

        response = await client.get("/api/evaluations")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 1
        assert len(data["evaluations"]) >= 1
        # Find our evaluation
        found = [e for e in data["evaluations"] if e["id"] == created["id"]]
        assert len(found) == 1
        assert found[0]["schema_id"] == eval_payload["schema_id"]
        assert found[0]["status"] == "pending"

    async def test_list_filter_by_schema_id(
        self,
        client: AsyncClient,
        eval_payload: dict[str, Any],
    ) -> None:
        await client.post("/api/evaluations", json=eval_payload)
        response = await client.get(
            "/api/evaluations",
            params={"schema_id": eval_payload["schema_id"]},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 1
        for ev in data["evaluations"]:
            assert ev["schema_id"] == eval_payload["schema_id"]

    async def test_list_filter_nonexistent_schema(
        self,
        client: AsyncClient,
    ) -> None:
        response = await client.get(
            "/api/evaluations",
            params={"schema_id": "nonexistent_schema"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["evaluations"] == []
        assert data["total"] == 0

    async def test_list_pagination(
        self,
        client: AsyncClient,
        eval_payload: dict[str, Any],
    ) -> None:
        """Pagination should limit results but report correct total."""
        created_ids = []
        for _ in range(3):
            payload = {**eval_payload, "model_id": str(uuid4())}
            resp = await client.post("/api/evaluations", json=payload)
            created_ids.append(resp.json()["id"])

        # Get first page
        response = await client.get(
            "/api/evaluations",
            params={"limit": 2, "offset": 0},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["evaluations"]) == 2
        assert data["total"] >= 3

        # Get second page
        response = await client.get(
            "/api/evaluations",
            params={"limit": 2, "offset": 2},
        )
        data2 = response.json()
        assert len(data2["evaluations"]) >= 1

        # No overlap between pages
        page1_ids = {e["id"] for e in data["evaluations"]}
        page2_ids = {e["id"] for e in data2["evaluations"]}
        assert page1_ids.isdisjoint(page2_ids)

    async def test_list_filter_by_status(
        self,
        client: AsyncClient,
        eval_payload: dict[str, Any],
    ) -> None:
        """Filtering by status should only return matching evaluations."""
        await client.post("/api/evaluations", json=eval_payload)

        response = await client.get(
            "/api/evaluations",
            params={"status": "pending"},
        )
        data = response.json()
        assert data["total"] >= 1
        for ev in data["evaluations"]:
            assert ev["status"] == "pending"

        # No approved evaluations
        response = await client.get(
            "/api/evaluations",
            params={"status": "approved"},
        )
        data = response.json()
        assert data["total"] == 0


# ---------------------------------------------------------------------------
# Get evaluation
# ---------------------------------------------------------------------------


class TestGetEvaluation:
    """Tests for GET /api/evaluations/{eval_id}."""

    async def test_get_existing(
        self,
        client: AsyncClient,
        eval_payload: dict[str, Any],
    ) -> None:
        create_resp = await client.post("/api/evaluations", json=eval_payload)
        created = create_resp.json()
        eval_id = created["id"]

        response = await client.get(f"/api/evaluations/{eval_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == eval_id
        assert data["schema_id"] == eval_payload["schema_id"]
        assert data["endpoint"] == eval_payload["endpoint"]
        assert data["status"] == "pending"
        assert data["model_id"] == eval_payload["model_id"]

    async def test_get_nonexistent(self, client: AsyncClient) -> None:
        fake_id = str(uuid4())
        response = await client.get(f"/api/evaluations/{fake_id}")
        assert response.status_code == 404
        body = response.json()
        assert "detail" in body
        assert body["detail"] == "Evaluation not found"

    async def test_get_invalid_uuid(self, client: AsyncClient) -> None:
        """Invalid UUID format should return 422."""
        response = await client.get("/api/evaluations/not-a-uuid")
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Approve / Reject
# ---------------------------------------------------------------------------


class TestApproveReject:
    """Tests for manual approve and reject endpoints."""

    async def test_approve_completed_evaluation(
        self,
        client: AsyncClient,
        eval_payload: dict[str, Any],
        session: AsyncSession,
        eval_service: EvalService,
    ) -> None:
        """Approve an evaluation in 'completed' status; verify DB state."""
        request = EvalCreateRequest(**eval_payload)
        evaluation = await eval_service.create_evaluation(session, request)

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

        response = await client.post(
            f"/api/evaluations/{evaluation.id}/approve",
            json={"review_notes": "Looks good for this use case"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "approved"
        assert data["review_notes"] == "Looks good for this use case"
        # Metrics should be preserved after approval
        assert data["metrics"] is not None
        assert data["metrics"]["composite_score"] == 0.73

        # Verify DB state
        result = await session.execute(
            select(EvaluationORM).where(EvaluationORM.id == evaluation.id)
        )
        row = result.scalar_one()
        assert row.status == "approved"
        assert row.review_notes == "Looks good for this use case"

    async def test_reject_completed_evaluation(
        self,
        client: AsyncClient,
        eval_payload: dict[str, Any],
        session: AsyncSession,
        eval_service: EvalService,
    ) -> None:
        """Reject an evaluation in 'completed' status; verify DB state."""
        request = EvalCreateRequest(**eval_payload)
        evaluation = await eval_service.create_evaluation(session, request)

        await eval_service._update_status(
            session,
            evaluation.id,
            EvalStatus.COMPLETED,
            metrics=EvalMetrics(
                json_parse_rate=0.5,
                schema_valid_rate=0.3,
                exact_match_rate=0.1,
                avg_latency_ms=500.0,
                p95_latency_ms=800.0,
                composite_score=0.23,
                total_samples=50,
                passed_samples=5,
                failed_samples=45,
            ),
        )

        response = await client.post(
            f"/api/evaluations/{evaluation.id}/reject",
            json={"review_notes": "Model quality too low for production"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "rejected"
        assert data["review_notes"] == "Model quality too low for production"

        # Verify DB state
        result = await session.execute(
            select(EvaluationORM).where(EvaluationORM.id == evaluation.id)
        )
        row = result.scalar_one()
        assert row.status == "rejected"

    async def test_approve_pending_returns_conflict(
        self,
        client: AsyncClient,
        eval_payload: dict[str, Any],
    ) -> None:
        """Cannot approve an evaluation that is still pending."""
        create_resp = await client.post("/api/evaluations", json=eval_payload)
        eval_id = create_resp.json()["id"]

        response = await client.post(
            f"/api/evaluations/{eval_id}/approve",
            json={"review_notes": "Premature approval"},
        )
        assert response.status_code == 409
        body = response.json()
        assert "pending" in body["detail"]
        assert "completed" in body["detail"].lower()

    async def test_reject_pending_returns_conflict(
        self,
        client: AsyncClient,
        eval_payload: dict[str, Any],
    ) -> None:
        """Cannot reject an evaluation that is still pending."""
        create_resp = await client.post("/api/evaluations", json=eval_payload)
        eval_id = create_resp.json()["id"]

        response = await client.post(
            f"/api/evaluations/{eval_id}/reject",
            json={"review_notes": "Premature rejection"},
        )
        assert response.status_code == 409
        body = response.json()
        assert "pending" in body["detail"]

    async def test_approve_already_approved_returns_conflict(
        self,
        client: AsyncClient,
        eval_payload: dict[str, Any],
        session: AsyncSession,
        eval_service: EvalService,
    ) -> None:
        """Cannot approve an already-approved evaluation (double-approve guard)."""
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

        # First approve should succeed
        resp1 = await client.post(
            f"/api/evaluations/{evaluation.id}/approve",
            json={"review_notes": "First approval"},
        )
        assert resp1.status_code == 200
        assert resp1.json()["status"] == "approved"

        # Second approve should 409
        resp2 = await client.post(
            f"/api/evaluations/{evaluation.id}/approve",
            json={"review_notes": "Second approval attempt"},
        )
        assert resp2.status_code == 409
        assert "approved" in resp2.json()["detail"].lower()

    async def test_reject_already_rejected_returns_conflict(
        self,
        client: AsyncClient,
        eval_payload: dict[str, Any],
        session: AsyncSession,
        eval_service: EvalService,
    ) -> None:
        """Cannot reject an already-rejected evaluation (double-reject guard)."""
        request = EvalCreateRequest(**eval_payload)
        evaluation = await eval_service.create_evaluation(session, request)

        await eval_service._update_status(
            session,
            evaluation.id,
            EvalStatus.COMPLETED,
        )

        # First reject
        resp1 = await client.post(
            f"/api/evaluations/{evaluation.id}/reject",
            json={"review_notes": "First rejection"},
        )
        assert resp1.status_code == 200

        # Second reject should 409
        resp2 = await client.post(
            f"/api/evaluations/{evaluation.id}/reject",
            json={"review_notes": "Second rejection attempt"},
        )
        assert resp2.status_code == 409

    async def test_approve_rejected_returns_conflict(
        self,
        client: AsyncClient,
        eval_payload: dict[str, Any],
        session: AsyncSession,
        eval_service: EvalService,
    ) -> None:
        """Cannot approve an evaluation that was already rejected."""
        request = EvalCreateRequest(**eval_payload)
        evaluation = await eval_service.create_evaluation(session, request)

        await eval_service._update_status(
            session,
            evaluation.id,
            EvalStatus.COMPLETED,
        )
        await eval_service._update_status(
            session,
            evaluation.id,
            EvalStatus.REJECTED,
            review_notes="bad model",
        )

        response = await client.post(
            f"/api/evaluations/{evaluation.id}/approve",
            json={"review_notes": "Trying to approve after reject"},
        )
        assert response.status_code == 409

    async def test_approve_nonexistent_returns_404(
        self,
        client: AsyncClient,
    ) -> None:
        fake_id = str(uuid4())
        response = await client.post(
            f"/api/evaluations/{fake_id}/approve",
            json={"review_notes": "Does not exist"},
        )
        assert response.status_code == 404
        assert response.json()["detail"] == "Evaluation not found"

    async def test_reject_nonexistent_returns_404(
        self,
        client: AsyncClient,
    ) -> None:
        fake_id = str(uuid4())
        response = await client.post(
            f"/api/evaluations/{fake_id}/reject",
            json={"review_notes": "Does not exist"},
        )
        assert response.status_code == 404
        assert response.json()["detail"] == "Evaluation not found"

    async def test_reject_requires_review_notes(
        self,
        client: AsyncClient,
        eval_payload: dict[str, Any],
    ) -> None:
        """Rejection must include a reason."""
        create_resp = await client.post("/api/evaluations", json=eval_payload)
        eval_id = create_resp.json()["id"]

        response = await client.post(
            f"/api/evaluations/{eval_id}/reject",
            json={},
        )
        assert response.status_code == 422

    async def test_approve_running_returns_conflict(
        self,
        client: AsyncClient,
        eval_payload: dict[str, Any],
        session: AsyncSession,
        eval_service: EvalService,
    ) -> None:
        """Cannot approve an evaluation that is still running."""
        request = EvalCreateRequest(**eval_payload)
        evaluation = await eval_service.create_evaluation(session, request)
        await eval_service._update_status(
            session,
            evaluation.id,
            EvalStatus.RUNNING,
        )

        response = await client.post(
            f"/api/evaluations/{evaluation.id}/approve",
            json={"review_notes": "Too early"},
        )
        assert response.status_code == 409
        assert "running" in response.json()["detail"]

    async def test_reject_failed_returns_conflict(
        self,
        client: AsyncClient,
        eval_payload: dict[str, Any],
        session: AsyncSession,
        eval_service: EvalService,
    ) -> None:
        """Cannot reject a failed evaluation."""
        request = EvalCreateRequest(**eval_payload)
        evaluation = await eval_service.create_evaluation(session, request)
        await eval_service._update_status(
            session,
            evaluation.id,
            EvalStatus.FAILED,
            error_message="something broke",
        )

        response = await client.post(
            f"/api/evaluations/{evaluation.id}/reject",
            json={"review_notes": "Already failed"},
        )
        assert response.status_code == 409


# ---------------------------------------------------------------------------
# Full evaluation pipeline (with mocked services)
# ---------------------------------------------------------------------------


class TestEvalPipeline:
    """Integration tests for the evaluation pipeline using mocked external services."""

    async def test_auto_approve_above_threshold(
        self,
        session: AsyncSession,
        eval_service: EvalService,
        eval_payload: dict[str, Any],
        ground_truth_samples: list[dict[str, Any]],
    ) -> None:
        """Model that scores above threshold should be auto-approved with correct DB state."""
        cast(Any, eval_service.collector_client).get_validated_samples = AsyncMock(
            return_value=ground_truth_samples
        )

        async def mock_complete(
            endpoint: str, messages: list, grammar: str | None = None, **kw: Any
        ) -> dict:
            for sample in ground_truth_samples:
                if sample["messages"] == messages:
                    return {"choices": [{"message": {"content": sample["response_content"]}}]}
            return {"choices": [{"message": {"content": '{"entities": []}'}}]}

        cast(Any, eval_service.inference_client).complete = AsyncMock(side_effect=mock_complete)

        request = EvalCreateRequest(**eval_payload)
        evaluation = await eval_service.create_evaluation(session, request)
        await eval_service.run_evaluation(session, evaluation.id)

        result = await eval_service.get(session, evaluation.id)
        assert result is not None
        assert result.status == EvalStatus.APPROVED
        assert result.auto_approved is True
        assert result.metrics is not None
        assert result.metrics.composite_score >= eval_payload["threshold"]
        assert result.metrics.exact_match_rate == 1.0
        assert result.metrics.total_samples == len(ground_truth_samples)
        assert result.metrics.passed_samples == len(ground_truth_samples)
        assert result.metrics.failed_samples == 0
        assert result.completed_at is not None
        assert result.started_at is not None
        assert result.error_message is None

        # Verify registry was notified with correct args
        report_mock = cast(Any, eval_service.registry_client).report_evaluation
        report_mock.assert_called_once()
        call_kwargs = report_mock.call_args
        assert call_kwargs[1]["model_id"] == evaluation.model_id
        assert call_kwargs[1]["approved"] is True
        assert call_kwargs[1]["metrics"].composite_score == result.metrics.composite_score

        # Verify DB state
        db_result = await session.execute(
            select(EvaluationORM).where(EvaluationORM.id == evaluation.id)
        )
        row = db_result.scalar_one()
        assert row.status == "approved"
        assert row.auto_approved is True
        assert row.metrics is not None

    async def test_hold_for_review_below_threshold(
        self,
        session: AsyncSession,
        eval_service: EvalService,
        eval_payload: dict[str, Any],
        ground_truth_samples: list[dict[str, Any]],
    ) -> None:
        """Model that scores below threshold should be held for manual review."""
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

        request = EvalCreateRequest(**eval_payload)
        evaluation = await eval_service.create_evaluation(session, request)
        await eval_service.run_evaluation(session, evaluation.id)

        result = await eval_service.get(session, evaluation.id)
        assert result is not None
        assert result.status == EvalStatus.COMPLETED
        assert result.auto_approved is False
        assert result.metrics is not None
        assert result.metrics.composite_score < eval_payload["threshold"]
        assert result.metrics.exact_match_rate == 0.0
        # JSON should still be valid (the wrong content is valid JSON)
        assert result.metrics.json_parse_rate == 1.0
        assert result.completed_at is not None
        assert result.sample_results is not None
        assert len(result.sample_results) == len(ground_truth_samples)

        # Verify registry was called with approved=False
        report_mock = cast(Any, eval_service.registry_client).report_evaluation
        report_mock.assert_called_once()
        assert report_mock.call_args[1]["approved"] is False

    async def test_evaluation_fails_on_no_ground_truth(
        self,
        session: AsyncSession,
        eval_service: EvalService,
        eval_payload: dict[str, Any],
    ) -> None:
        """Evaluation should fail if no ground truth samples are available."""
        cast(Any, eval_service.collector_client).get_validated_samples = AsyncMock(return_value=[])

        request = EvalCreateRequest(**eval_payload)
        evaluation = await eval_service.create_evaluation(session, request)
        await eval_service.run_evaluation(session, evaluation.id)

        result = await eval_service.get(session, evaluation.id)
        assert result is not None
        assert result.status == EvalStatus.FAILED
        assert result.error_message is not None
        assert "No ground truth" in result.error_message
        assert result.metrics is None
        assert result.auto_approved is False

    async def test_evaluation_fails_on_collector_error(
        self,
        session: AsyncSession,
        eval_service: EvalService,
        eval_payload: dict[str, Any],
    ) -> None:
        """Evaluation should fail gracefully if data-collector is unavailable."""
        cast(Any, eval_service.collector_client).get_validated_samples = AsyncMock(
            side_effect=ConnectionError("data-collector unreachable")
        )

        request = EvalCreateRequest(**eval_payload)
        evaluation = await eval_service.create_evaluation(session, request)
        await eval_service.run_evaluation(session, evaluation.id)

        result = await eval_service.get(session, evaluation.id)
        assert result is not None
        assert result.status == EvalStatus.FAILED
        assert result.error_message is not None
        assert "data-collector unreachable" in result.error_message
        assert result.completed_at is not None

    async def test_partial_inference_failures(
        self,
        session: AsyncSession,
        eval_service: EvalService,
        eval_payload: dict[str, Any],
        ground_truth_samples: list[dict[str, Any]],
    ) -> None:
        """Some inference calls failing should not crash the pipeline.

        Failed samples should still be counted with json_valid=False, schema_valid=False,
        exact_match=False. Uses httpx.ReadTimeout because that's the exception
        the real InferenceClient raises on a Pi hardware timeout (httpx native
        timeout, not a built-in TimeoutError).
        """
        import httpx

        cast(Any, eval_service.collector_client).get_validated_samples = AsyncMock(
            return_value=ground_truth_samples
        )

        call_count = 0

        async def flaky_complete(
            endpoint: str, messages: list, grammar: str | None = None, **kw: Any
        ) -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise httpx.ReadTimeout("Pi hardware timeout")
            return {"choices": [{"message": {"content": '{"entities": []}'}}]}

        cast(Any, eval_service.inference_client).complete = AsyncMock(side_effect=flaky_complete)

        request = EvalCreateRequest(**eval_payload)
        evaluation = await eval_service.create_evaluation(session, request)
        await eval_service.run_evaluation(session, evaluation.id)

        result = await eval_service.get(session, evaluation.id)
        assert result is not None
        assert result.status in (EvalStatus.COMPLETED, EvalStatus.APPROVED)
        assert result.metrics is not None
        assert result.metrics.total_samples == len(ground_truth_samples)
        # The failed sample should show up in results
        assert len(result.sample_results) == len(ground_truth_samples)
        error_results = [r for r in result.sample_results if "error" in r]
        assert len(error_results) == 1
        assert "Pi hardware timeout" in error_results[0]["error"]
        # Failed sample must have all the per-sample metrics zeroed out
        failed = error_results[0]
        assert failed["json_valid"] is False
        assert failed["schema_valid"] is False
        assert failed["exact_match"] is False
        assert failed["predicted"] == ""
        assert failed["latency_ms"] >= 0.0

    async def test_manual_approve_after_hold(
        self,
        session: AsyncSession,
        eval_service: EvalService,
        eval_payload: dict[str, Any],
        ground_truth_samples: list[dict[str, Any]],
    ) -> None:
        """After hold for review, manual approval should work and notify registry."""
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

        request = EvalCreateRequest(**eval_payload)
        evaluation = await eval_service.create_evaluation(session, request)
        await eval_service.run_evaluation(session, evaluation.id)

        held = await eval_service.get(session, evaluation.id)
        assert held is not None
        assert held.status == EvalStatus.COMPLETED

        cast(Any, eval_service.registry_client).report_evaluation.reset_mock()

        approved = await eval_service.approve(
            session, evaluation.id, review_notes="Acceptable for low-traffic schema"
        )
        assert approved is not None
        assert approved.status == EvalStatus.APPROVED
        assert approved.review_notes == "Acceptable for low-traffic schema"

        # Verify registry was notified of manual approval with correct model_id
        report_mock = cast(Any, eval_service.registry_client).report_evaluation
        report_mock.assert_called_once()
        assert report_mock.call_args[1]["model_id"] == evaluation.model_id
        assert report_mock.call_args[1]["approved"] is True

    async def test_manual_reject_after_hold(
        self,
        session: AsyncSession,
        eval_service: EvalService,
        eval_payload: dict[str, Any],
        ground_truth_samples: list[dict[str, Any]],
    ) -> None:
        """After hold for review, manual rejection should work."""
        cast(Any, eval_service.collector_client).get_validated_samples = AsyncMock(
            return_value=ground_truth_samples
        )

        cast(Any, eval_service.inference_client).complete = AsyncMock(
            return_value={"choices": [{"message": {"content": "not json at all"}}]}
        )

        request = EvalCreateRequest(**eval_payload)
        evaluation = await eval_service.create_evaluation(session, request)
        await eval_service.run_evaluation(session, evaluation.id)

        held = await eval_service.get(session, evaluation.id)
        assert held is not None
        assert held.status == EvalStatus.COMPLETED
        # Non-JSON output should have json_parse_rate == 0
        assert held.metrics is not None
        assert held.metrics.json_parse_rate == 0.0

        rejected = await eval_service.reject(
            session, evaluation.id, review_notes="Output quality unacceptable"
        )
        assert rejected is not None
        assert rejected.status == EvalStatus.REJECTED
        assert rejected.review_notes == "Output quality unacceptable"

        # Registry should NOT be notified on reject (only on approve)
        # Verify by checking the call count didn't increase after reject
        # (it was called once during run_evaluation for the "evaluated" report)

    async def test_structural_scoring_profile_pipeline(
        self,
        session: AsyncSession,
        eval_service: EvalService,
        eval_payload: dict[str, Any],
        ground_truth_samples: list[dict[str, Any]],
    ) -> None:
        """Structural scoring profile should weight schema_valid_rate more heavily."""
        cast(Any, eval_service.collector_client).get_validated_samples = AsyncMock(
            return_value=ground_truth_samples
        )

        # Return valid JSON that matches schema but doesn't exact-match
        cast(Any, eval_service.inference_client).complete = AsyncMock(
            return_value={
                "choices": [
                    {
                        "message": {
                            "content": '{"entities": [{"text": "Different", "type": "OTHER"}]}'
                        }
                    }
                ]
            }
        )

        payload = {**eval_payload, "scoring_profile": "structural", "threshold": 0.3}
        request = EvalCreateRequest.model_validate(payload)
        evaluation = await eval_service.create_evaluation(session, request)
        await eval_service.run_evaluation(session, evaluation.id)

        result = await eval_service.get(session, evaluation.id)
        assert result is not None
        assert result.scoring_profile == "structural"
        assert result.metrics is not None
        # JSON is valid, schema_valid depends on json_schema validation, exact_match is 0
        assert result.metrics.json_parse_rate == 1.0
        assert result.metrics.exact_match_rate == 0.0
        # Verify structural composite formula is used:
        # 0.20 * json_parse + 0.60 * schema_valid + 0.20 * exact_match
        expected_composite = (
            0.20 * result.metrics.json_parse_rate
            + 0.60 * result.metrics.schema_valid_rate
            + 0.20 * result.metrics.exact_match_rate
        )
        assert abs(result.metrics.composite_score - expected_composite) < 1e-9

    async def test_all_inference_failures_still_completes(
        self,
        session: AsyncSession,
        eval_service: EvalService,
        eval_payload: dict[str, Any],
        ground_truth_samples: list[dict[str, Any]],
    ) -> None:
        """All inference calls failing should produce metrics with all zeros.

        Uses httpx.ConnectTimeout (subclass of httpx.HTTPError) because that's
        what the real InferenceClient raises when the Pi node is unreachable.
        """
        import httpx

        cast(Any, eval_service.collector_client).get_validated_samples = AsyncMock(
            return_value=ground_truth_samples
        )

        cast(Any, eval_service.inference_client).complete = AsyncMock(
            side_effect=httpx.ConnectTimeout("All calls timeout")
        )

        request = EvalCreateRequest(**eval_payload)
        evaluation = await eval_service.create_evaluation(session, request)
        await eval_service.run_evaluation(session, evaluation.id)

        result = await eval_service.get(session, evaluation.id)
        assert result is not None
        assert result.status == EvalStatus.COMPLETED  # Not FAILED -- pipeline completes
        assert result.metrics is not None
        assert result.metrics.json_parse_rate == 0.0
        assert result.metrics.schema_valid_rate == 0.0
        assert result.metrics.exact_match_rate == 0.0
        assert result.metrics.composite_score == 0.0
        # Every sample must have an error recorded
        assert len(result.sample_results) == len(ground_truth_samples)
        for sr in result.sample_results:
            assert sr["error"]
            assert "All calls timeout" in sr["error"]

    async def test_auto_approve_registry_notification_failure_still_approves(
        self,
        session: AsyncSession,
        eval_service: EvalService,
        eval_payload: dict[str, Any],
        ground_truth_samples: list[dict[str, Any]],
    ) -> None:
        """If registry.report_evaluation raises during auto-approve, the evaluation
        must still be APPROVED in the DB and a registry_notification_failed
        warning must be logged. A failing registry must not roll back approval."""
        import httpx
        import structlog.testing

        cast(Any, eval_service.collector_client).get_validated_samples = AsyncMock(
            return_value=ground_truth_samples
        )

        async def mock_complete(
            endpoint: str, messages: list, grammar: str | None = None, **kw: Any
        ) -> dict:
            for sample in ground_truth_samples:
                if sample["messages"] == messages:
                    return {"choices": [{"message": {"content": sample["response_content"]}}]}
            return {"choices": [{"message": {"content": '{"entities": []}'}}]}

        cast(Any, eval_service.inference_client).complete = AsyncMock(side_effect=mock_complete)
        cast(Any, eval_service.registry_client).report_evaluation = AsyncMock(
            side_effect=httpx.ConnectError("registry unreachable")
        )

        request = EvalCreateRequest(**eval_payload)
        evaluation = await eval_service.create_evaluation(session, request)

        with structlog.testing.capture_logs() as captured:
            await eval_service.run_evaluation(session, evaluation.id)

        result = await eval_service.get(session, evaluation.id)
        assert result is not None
        assert result.status == EvalStatus.APPROVED
        assert result.auto_approved is True
        assert result.metrics is not None
        assert result.metrics.composite_score >= eval_payload["threshold"]
        # The warning log was emitted with the eval_id and model_id
        assert any(
            log["event"] == "registry_notification_failed"
            and log.get("eval_id") == str(evaluation.id)
            for log in captured
        ), f"expected registry_notification_failed log; got: {captured}"

    async def test_hold_for_review_registry_notification_failure_still_completes(
        self,
        session: AsyncSession,
        eval_service: EvalService,
        eval_payload: dict[str, Any],
        ground_truth_samples: list[dict[str, Any]],
    ) -> None:
        """If registry.report_evaluation raises during the below-threshold path,
        the evaluation must still reach COMPLETED (held for review) and log the
        warning. Nobody should be stuck because registry is flaky."""
        import httpx
        import structlog.testing

        cast(Any, eval_service.collector_client).get_validated_samples = AsyncMock(
            return_value=ground_truth_samples
        )
        cast(Any, eval_service.inference_client).complete = AsyncMock(
            return_value={
                "choices": [{"message": {"content": '{"entities": [{"text": "X", "type": "X"}]}'}}]
            }
        )
        cast(Any, eval_service.registry_client).report_evaluation = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "5xx",
                request=httpx.Request("PATCH", "http://registry:8155/api/models/x/status"),
                response=httpx.Response(503),
            )
        )

        request = EvalCreateRequest(**eval_payload)
        evaluation = await eval_service.create_evaluation(session, request)

        with structlog.testing.capture_logs() as captured:
            await eval_service.run_evaluation(session, evaluation.id)

        result = await eval_service.get(session, evaluation.id)
        assert result is not None
        assert result.status == EvalStatus.COMPLETED  # held for review
        assert result.auto_approved is False
        assert any(
            log["event"] == "registry_notification_failed"
            and log.get("eval_id") == str(evaluation.id)
            for log in captured
        ), f"expected registry_notification_failed log; got: {captured}"

    async def test_manual_approve_registry_notification_failure_still_approves(
        self,
        session: AsyncSession,
        eval_service: EvalService,
        eval_payload: dict[str, Any],
        ground_truth_samples: list[dict[str, Any]],
    ) -> None:
        """Manual approve after hold must succeed even when the follow-up
        registry notification fails. The DB state must reflect APPROVED and
        the warning must be logged."""
        import httpx
        import structlog.testing

        cast(Any, eval_service.collector_client).get_validated_samples = AsyncMock(
            return_value=ground_truth_samples
        )
        # First run produces held-for-review (wrong content)
        cast(Any, eval_service.inference_client).complete = AsyncMock(
            return_value={
                "choices": [
                    {"message": {"content": '{"entities": [{"text": "Wrong", "type": "WRONG"}]}'}}
                ]
            }
        )
        # Initial report_evaluation during run_evaluation is OK
        report_mock = AsyncMock(return_value={"status": "evaluated"})
        cast(Any, eval_service.registry_client).report_evaluation = report_mock

        request = EvalCreateRequest(**eval_payload)
        evaluation = await eval_service.create_evaluation(session, request)
        await eval_service.run_evaluation(session, evaluation.id)

        held = await eval_service.get(session, evaluation.id)
        assert held is not None
        assert held.status == EvalStatus.COMPLETED

        # Now the manual-approve registry notification fails
        report_mock.side_effect = httpx.ConnectError("registry down again")

        with structlog.testing.capture_logs() as captured:
            approved = await eval_service.approve(
                session, evaluation.id, review_notes="manual override"
            )

        assert approved is not None
        assert approved.status == EvalStatus.APPROVED
        assert approved.review_notes == "manual override"
        assert any(
            log["event"] == "registry_notification_failed"
            and log.get("eval_id") == str(evaluation.id)
            for log in captured
        ), f"expected registry_notification_failed log; got: {captured}"
