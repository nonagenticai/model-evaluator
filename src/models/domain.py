"""Pydantic v2 domain models for the model-evaluator service.

These models represent evaluation requests, metrics, gate decisions,
and the API request/response schemas.
"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from util._uuid7 import uuid7

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class EvalStatus(StrEnum):
    """Lifecycle status of an evaluation run.

    Valid transitions:
        pending -> running
        running -> completed | approved | failed
        completed -> approved | rejected
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    APPROVED = "approved"
    REJECTED = "rejected"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class EvalMetrics(BaseModel):
    """Computed evaluation metrics for a model evaluation run."""

    model_config = ConfigDict(from_attributes=True)

    json_parse_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Fraction of outputs that parse as valid JSON"
    )
    schema_valid_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Fraction that match the output Pydantic/JSON schema"
    )
    exact_match_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Fraction of exact matches to ground truth"
    )
    avg_latency_ms: float = Field(
        ..., ge=0.0, description="Average inference latency on real Pi hardware (ms)"
    )
    p95_latency_ms: float = Field(..., ge=0.0, description="P95 inference latency (ms)")
    composite_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Weighted composite score: 0.15*json + 0.35*schema + 0.50*exact_match",
    )
    total_samples: int = Field(..., ge=0, description="Total samples evaluated")
    passed_samples: int = Field(..., ge=0, description="Samples that matched ground truth exactly")
    failed_samples: int = Field(..., ge=0, description="Samples that did not match ground truth")


# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------


class Evaluation(BaseModel):
    """Full representation of a persisted evaluation run."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID = Field(default_factory=uuid7, description="Unique evaluation identifier")
    model_id: UUID = Field(..., description="References model version in model-registry")
    schema_id: str = Field(..., description="Schema identifier (e.g. ks_EntitySpotterAgent)")
    status: EvalStatus = Field(default=EvalStatus.PENDING, description="Current evaluation status")
    endpoint: str = Field(..., description="llama.cpp endpoint to test against")
    metrics: EvalMetrics | None = Field(
        default=None, description="Computed metrics after evaluation completes"
    )
    threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Composite score threshold for auto-approval",
    )
    max_samples: int = Field(
        default=100, ge=10, le=1000, description="Maximum ground truth samples to evaluate"
    )
    auto_approved: bool = Field(
        default=False, description="Whether the model was auto-approved based on composite score"
    )
    review_notes: str | None = Field(default=None, description="Notes from manual review")
    sample_results: list[dict[str, Any]] = Field(
        default_factory=list, description="Per-sample evaluation details"
    )
    json_schema: dict[str, Any] | None = Field(
        default=None, description="JSON schema for output validation"
    )
    gbnf_grammar: str | None = Field(
        default=None, description="GBNF grammar for constrained decoding on llama.cpp"
    )
    scoring_profile: str = Field(
        default="default", description="Scoring profile used for composite score calculation"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When this evaluation was created",
    )
    started_at: datetime | None = Field(default=None, description="When evaluation started running")
    completed_at: datetime | None = Field(default=None, description="When evaluation finished")
    error_message: str | None = Field(
        default=None, description="Error message if evaluation failed"
    )


# ---------------------------------------------------------------------------
# API request / response schemas
# ---------------------------------------------------------------------------


class EvalCreateRequest(BaseModel):
    """Request body for ``POST /api/evaluations``."""

    model_id: UUID = Field(..., description="Model version UUID from model-registry")
    schema_id: str = Field(..., description="Schema identifier")
    endpoint: str = Field(..., description="llama.cpp endpoint to test against")
    threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Composite score threshold for auto-approval. "
        "If omitted, falls back to the service's default_threshold setting.",
    )
    max_samples: int = Field(
        default=100, ge=10, le=1000, description="Maximum ground truth samples"
    )
    json_schema: dict[str, Any] | None = Field(
        default=None, description="JSON schema for output validation"
    )
    gbnf_grammar: str | None = Field(
        default=None, description="GBNF grammar for constrained decoding"
    )
    scoring_profile: str = Field(
        default="default",
        description="Scoring profile: 'default' (exact_match heavy) or 'structural' (schema_valid heavy)",
    )


class EvalListResponse(BaseModel):
    """Response body for ``GET /api/evaluations``."""

    evaluations: list[Evaluation]
    total: int


class EvalApproveRequest(BaseModel):
    """Request body for ``POST /api/evaluations/{eval_id}/approve``."""

    review_notes: str | None = Field(default=None, description="Optional notes from reviewer")


class EvalRejectRequest(BaseModel):
    """Request body for ``POST /api/evaluations/{eval_id}/reject``."""

    review_notes: str = Field(..., description="Reason for rejection (required)")


class LivenessResponse(BaseModel):
    """Liveness probe response body."""

    status: str = "alive"


class HealthResponse(BaseModel):
    """Health-check response body."""

    status: str
    service: str
    version: str


class ReadinessResponse(BaseModel):
    """Readiness-check response body.

    Reports partial readiness: database is the primary gate, while
    upstream services (inference endpoint, data-collector) are checked
    but do not block readiness -- they report degraded status instead.
    """

    status: str
    database: str
    inference: str = Field(
        default="unknown",
        description="Inference endpoint health: ok|error|unknown",
    )
    collector: str = Field(
        default="unknown",
        description="Data-collector health: ok|error|unknown",
    )
