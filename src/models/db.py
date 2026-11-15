"""SQLAlchemy ORM models for the model-evaluator service.

These models mirror the Pydantic schemas in models/domain.py and provide
persistence to PostgreSQL via psycopg.
"""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, Index, Integer, String, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from db import Base
from util._uuid7 import uuid7


class EvaluationORM(Base):
    """ORM model for a model evaluation run."""

    __tablename__ = "evaluations"

    id: Mapped[Any] = mapped_column(Uuid, primary_key=True, default=uuid7)
    model_id: Mapped[Any] = mapped_column(Uuid, nullable=False, index=True)
    schema_id: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending", index=True)
    endpoint: Mapped[str] = mapped_column(String(1024), nullable=False)
    metrics: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    threshold: Mapped[float] = mapped_column(Float, nullable=False, default=0.85)
    max_samples: Mapped[int] = mapped_column(Integer, nullable=False, default=100)
    auto_approved: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    review_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    sample_results: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False, default=list)
    json_schema: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    gbnf_grammar: Mapped[str | None] = mapped_column(Text, nullable=True)
    scoring_profile: Mapped[str] = mapped_column(String(32), nullable=False, default="default")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index("ix_evaluations_model_status", "model_id", "status"),
        Index("ix_evaluations_schema_status", "schema_id", "status"),
        Index("ix_evaluations_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<Evaluation {self.id} model={self.model_id} schema={self.schema_id} [{self.status}]>"
        )
