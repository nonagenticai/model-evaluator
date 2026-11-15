"""Baseline migration: create evaluations table with all columns.

Replaces the previous scoring_profile-only migration. This captures the
full schema so that Alembic owns the DDL going forward. Existing databases
that already have the table are handled via an existence check.

Revision ID: 001_add_scoring_profile
Revises:
Create Date: 2026-03-15
"""

import sqlalchemy as sa
from alembic import op

revision = "001_add_scoring_profile"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    # Skip if the table already exists (existing databases).
    result = conn.execute(
        sa.text(
            "SELECT EXISTS ("
            "  SELECT 1 FROM information_schema.tables "
            "  WHERE table_name = 'evaluations'"
            ")"
        )
    )
    if result.scalar():
        # Table exists -- ensure scoring_profile column is present
        # (handles databases created before this column was added).
        col_result = conn.execute(
            sa.text(
                "SELECT EXISTS ("
                "  SELECT 1 FROM information_schema.columns "
                "  WHERE table_name = 'evaluations' "
                "  AND column_name = 'scoring_profile'"
                ")"
            )
        )
        if not col_result.scalar():
            op.add_column(
                "evaluations",
                sa.Column(
                    "scoring_profile",
                    sa.String(32),
                    nullable=False,
                    server_default="default",
                ),
            )
        return

    op.create_table(
        "evaluations",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("model_id", sa.Uuid(), nullable=False),
        sa.Column("schema_id", sa.String(256), nullable=False),
        sa.Column("status", sa.String(32), nullable=False, server_default="pending"),
        sa.Column("endpoint", sa.String(1024), nullable=False),
        sa.Column("metrics", sa.JSON(), nullable=True),
        sa.Column("threshold", sa.Float(), nullable=False, server_default="0.85"),
        sa.Column("max_samples", sa.Integer(), nullable=False, server_default="100"),
        sa.Column("auto_approved", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("review_notes", sa.Text(), nullable=True),
        sa.Column("sample_results", sa.JSON(), nullable=False, server_default="'[]'"),
        sa.Column("json_schema", sa.JSON(), nullable=True),
        sa.Column("gbnf_grammar", sa.Text(), nullable=True),
        sa.Column("scoring_profile", sa.String(32), nullable=False, server_default="default"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
    )

    # Indexes matching EvaluationORM.__table_args__
    op.create_index("ix_evaluations_model_id", "evaluations", ["model_id"])
    op.create_index("ix_evaluations_schema_id", "evaluations", ["schema_id"])
    op.create_index("ix_evaluations_status", "evaluations", ["status"])
    op.create_index("ix_evaluations_model_status", "evaluations", ["model_id", "status"])
    op.create_index("ix_evaluations_schema_status", "evaluations", ["schema_id", "status"])
    op.create_index("ix_evaluations_created_at", "evaluations", ["created_at"])


def downgrade() -> None:
    op.drop_table("evaluations")
