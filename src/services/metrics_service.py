"""Evaluation metrics computation.

Computes the suite of evaluation metrics from per-sample inference results
and produces a composite score used for the auto-approval gate.

Scoring profiles:
  default:    composite = 0.15 * json_parse + 0.35 * schema_valid + 0.50 * exact_match
  structural: composite = 0.20 * json_parse + 0.60 * schema_valid + 0.20 * exact_match
"""

from typing import Literal

import structlog

from models.domain import EvalMetrics

logger = structlog.stdlib.get_logger(__name__)

ScoringProfile = Literal["default", "structural"]

_WEIGHTS: dict[ScoringProfile, tuple[float, float, float]] = {
    "default": (0.15, 0.35, 0.50),
    "structural": (0.20, 0.60, 0.20),
}


class MetricsService:
    """Stateless service that computes evaluation metrics from result sets."""

    def compute(
        self,
        results: list[dict],
        latencies: list[float],
        scoring_profile: ScoringProfile = "default",
    ) -> EvalMetrics:
        """Compute evaluation metrics from individual sample results.

        Args:
            results: List of per-sample result dicts, each containing:
                - json_valid (bool)
                - schema_valid (bool)
                - exact_match (bool)
            latencies: List of per-sample latency values in milliseconds.
            scoring_profile: Scoring profile to use for composite score.
                "default" emphasizes exact_match (0.50 weight).
                "structural" emphasizes schema_valid (0.60 weight),
                suitable for schemas with large/variable outputs where
                exact match is impractical.

        Returns:
            Computed EvalMetrics with all rates, latency stats, and composite score.

        Raises:
            ValueError: If results list is empty.
        """
        total = len(results)
        if total == 0:
            raise ValueError("No results to compute metrics from")

        json_parse_count = sum(1 for r in results if r["json_valid"])
        schema_valid_count = sum(1 for r in results if r["schema_valid"])
        exact_match_count = sum(1 for r in results if r["exact_match"])

        sorted_latencies = sorted(latencies)
        p95_idx = min(int(0.95 * len(sorted_latencies)), len(sorted_latencies) - 1)

        json_parse_rate = json_parse_count / total
        schema_valid_rate = schema_valid_count / total
        exact_match_rate = exact_match_count / total

        # Composite: weighted average using selected profile
        w_json, w_schema, w_exact = _WEIGHTS[scoring_profile]
        composite = (
            json_parse_rate * w_json + schema_valid_rate * w_schema + exact_match_rate * w_exact
        )

        avg_latency = sum(latencies) / total
        p95_latency = sorted_latencies[p95_idx] if sorted_latencies else 0.0

        logger.info(
            "metrics_computed",
            total_samples=total,
            scoring_profile=scoring_profile,
            json_parse_rate=round(json_parse_rate, 4),
            schema_valid_rate=round(schema_valid_rate, 4),
            exact_match_rate=round(exact_match_rate, 4),
            composite_score=round(composite, 4),
            avg_latency_ms=round(avg_latency, 2),
            p95_latency_ms=round(p95_latency, 2),
        )

        return EvalMetrics(
            json_parse_rate=json_parse_rate,
            schema_valid_rate=schema_valid_rate,
            exact_match_rate=exact_match_rate,
            avg_latency_ms=avg_latency,
            p95_latency_ms=p95_latency,
            composite_score=composite,
            total_samples=total,
            passed_samples=exact_match_count,
            failed_samples=total - exact_match_count,
        )
