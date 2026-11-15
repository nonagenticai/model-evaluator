"""Tests for the MetricsService.

Tests cover:
- Perfect scores (all metrics at 1.0)
- Zero scores (all metrics at 0.0)
- Mixed results
- Composite score formula verification
- Edge cases (single sample, all same latency)
- P95 latency computation
- Empty results handling
"""

import pytest

from services.metrics_service import MetricsService


@pytest.fixture()
def svc() -> MetricsService:
    return MetricsService()


class TestMetricsCompute:
    """Tests for MetricsService.compute()."""

    def test_perfect_scores(self, svc: MetricsService) -> None:
        """All samples pass all checks."""
        results = [
            {"json_valid": True, "schema_valid": True, "exact_match": True} for _ in range(10)
        ]
        latencies = [100.0] * 10

        metrics = svc.compute(results, latencies)

        assert metrics.json_parse_rate == 1.0
        assert metrics.schema_valid_rate == 1.0
        assert metrics.exact_match_rate == 1.0
        assert metrics.composite_score == 1.0
        assert metrics.total_samples == 10
        assert metrics.passed_samples == 10
        assert metrics.failed_samples == 0
        assert metrics.avg_latency_ms == 100.0
        assert metrics.p95_latency_ms == 100.0

    def test_zero_scores(self, svc: MetricsService) -> None:
        """All samples fail all checks."""
        results = [
            {"json_valid": False, "schema_valid": False, "exact_match": False} for _ in range(10)
        ]
        latencies = [200.0] * 10

        metrics = svc.compute(results, latencies)

        assert metrics.json_parse_rate == 0.0
        assert metrics.schema_valid_rate == 0.0
        assert metrics.exact_match_rate == 0.0
        assert metrics.composite_score == 0.0
        assert metrics.total_samples == 10
        assert metrics.passed_samples == 0
        assert metrics.failed_samples == 10

    def test_mixed_results(self, svc: MetricsService) -> None:
        """Some samples pass, some fail."""
        results = [
            {"json_valid": True, "schema_valid": True, "exact_match": True},
            {"json_valid": True, "schema_valid": True, "exact_match": False},
            {"json_valid": True, "schema_valid": False, "exact_match": False},
            {"json_valid": False, "schema_valid": False, "exact_match": False},
        ]
        latencies = [100.0, 150.0, 200.0, 300.0]

        metrics = svc.compute(results, latencies)

        assert metrics.json_parse_rate == 0.75  # 3/4
        assert metrics.schema_valid_rate == 0.50  # 2/4
        assert metrics.exact_match_rate == 0.25  # 1/4

        # composite = 0.15*0.75 + 0.35*0.50 + 0.50*0.25 = 0.1125 + 0.175 + 0.125 = 0.4125
        expected_composite = 0.15 * 0.75 + 0.35 * 0.50 + 0.50 * 0.25
        assert abs(metrics.composite_score - expected_composite) < 1e-9

        assert metrics.total_samples == 4
        assert metrics.passed_samples == 1
        assert metrics.failed_samples == 3

    def test_composite_score_formula(self, svc: MetricsService) -> None:
        """Verify the exact composite score weights."""
        results = [
            {"json_valid": True, "schema_valid": False, "exact_match": False},
        ] * 100

        latencies = [100.0] * 100

        metrics = svc.compute(results, latencies)

        # json_parse_rate = 1.0, schema_valid_rate = 0.0, exact_match_rate = 0.0
        expected = 0.15 * 1.0 + 0.35 * 0.0 + 0.50 * 0.0
        assert abs(metrics.composite_score - expected) < 1e-9
        assert metrics.composite_score == 0.15

    def test_gbnf_grammar_scenario(self, svc: MetricsService) -> None:
        """With GBNF grammar, json_parse_rate should be 100%.

        Focus shifts to schema_valid_rate and exact_match_rate.
        """
        # GBNF guarantees JSON validity but not semantic correctness
        results = [
            {"json_valid": True, "schema_valid": True, "exact_match": True},
            {"json_valid": True, "schema_valid": True, "exact_match": True},
            {"json_valid": True, "schema_valid": True, "exact_match": False},
            {"json_valid": True, "schema_valid": False, "exact_match": False},
            {"json_valid": True, "schema_valid": False, "exact_match": False},
        ]
        latencies = [120.0, 130.0, 140.0, 150.0, 160.0]

        metrics = svc.compute(results, latencies)

        assert metrics.json_parse_rate == 1.0  # GBNF guarantees this
        assert metrics.schema_valid_rate == 0.6  # 3/5
        assert metrics.exact_match_rate == 0.4  # 2/5

        expected = 0.15 * 1.0 + 0.35 * 0.6 + 0.50 * 0.4
        assert abs(metrics.composite_score - expected) < 1e-9

    def test_single_sample(self, svc: MetricsService) -> None:
        """Handle edge case of a single sample."""
        results = [
            {"json_valid": True, "schema_valid": True, "exact_match": True},
        ]
        latencies = [42.0]

        metrics = svc.compute(results, latencies)

        assert metrics.total_samples == 1
        assert metrics.composite_score == 1.0
        assert metrics.avg_latency_ms == 42.0
        assert metrics.p95_latency_ms == 42.0

    def test_empty_results_raises(self, svc: MetricsService) -> None:
        """Empty results should raise ValueError."""
        with pytest.raises(ValueError, match="No results"):
            svc.compute([], [])

    def test_latency_statistics(self, svc: MetricsService) -> None:
        """Verify avg and P95 latency computation."""
        results = [
            {"json_valid": True, "schema_valid": True, "exact_match": True} for _ in range(20)
        ]
        # Latencies from 10 to 200 in steps of 10
        latencies = [float(i * 10) for i in range(1, 21)]

        metrics = svc.compute(results, latencies)

        # Average: sum(10..200)/20 = 2100/20 = 105
        assert abs(metrics.avg_latency_ms - 105.0) < 1e-9

        # P95: index = int(0.95*20) = 19, but min(19, 19) = 19 -> sorted[19] = 200
        assert metrics.p95_latency_ms == 200.0

    def test_p95_latency_out_of_order(self, svc: MetricsService) -> None:
        """P95 should handle unsorted latency input."""
        results = [
            {"json_valid": True, "schema_valid": True, "exact_match": True} for _ in range(100)
        ]
        # Unsorted latencies
        latencies = [float(100 - i) for i in range(100)]

        metrics = svc.compute(results, latencies)

        # Sorted: [1.0, 2.0, ..., 100.0]
        # P95 index = min(int(0.95*100), 99) = min(95, 99) = 95 -> sorted[95] = 96.0
        assert metrics.p95_latency_ms == 96.0

    def test_p95_latency_skewed_distribution(self, svc: MetricsService) -> None:
        """With a skewed latency distribution (long right tail), p95 must
        clearly exceed the mean and point at the 95th percentile slot --
        verifies p95 is not just returning the mean."""
        results = [
            {"json_valid": True, "schema_valid": True, "exact_match": True} for _ in range(10)
        ]
        # Heavy right-tail distribution -- most fast, a few very slow
        latencies = [50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 150.0, 200.0, 500.0, 1000.0]

        metrics = svc.compute(results, latencies)

        # Mean = (50+60+70+80+90+100+150+200+500+1000)/10 = 2300/10 = 230
        assert abs(metrics.avg_latency_ms - 230.0) < 1e-9
        # Sorted is already ascending; p95 index = min(int(0.95*10), 9) = min(9, 9) = 9
        # so p95 = sorted[9] = 1000.0
        assert metrics.p95_latency_ms == 1000.0
        # Sanity: p95 must be strictly greater than mean for a right-skewed dist
        assert metrics.p95_latency_ms > metrics.avg_latency_ms

    def test_threshold_boundary_above(self, svc: MetricsService) -> None:
        """Composite score exactly at threshold (0.85) should pass."""
        # We need: 0.15*json + 0.35*schema + 0.50*exact = 0.85
        # json=1.0, schema=1.0, exact=0.7 -> 0.15 + 0.35 + 0.35 = 0.85
        total = 100
        exact_count = 70

        results = []
        for i in range(total):
            results.append(
                {
                    "json_valid": True,
                    "schema_valid": True,
                    "exact_match": i < exact_count,
                }
            )
        latencies = [100.0] * total

        metrics = svc.compute(results, latencies)

        assert abs(metrics.composite_score - 0.85) < 1e-9
        # At exactly threshold, >= should be True
        assert metrics.composite_score >= 0.85

    def test_threshold_boundary_below(self, svc: MetricsService) -> None:
        """Composite score just below threshold should not pass."""
        # json=1.0, schema=1.0, exact=0.69 -> 0.15 + 0.35 + 0.345 = 0.845
        total = 100
        exact_count = 69

        results = []
        for i in range(total):
            results.append(
                {
                    "json_valid": True,
                    "schema_valid": True,
                    "exact_match": i < exact_count,
                }
            )
        latencies = [100.0] * total

        metrics = svc.compute(results, latencies)

        expected = 0.15 * 1.0 + 0.35 * 1.0 + 0.50 * 0.69
        assert abs(metrics.composite_score - expected) < 1e-9
        assert metrics.composite_score < 0.85

    def test_high_latency_does_not_affect_composite(self, svc: MetricsService) -> None:
        """Composite score is based on accuracy, not latency."""
        results = [
            {"json_valid": True, "schema_valid": True, "exact_match": True} for _ in range(10)
        ]
        # Very high latencies
        latencies = [5000.0] * 10

        metrics = svc.compute(results, latencies)

        assert metrics.composite_score == 1.0
        assert metrics.avg_latency_ms == 5000.0
        assert metrics.p95_latency_ms == 5000.0

    def test_structural_scoring_profile_perfect(self, svc: MetricsService) -> None:
        """Structural profile with perfect scores should still equal 1.0."""
        results = [
            {"json_valid": True, "schema_valid": True, "exact_match": True} for _ in range(10)
        ]
        latencies = [100.0] * 10

        metrics = svc.compute(results, latencies, scoring_profile="structural")

        assert metrics.composite_score == 1.0

    def test_structural_scoring_profile_weights(self, svc: MetricsService) -> None:
        """Structural profile: 0.20*json + 0.60*schema + 0.20*exact_match."""
        results = [
            {"json_valid": True, "schema_valid": True, "exact_match": False},
        ] * 10
        latencies = [100.0] * 10

        metrics = svc.compute(results, latencies, scoring_profile="structural")

        # json=1.0, schema=1.0, exact=0.0
        # structural: 0.20*1.0 + 0.60*1.0 + 0.20*0.0 = 0.80
        expected = 0.20 * 1.0 + 0.60 * 1.0 + 0.20 * 0.0
        assert abs(metrics.composite_score - expected) < 1e-9
        assert metrics.composite_score == 0.8

    def test_structural_vs_default_different_scores(self, svc: MetricsService) -> None:
        """Same data should produce different composite scores for different profiles."""
        results = [
            {"json_valid": True, "schema_valid": True, "exact_match": False},
        ] * 10
        latencies = [100.0] * 10

        default_metrics = svc.compute(results, latencies, scoring_profile="default")
        structural_metrics = svc.compute(results, latencies, scoring_profile="structural")

        # default: 0.15*1.0 + 0.35*1.0 + 0.50*0.0 = 0.50
        # structural: 0.20*1.0 + 0.60*1.0 + 0.20*0.0 = 0.80
        assert abs(default_metrics.composite_score - 0.50) < 1e-9
        assert abs(structural_metrics.composite_score - 0.80) < 1e-9
        assert structural_metrics.composite_score > default_metrics.composite_score

    def test_structural_profile_schema_heavy(self, svc: MetricsService) -> None:
        """Structural profile should give higher score when schema_valid is high but exact is low."""
        results = [
            {"json_valid": True, "schema_valid": True, "exact_match": False},
            {"json_valid": True, "schema_valid": True, "exact_match": False},
            {"json_valid": True, "schema_valid": False, "exact_match": False},
            {"json_valid": True, "schema_valid": False, "exact_match": False},
        ]
        latencies = [100.0] * 4

        metrics = svc.compute(results, latencies, scoring_profile="structural")

        # json=1.0, schema=0.5, exact=0.0
        # structural: 0.20*1.0 + 0.60*0.5 + 0.20*0.0 = 0.50
        expected = 0.20 * 1.0 + 0.60 * 0.5 + 0.20 * 0.0
        assert abs(metrics.composite_score - expected) < 1e-9
