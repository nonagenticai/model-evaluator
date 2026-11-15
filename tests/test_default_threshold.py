"""Tests for Settings.default_threshold wiring in EvalService.

Verifies that:
- When no threshold is provided in the request, the service uses Settings.default_threshold
- When a threshold IS provided in the request, it overrides the Settings default
"""

from unittest.mock import AsyncMock
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from models.domain import EvalCreateRequest
from services.collector_client import CollectorClient
from services.eval_service import EvalService
from services.inference_client import InferenceClient
from services.metrics_service import MetricsService
from services.registry_client import RegistryClient
from settings import Settings


def _make_eval_service(
    settings: Settings,
    mock_collector: CollectorClient | None = None,
    mock_inference: InferenceClient | None = None,
    mock_registry: RegistryClient | None = None,
) -> EvalService:
    """Build an EvalService with the given settings and mock clients."""
    return EvalService(
        collector_client=mock_collector or AsyncMock(spec=CollectorClient),
        inference_client=mock_inference or AsyncMock(spec=InferenceClient),
        metrics_service=MetricsService(),
        registry_client=mock_registry or AsyncMock(spec=RegistryClient),
        settings=settings,
    )


class TestDefaultThresholdFallback:
    """Tests that EvalService falls back to Settings.default_threshold when request omits threshold."""

    async def test_uses_default_threshold_when_request_omits_threshold(
        self,
        session: AsyncSession,
    ) -> None:
        """When no threshold in the request, the evaluation should use Settings.default_threshold."""
        custom_threshold = 0.92
        settings = Settings(default_threshold=custom_threshold)
        svc = _make_eval_service(settings)

        request = EvalCreateRequest(
            model_id=uuid4(),
            schema_id="test_schema",
            endpoint="http://localhost:8080",
            # threshold intentionally omitted -- should fall back to settings
        )

        evaluation = await svc.create_evaluation(session, request)
        assert evaluation.threshold == custom_threshold

    async def test_uses_default_threshold_0_70(
        self,
        session: AsyncSession,
    ) -> None:
        """Verify a non-default Settings.default_threshold value is used."""
        settings = Settings(default_threshold=0.70)
        svc = _make_eval_service(settings)

        request = EvalCreateRequest(
            model_id=uuid4(),
            schema_id="test_schema",
            endpoint="http://localhost:8080",
        )

        evaluation = await svc.create_evaluation(session, request)
        assert evaluation.threshold == 0.70

    async def test_settings_default_matches_settings_class_default(
        self,
        session: AsyncSession,
    ) -> None:
        """When Settings uses its own default (0.85), the evaluation should get 0.85."""
        settings = Settings()  # default_threshold=0.85
        svc = _make_eval_service(settings)

        request = EvalCreateRequest(
            model_id=uuid4(),
            schema_id="test_schema",
            endpoint="http://localhost:8080",
        )

        evaluation = await svc.create_evaluation(session, request)
        assert evaluation.threshold == 0.85


class TestRequestThresholdOverridesDefault:
    """Tests that an explicit threshold in the request overrides Settings.default_threshold."""

    async def test_request_threshold_overrides_settings(
        self,
        session: AsyncSession,
    ) -> None:
        """When threshold IS provided in the request, it should be used instead of the settings default."""
        settings = Settings(default_threshold=0.92)
        svc = _make_eval_service(settings)

        request = EvalCreateRequest(
            model_id=uuid4(),
            schema_id="test_schema",
            endpoint="http://localhost:8080",
            threshold=0.75,
        )

        evaluation = await svc.create_evaluation(session, request)
        assert evaluation.threshold == 0.75

    async def test_request_threshold_zero_overrides_settings(
        self,
        session: AsyncSession,
    ) -> None:
        """A threshold of 0.0 in the request should be used, not treated as missing."""
        settings = Settings(default_threshold=0.92)
        svc = _make_eval_service(settings)

        request = EvalCreateRequest(
            model_id=uuid4(),
            schema_id="test_schema",
            endpoint="http://localhost:8080",
            threshold=0.0,
        )

        evaluation = await svc.create_evaluation(session, request)
        assert evaluation.threshold == 0.0

    async def test_request_threshold_one_overrides_settings(
        self,
        session: AsyncSession,
    ) -> None:
        """A threshold of 1.0 in the request should be used."""
        settings = Settings(default_threshold=0.50)
        svc = _make_eval_service(settings)

        request = EvalCreateRequest(
            model_id=uuid4(),
            schema_id="test_schema",
            endpoint="http://localhost:8080",
            threshold=1.0,
        )

        evaluation = await svc.create_evaluation(session, request)
        assert evaluation.threshold == 1.0
