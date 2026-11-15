"""Settings refuse to start in production with empty URL placeholders."""

import pytest

from settings import Settings


def _mk(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "database_url": "postgresql+psycopg://u:p@h:6432/model_evaluator",  # pragma: allowlist secret
        "data_collector_url": "http://dc:8151",
        "model_registry_url": "http://mr:8155",
        "adapter_manager_url": "http://v:8001",
        "inference_endpoint_url": "http://inf:8150/api/chat/completions",
        "otlp_traces_endpoint": "http://alloy:4318/v1/traces",
        "environment": "production",
    }
    base.update(overrides)
    return Settings.model_validate(base)


class TestProductionValidation:
    def test_complete_config_allows_init(self) -> None:
        settings = _mk()
        assert settings.environment == "production"

    @pytest.mark.parametrize(
        "empty_field",
        [
            "database_url",
            "data_collector_url",
            "model_registry_url",
            "adapter_manager_url",
            "inference_endpoint_url",
            "otlp_traces_endpoint",
        ],
    )
    def test_empty_required_raises(self, empty_field: str) -> None:
        with pytest.raises(RuntimeError, match=empty_field):
            _mk(**{empty_field: ""})

    def test_test_environment_allows_empty(self) -> None:
        Settings.model_validate({"environment": "test"})
