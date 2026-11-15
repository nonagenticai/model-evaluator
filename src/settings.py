"""Application settings using Pydantic BaseSettings with env_prefix."""

from functools import lru_cache
from typing import ClassVar

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Model Evaluator service configuration.

    All settings can be overridden via environment variables prefixed with
    ``MODEL_EVALUATOR_``.  For example, ``MODEL_EVALUATOR_PORT=9000`` overrides
    the default port.

    Environment-specific values (service URLs, DB DSNs, OTLP endpoint) carry
    empty-string placeholders here; real values are injected from
    ``gitops-production/clusters/production/services/model-evaluator/deployment.yaml``
    at deploy time. The ``model_post_init`` validator fails fast in
    production when any required field is blank.
    """

    model_config = SettingsConfigDict(
        env_prefix="MODEL_EVALUATOR_",
        case_sensitive=False,
    )

    service_name: str = "model-evaluator"
    port: int = Field(default=8154, description="HTTP listen port")
    database_url: str = Field(
        default="",
        description="PostgreSQL DSN (psycopg3 driver)",
    )
    data_collector_url: str = Field(
        default="",
        description="URL of the data-collector service",
    )
    model_registry_url: str = Field(
        default="",
        description="URL of the model-registry service",
    )
    adapter_manager_url: str = Field(
        default="",
        description="URL of the adapter-manager sidecar for loading LoRA adapters",
    )
    default_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Default composite score threshold for auto-approval",
    )
    inference_endpoint_url: str = Field(
        default="",
        description="Default inference endpoint for health checks (routed via inference-router)",
    )
    inference_timeout: float = Field(
        default=30.0,
        ge=1.0,
        description="Timeout in seconds for inference requests",
    )
    otlp_traces_endpoint: str = Field(
        default="",
        description="OTLP HTTP endpoint for trace export",
    )
    environment: str = Field(default="production", description="Deployment environment tag")
    log_level: str = Field(default="INFO", description="Root log level")
    log_format: str = Field(
        default="json",
        description="Log format: 'json' for production, 'console' for local dev",
    )

    _REQUIRED_IN_PRODUCTION: ClassVar[tuple[str, ...]] = (
        "database_url",
        "data_collector_url",
        "model_registry_url",
        "adapter_manager_url",
        "inference_endpoint_url",
        "otlp_traces_endpoint",
    )

    def model_post_init(self, __context: object) -> None:
        """Normalize database URL + validate required fields in production."""
        if self.database_url.startswith("postgresql+asyncpg://"):
            self.database_url = self.database_url.replace(
                "postgresql+asyncpg://", "postgresql+psycopg://", 1
            )
        elif self.database_url.startswith("postgresql://"):
            self.database_url = self.database_url.replace(
                "postgresql://", "postgresql+psycopg://", 1
            )
        if self.environment == "production":
            missing = [n for n in self._REQUIRED_IN_PRODUCTION if not getattr(self, n)]
            if missing:
                raise RuntimeError(
                    f"model-evaluator settings missing required fields in production: {missing}. "
                    "Set them via env vars on the Deployment (see gitops-production)."
                )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
