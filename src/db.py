"""SQLAlchemy async engine and session factory (psycopg3 driver)."""

from collections.abc import AsyncGenerator

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from settings import Settings

logger = structlog.stdlib.get_logger(__name__)


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""


_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


async def init_db(settings: Settings) -> None:
    """Create the async engine and session factory.

    Must be called once during application startup (inside the lifespan
    context manager).
    """
    global _engine, _session_factory  # noqa: PLW0603

    _engine = create_async_engine(
        settings.database_url,
        echo=False,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
    )
    _session_factory = async_sessionmaker(
        bind=_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    # Import ORM models so metadata is populated, then create tables.
    # create_all is idempotent (skips existing tables) and serves as a
    # safety net for fresh databases before Alembic migrations run.
    import models.db  # noqa: F401

    try:
        async with _engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    except Exception:
        logger.debug("table_creation_skipped_likely_race_condition", exc_info=True)

    # Best-effort bootstrap stamp for fresh DBs (local/test, or any path
    # without a migrate init-container). Under a restricted no-DDL runtime role
    # a *missing* alembic_version would make the CREATE raise permission-denied
    # — swallow it (log) rather than crashloop startup, deferring to alembic
    # (`alembic upgrade head`) as the schema source of truth.
    try:
        async with _engine.begin() as conn:
            result = await conn.execute(
                text(
                    "SELECT EXISTS ("
                    "  SELECT 1 FROM information_schema.tables "
                    "  WHERE table_name = 'alembic_version'"
                    ")"
                )
            )
            has_alembic = result.scalar()
            if not has_alembic:
                await conn.execute(
                    text(
                        "CREATE TABLE alembic_version ("
                        "  version_num VARCHAR(32) NOT NULL, "
                        "  CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num)"
                        ")"
                    )
                )
                await conn.execute(
                    text(
                        "INSERT INTO alembic_version (version_num) "
                        "VALUES ('001_add_scoring_profile')"
                    )
                )
                logger.info("alembic_version_stamped", revision="001_add_scoring_profile")
    except Exception:
        logger.warning("alembic_version_stamp_skipped", exc_info=True)


async def get_session() -> AsyncGenerator[AsyncSession]:
    """FastAPI dependency that yields an async session and closes it after use."""
    if _session_factory is None:
        raise RuntimeError("Database not initialised - call init_db() first")
    async with _session_factory() as session:
        yield session


async def get_engine_for_check() -> bool:
    """Execute a trivial query to verify the database connection is alive."""
    if _engine is None:
        return False
    try:
        async with _engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception:
        logger.debug("health_check_failed", exc_info=True)
        return False


async def dispose_engine() -> None:
    """Dispose of the engine pool on shutdown."""
    global _engine, _session_factory  # noqa: PLW0603
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
