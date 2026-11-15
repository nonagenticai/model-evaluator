"""Tests for the db module.

Covers:
- init_db: creates engine, session factory, sets globals, stamps alembic
- get_session: yields a real async session, raises if not initialized
- get_engine_for_check: verifies connection via real engine
- dispose_engine: disposes engine and resets globals

Uses mocks for init_db because it hard-codes pool_size/max_overflow
which are PostgreSQL-only.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.ext.asyncio import AsyncSession


def _make_mock_engine():
    """Build a mock engine with two separate begin() contexts.

    init_db calls engine.begin() twice:
      1. create_all (run_sync)
      2. alembic stamp (execute with information_schema check)
    Each call gets its own connection mock so assertions stay clear.
    """
    mock_conn_create = AsyncMock()
    mock_conn_create.run_sync = AsyncMock()
    mock_conn_create.execute = AsyncMock()

    mock_begin_create = AsyncMock()
    mock_begin_create.__aenter__ = AsyncMock(return_value=mock_conn_create)
    mock_begin_create.__aexit__ = AsyncMock(return_value=False)

    # The alembic-stamp connection returns "no alembic_version table"
    # so the stamp branch executes CREATE TABLE + INSERT.
    mock_scalar_result = MagicMock()
    mock_scalar_result.scalar.return_value = False

    mock_conn_stamp = AsyncMock()
    mock_conn_stamp.execute = AsyncMock(return_value=mock_scalar_result)

    mock_begin_stamp = AsyncMock()
    mock_begin_stamp.__aenter__ = AsyncMock(return_value=mock_conn_stamp)
    mock_begin_stamp.__aexit__ = AsyncMock(return_value=False)

    mock_engine = MagicMock()
    mock_engine.begin = MagicMock(side_effect=[mock_begin_create, mock_begin_stamp])

    return mock_engine, mock_conn_create, mock_conn_stamp


class TestInitDb:
    """Tests for init_db().

    init_db hard-codes pool_size and max_overflow (PostgreSQL-only), so
    these tests mock create_async_engine to verify the function sets the
    module globals, calls create_all, and stamps Alembic.
    """

    async def test_init_db_sets_engine_and_factory(self) -> None:
        """init_db should set _engine and _session_factory globals."""
        import db as db_module
        from settings import Settings

        orig_engine = db_module._engine
        orig_factory = db_module._session_factory

        try:
            db_module._engine = None
            db_module._session_factory = None

            settings = Settings(
                database_url="postgresql+psycopg://user:pass@localhost/testdb"  # pragma: allowlist secret
            )

            mock_engine, mock_conn_create, mock_conn_stamp = _make_mock_engine()

            with patch("db.create_async_engine", return_value=mock_engine):
                await db_module.init_db(settings)

            assert db_module._engine is mock_engine
            assert db_module._session_factory is not None

            # Verify create_all was called via run_sync
            mock_conn_create.run_sync.assert_called_once()

            # Verify alembic stamp: information_schema check + CREATE TABLE + INSERT
            assert mock_conn_stamp.execute.call_count == 3
        finally:
            db_module._engine = orig_engine
            db_module._session_factory = orig_factory

    async def test_init_db_skips_stamp_when_alembic_exists(self) -> None:
        """init_db should not stamp if alembic_version table already exists."""
        import db as db_module
        from settings import Settings

        orig_engine = db_module._engine
        orig_factory = db_module._session_factory

        try:
            db_module._engine = None
            db_module._session_factory = None

            settings = Settings(
                database_url="postgresql+psycopg://user:pass@localhost/testdb"  # pragma: allowlist secret
            )

            mock_conn_create = AsyncMock()
            mock_conn_create.run_sync = AsyncMock()

            mock_begin_create = AsyncMock()
            mock_begin_create.__aenter__ = AsyncMock(return_value=mock_conn_create)
            mock_begin_create.__aexit__ = AsyncMock(return_value=False)

            # alembic_version already exists
            mock_scalar_result = MagicMock()
            mock_scalar_result.scalar.return_value = True

            mock_conn_stamp = AsyncMock()
            mock_conn_stamp.execute = AsyncMock(return_value=mock_scalar_result)

            mock_begin_stamp = AsyncMock()
            mock_begin_stamp.__aenter__ = AsyncMock(return_value=mock_conn_stamp)
            mock_begin_stamp.__aexit__ = AsyncMock(return_value=False)

            mock_engine = MagicMock()
            mock_engine.begin = MagicMock(side_effect=[mock_begin_create, mock_begin_stamp])

            with patch("db.create_async_engine", return_value=mock_engine):
                await db_module.init_db(settings)

            # Only the information_schema check, no CREATE/INSERT
            assert mock_conn_stamp.execute.call_count == 1
        finally:
            db_module._engine = orig_engine
            db_module._session_factory = orig_factory

    async def test_init_db_tolerates_stamp_ddl_denied(self) -> None:
        """init_db must not crashloop when the alembic_version stamp is denied DDL.

        Under a restricted (no-DDL) runtime role, `CREATE TABLE alembic_version`
        raises permission-denied. The stamp is best-effort — init_db swallows it
        and returns normally, deferring schema authority to `alembic upgrade head`.
        """
        import db as db_module
        from settings import Settings

        orig_engine = db_module._engine
        orig_factory = db_module._session_factory

        try:
            db_module._engine = None
            db_module._session_factory = None

            settings = Settings(
                database_url="postgresql+psycopg://user:pass@localhost/testdb"  # pragma: allowlist secret
            )

            # create_all begin() succeeds; the stamp begin() raises permission-denied.
            mock_conn_create = AsyncMock()
            mock_conn_create.run_sync = AsyncMock()
            mock_begin_create = AsyncMock()
            mock_begin_create.__aenter__ = AsyncMock(return_value=mock_conn_create)
            mock_begin_create.__aexit__ = AsyncMock(return_value=False)

            mock_begin_stamp = AsyncMock()
            mock_begin_stamp.__aenter__ = AsyncMock(
                side_effect=ProgrammingError(
                    "CREATE TABLE alembic_version ...",
                    {},
                    Exception("permission denied for schema public"),
                )
            )
            mock_begin_stamp.__aexit__ = AsyncMock(return_value=False)

            mock_engine = MagicMock()
            mock_engine.begin = MagicMock(side_effect=[mock_begin_create, mock_begin_stamp])

            with patch("db.create_async_engine", return_value=mock_engine):
                # Must return normally despite the stamp raising permission-denied.
                await db_module.init_db(settings)

            assert db_module._engine is mock_engine
        finally:
            db_module._engine = orig_engine
            db_module._session_factory = orig_factory

    async def test_init_db_passes_pool_config(self) -> None:
        """init_db should configure pool_size and max_overflow."""
        import db as db_module
        from settings import Settings

        orig_engine = db_module._engine
        orig_factory = db_module._session_factory

        try:
            db_module._engine = None
            db_module._session_factory = None

            settings = Settings(
                database_url="postgresql+psycopg://user:pass@localhost/testdb"  # pragma: allowlist secret
            )

            mock_engine, _, _ = _make_mock_engine()

            with patch("db.create_async_engine", return_value=mock_engine) as mock_create:
                await db_module.init_db(settings)

            # Verify pool configuration was passed
            call_kwargs = mock_create.call_args[1]
            assert call_kwargs["pool_size"] == 10
            assert call_kwargs["max_overflow"] == 20
            assert call_kwargs["pool_pre_ping"] is True
        finally:
            db_module._engine = orig_engine
            db_module._session_factory = orig_factory


class TestGetSession:
    """Tests for get_session()."""

    async def test_get_session_raises_when_not_initialized(self) -> None:
        """get_session should raise RuntimeError if _session_factory is None."""
        import db as db_module

        orig_factory = db_module._session_factory
        try:
            db_module._session_factory = None

            with pytest.raises(RuntimeError, match="Database not initialised"):
                async for _ in db_module.get_session():
                    pass
        finally:
            db_module._session_factory = orig_factory

    async def test_get_session_yields_working_session(self, engine) -> None:
        """get_session should yield a real working session that can execute queries."""
        from sqlalchemy.ext.asyncio import async_sessionmaker

        import db as db_module

        orig_factory = db_module._session_factory
        try:
            db_module._session_factory = async_sessionmaker(
                bind=engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )

            sessions = []
            async for s in db_module.get_session():
                sessions.append(s)
                # Verify the session can execute a real query
                result = await s.execute(text("SELECT 1"))
                assert result.scalar() == 1

            assert len(sessions) == 1
            assert isinstance(sessions[0], AsyncSession)
        finally:
            db_module._session_factory = orig_factory


class TestGetEngineForCheck:
    """Tests for get_engine_for_check()."""

    async def test_returns_false_when_no_engine(self) -> None:
        """Should return False if _engine is None."""
        import db as db_module

        orig_engine = db_module._engine
        try:
            db_module._engine = None
            result = await db_module.get_engine_for_check()
            assert result is False
        finally:
            db_module._engine = orig_engine

    async def test_returns_true_with_real_engine(self, engine) -> None:
        """Should return True when a real working engine is available."""
        import db as db_module

        orig_engine = db_module._engine
        try:
            db_module._engine = engine
            result = await db_module.get_engine_for_check()
            assert result is True
        finally:
            db_module._engine = orig_engine

    async def test_returns_false_on_connection_error(self) -> None:
        """Should return False when connect() raises an exception."""
        import db as db_module

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(side_effect=Exception("Connection refused"))
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        mock_engine = MagicMock()
        mock_engine.connect = MagicMock(return_value=mock_cm)

        orig_engine = db_module._engine
        try:
            db_module._engine = mock_engine
            result = await db_module.get_engine_for_check()
            assert result is False
        finally:
            db_module._engine = orig_engine


class TestDisposeEngine:
    """Tests for dispose_engine()."""

    async def test_dispose_resets_globals(self) -> None:
        """dispose_engine should set _engine and _session_factory to None."""
        import db as db_module

        orig_engine = db_module._engine
        orig_factory = db_module._session_factory
        try:
            # Use a mock engine with an async dispose method
            mock_engine = MagicMock()
            mock_engine.dispose = AsyncMock()
            db_module._engine = mock_engine
            db_module._session_factory = MagicMock()

            await db_module.dispose_engine()

            assert db_module._engine is None
            assert db_module._session_factory is None
            mock_engine.dispose.assert_called_once()
        finally:
            db_module._engine = orig_engine
            db_module._session_factory = orig_factory

    async def test_dispose_noop_when_no_engine(self) -> None:
        """dispose_engine should be safe to call with no engine."""
        import db as db_module

        orig_engine = db_module._engine
        orig_factory = db_module._session_factory
        try:
            db_module._engine = None
            db_module._session_factory = None

            # Should not raise
            await db_module.dispose_engine()

            assert db_module._engine is None
            assert db_module._session_factory is None
        finally:
            db_module._engine = orig_engine
            db_module._session_factory = orig_factory

    async def test_dispose_makes_engine_unusable(self) -> None:
        """After dispose, the engine should no longer work for queries."""
        import db as db_module

        orig_engine = db_module._engine
        orig_factory = db_module._session_factory
        try:
            mock_engine = MagicMock()
            mock_engine.dispose = AsyncMock()
            db_module._engine = mock_engine
            db_module._session_factory = None

            await db_module.dispose_engine()

            # Engine reference is now None
            assert db_module._engine is None

            # get_engine_for_check should return False
            result = await db_module.get_engine_for_check()
            assert result is False
        finally:
            db_module._engine = orig_engine
            db_module._session_factory = orig_factory
