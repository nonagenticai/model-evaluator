"""Cross-version uuid7() shim.

PEP 770 added ``uuid.uuid7`` in Python 3.14. This service targets 3.13 (spaCy
has no 3.14 wheels yet — see Plan 3 Lane A python-version decision), so we
import from the ``uuid_utils.compat`` backport, which returns stdlib
``uuid.UUID`` instances so callsites stay drop-in compatible.

When the service moves to 3.14, swap this single import to ``from uuid``.
"""

from __future__ import annotations

from uuid_utils.compat import uuid7

__all__ = ["uuid7"]
