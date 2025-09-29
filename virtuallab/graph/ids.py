"""Utility helpers for generating identifiers and timestamps."""
from __future__ import annotations

import datetime as _dt
import uuid


def new_id(prefix: str) -> str:
    """Return a deterministic identifier with the provided ``prefix``."""

    return f"{prefix}_{uuid.uuid4().hex}"


def utc_now() -> str:
    """Return the current UTC time formatted as an ISO 8601 string."""

    return _dt.datetime.utcnow().replace(tzinfo=_dt.timezone.utc).isoformat()
