"""Configuration helpers for loading environment variables.

This module ensures that variables defined in a project-level ``.env`` file
are loaded before attempting to access them.  Consumers should rely on the
``get_env`` helper instead of using :func:`os.getenv` directly so that the
configuration is loaded in a single, well-defined place.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


@lru_cache(maxsize=1)
def _load_environment() -> None:
    """Load environment variables from the project's ``.env`` file.

    The loader first attempts to read ``.env`` from the repository root.  If the
    file does not exist we still call :func:`load_dotenv` to allow the default
    discovery mechanism to run (e.g., for users who store the file elsewhere).
    Subsequent calls are cached so the file is only read once per process.
    """

    project_root = Path(__file__).resolve().parents[1]
    env_path = project_root / ".env"

    if env_path.exists():
        load_dotenv(env_path, override=False)
    else:
        load_dotenv(override=False)


def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """Return the value for ``key`` from the environment.

    Parameters
    ----------
    key:
        The name of the environment variable to look up.
    default:
        The value to return when ``key`` is not present.
    """

    _load_environment()
    return os.environ.get(key, default)


__all__ = ["get_env"]

