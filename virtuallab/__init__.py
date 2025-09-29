"""VirtualLab package initialization.

This module exposes the primary entry point used by external callers to
interact with the virtual laboratory orchestration system.
"""

from .api import VirtualLab_tool

__all__ = ["VirtualLab_tool"]
