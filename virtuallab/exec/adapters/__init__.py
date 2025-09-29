"""Adapters bridging VirtualLab with external execution engines."""

from .engineer import EngineerAdapter
from .local import LocalFuncAdapter
from .openai_model import OpenAIModelAdapter

__all__ = [
    "EngineerAdapter",
    "LocalFuncAdapter",
    "OpenAIModelAdapter",
]
