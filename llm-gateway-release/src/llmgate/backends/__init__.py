"""Backend adapters. All implement ``Backend`` (async chat, async streaming, health)."""

from llmgate.backends.base import Backend, BackendError
from llmgate.backends.fake import FakeBackend
from llmgate.backends.openai_compat import OpenAICompatBackend
from llmgate.backends.vllm import VLLMBackend

__all__ = ["Backend", "BackendError", "FakeBackend", "OpenAICompatBackend", "VLLMBackend"]
