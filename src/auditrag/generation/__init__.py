"""Generation module for AuditRAG Phase 7 (Multimodal Grounded VLM Answer Generation)."""

from auditrag.generation.prompt import SYSTEM_PROMPT, build_grounded_prompt
from auditrag.generation.vlm import (
    BaseVLMProvider,
    GeneratedAnswer,
    MockVLMProvider,
    OpenAIVLMProvider,
    VLMGenerator,
    get_vlm_provider,
)

__all__ = [
    "SYSTEM_PROMPT",
    "build_grounded_prompt",
    "BaseVLMProvider",
    "MockVLMProvider",
    "OpenAIVLMProvider",
    "GeneratedAnswer",
    "VLMGenerator",
    "get_vlm_provider",
]
