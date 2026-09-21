"""Tokenization utilities for lexical retrieval in AuditRAG."""

from __future__ import annotations

import re
from typing import List

_TOKEN_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\b|\b[a-zA-Z0-9]+\b")


def tokenize_text(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric tokens, preserving numbers, decimals, years, and acronyms."""
    if not text:
        return []
    return _TOKEN_PATTERN.findall(text.lower())
