"""Data module for AuditRAG, providing loaders and schema representations."""

from auditrag.data.loader import (
    BoundingBox,
    Document,
    DocumentBlock,
    DocumentPage,
    EvidenceSpan,
    Question,
    TATDQADataset,
    TATDQALoader,
    WordsData,
    WordToken,
)

__all__ = [
    "BoundingBox",
    "WordToken",
    "WordsData",
    "DocumentBlock",
    "DocumentPage",
    "Document",
    "EvidenceSpan",
    "Question",
    "TATDQADataset",
    "TATDQALoader",
]
