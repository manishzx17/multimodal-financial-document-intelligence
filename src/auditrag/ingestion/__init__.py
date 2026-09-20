"""Ingestion module for AuditRAG Phase 2.

Handles normalization of TAT-DQA documents into structured representations:
document_id -> pages -> page_number, text, blocks, bbox, words, page_image.
"""

from auditrag.ingestion.models import (
    NormalizedBlock,
    NormalizedDocument,
    NormalizedPage,
    NormalizedWords,
)
from auditrag.ingestion.pdf_parser import TATDQAParser
from auditrag.ingestion.pipeline import DocumentIngestionPipeline
from auditrag.ingestion.table_extractor import get_document_blocks

__all__ = [
    "NormalizedBlock",
    "NormalizedDocument",
    "NormalizedPage",
    "NormalizedWords",
    "TATDQAParser",
    "DocumentIngestionPipeline",
    "get_document_blocks",
]
