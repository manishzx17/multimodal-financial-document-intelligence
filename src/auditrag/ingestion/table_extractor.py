"""Table and block preservation module for AuditRAG.

Preserves existing TAT-DQA block and tabular information as-is without
lossy parsing or premature table reconstruction.
"""

from __future__ import annotations

from typing import List

from auditrag.ingestion.models import NormalizedBlock, NormalizedDocument


def get_document_blocks(doc: NormalizedDocument) -> List[NormalizedBlock]:
    """Retrieve all normalized blocks across all pages in reading order."""
    blocks: List[NormalizedBlock] = []
    for page in doc.pages:
        blocks.extend(page.blocks)
    return blocks
