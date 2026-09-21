"""Chunking module for AuditRAG Phase 3.

Implements structured block-window chunking that preserves document, page,
block UUID, coordinate, and page-image metadata.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

from auditrag.ingestion.models import NormalizedBlock, NormalizedDocument, NormalizedPage


class TextChunk(BaseModel):
    """Represents a text chunk created from one or more structured document blocks."""

    chunk_id: str
    document_id: str
    page_number: int
    text: str
    block_uuids: List[str] = Field(default_factory=list)
    bboxes: List[List[int]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="ignore")


def chunk_page(
    page: NormalizedPage,
    document_id: str,
    source: Optional[str] = None,
    min_chars: int = 300,
    max_chars: int = 600,
) -> List[TextChunk]:
    """Chunk a single normalized page using structured block-windowing (300-600 characters).

    Preserves block boundaries so financial table rows and statements are kept intact.
    """
    chunks: List[TextChunk] = []
    current_blocks: List[NormalizedBlock] = []
    current_len = 0
    chunk_index = 0

    sorted_blocks = sorted(page.blocks, key=lambda b: b.order)

    def _flush_window():
        nonlocal chunk_index, current_blocks, current_len
        if not current_blocks:
            return

        chunk_text = "\n".join(b.text.strip() for b in current_blocks if b.text.strip())
        if chunk_text:
            block_uuids = [b.uuid for b in current_blocks]
            bboxes = [b.bbox for b in current_blocks]
            min_order = min(b.order for b in current_blocks)
            max_order = max(b.order for b in current_blocks)

            meta: Dict[str, Any] = {
                "source": source,
                "page_image": page.page_image,
                "order_range": [min_order, max_order],
                "block_count": len(current_blocks),
            }

            chunk = TextChunk(
                chunk_id=f"{document_id}_p{page.page_number}_c{chunk_index}",
                document_id=document_id,
                page_number=page.page_number,
                text=chunk_text,
                block_uuids=block_uuids,
                bboxes=bboxes,
                metadata=meta,
            )
            chunks.append(chunk)
            chunk_index += 1

        current_blocks = []
        current_len = 0

    for block in sorted_blocks:
        b_text = block.text.strip()
        if not b_text:
            continue

        b_len = len(b_text)

        # If adding this block stays within max_chars
        if current_len + b_len <= max_chars:
            current_blocks.append(block)
            current_len += b_len + 1  # newline
        else:
            # If current window already has sufficient characters (>= min_chars)
            if current_len >= min_chars:
                _flush_window()
                current_blocks.append(block)
                current_len = b_len
            else:
                # If window has some blocks but would exceed max_chars
                if current_blocks:
                    _flush_window()
                    current_blocks.append(block)
                    current_len = b_len
                else:
                    # Single block is larger than max_chars, keep it whole
                    current_blocks.append(block)
                    current_len = b_len
                    _flush_window()

    _flush_window()
    return chunks


def chunk_document(
    doc: NormalizedDocument,
    min_chars: int = 300,
    max_chars: int = 600,
) -> List[TextChunk]:
    """Chunk all pages of a normalized document."""
    all_chunks: List[TextChunk] = []
    for page in doc.pages:
        page_chunks = chunk_page(
            page=page,
            document_id=doc.document_id,
            source=doc.source,
            min_chars=min_chars,
            max_chars=max_chars,
        )
        all_chunks.extend(page_chunks)
    return all_chunks
