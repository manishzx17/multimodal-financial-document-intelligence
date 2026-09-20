"""Normalized document models for Phase 2 Document Ingestion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class NormalizedWords(BaseModel):
    """Word-level tokens and coordinates within a block."""

    word_list: List[str] = Field(default_factory=list)
    bbox_list: List[List[int]] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")


class NormalizedBlock(BaseModel):
    """Represents a text/layout block preserving reading order, words, and coordinates."""

    uuid: str
    text: str
    bbox: List[int]
    order: int
    words: NormalizedWords = Field(default_factory=NormalizedWords)

    model_config = ConfigDict(extra="ignore")


class NormalizedPage(BaseModel):
    """Represents a normalized document page."""

    page_number: int
    text: str
    blocks: List[NormalizedBlock] = Field(default_factory=list)
    bbox: List[int] = Field(default_factory=lambda: [0, 0, 0, 0])
    words: Optional[NormalizedWords] = None
    page_image: Optional[str] = None

    model_config = ConfigDict(extra="ignore")

    @property
    def page_idx(self) -> int:
        """Alias for compatibility with Phase 1 DocumentPage."""
        return self.page_number

    @property
    def image_path(self) -> Optional[str]:
        """Alias for compatibility with Phase 1 DocumentPage."""
        return self.page_image

    def get_block(self, block_uuid: str) -> Optional[NormalizedBlock]:
        """Lookup a block by UUID on this page."""
        for b in self.blocks:
            if b.uuid == block_uuid:
                return b
        return None


class NormalizedDocument(BaseModel):
    """Normalized document representation for AuditRAG.

    Structure:
    document_id -> pages -> page_number, text, blocks, bbox, words, page_image.
    """

    document_id: str
    source: Optional[str] = None
    pdf_path: Optional[str] = None
    page_count: int
    pages: List[NormalizedPage] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")

    @property
    def uid(self) -> str:
        """Alias for compatibility with Phase 1 Document."""
        return self.document_id

    @property
    def full_text(self) -> str:
        """Concatenated text of all pages in reading order."""
        return "\n\n".join(f"--- Page {p.page_number} ---\n{p.text}" for p in self.pages)

    def get_page(self, page_number: int) -> Optional[NormalizedPage]:
        """Retrieve a page by 1-indexed page number."""
        for p in self.pages:
            if p.page_number == page_number:
                return p
        return None

    def find_block(self, block_uuid: str) -> Optional[tuple[NormalizedPage, NormalizedBlock]]:
        """Search across all pages to find a block by UUID."""
        for page in self.pages:
            block = page.get_block(block_uuid)
            if block is not None:
                return page, block
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize model to a dictionary representation."""
        return self.model_dump()

    def to_json(self, indent: int = 2) -> str:
        """Serialize model to a formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save_to_file(self, filepath: Union[str, Path], indent: int = 2) -> Path:
        """Write normalized document JSON to disk."""
        target_path = Path(filepath)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(self.to_json(indent=indent))
        return target_path

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> NormalizedDocument:
        """Construct NormalizedDocument from dictionary."""
        return cls.model_validate(data)

    @classmethod
    def from_file(cls, filepath: Union[str, Path]) -> NormalizedDocument:
        """Load NormalizedDocument from disk JSON file."""
        target_path = Path(filepath)
        if not target_path.exists():
            raise FileNotFoundError(f"Document file not found: {target_path}")
        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
