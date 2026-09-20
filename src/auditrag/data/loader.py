"""TAT-DQA dataset loader and internal schema representation.

This module provides Pydantic models representing the TAT-DQA multimodal financial
document structure, ground-truth questions, answers, derivations, and bounding-box
coordinate mappings.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field


class BoundingBox(BaseModel):
    """Represents a 2D bounding box coordinate [x0, y0, x1, y1] in pixel/point space."""

    x0: float
    y0: float
    x1: float
    y1: float

    model_config = ConfigDict(frozen=True)

    @classmethod
    def from_list(cls, coords: List[Union[int, float]]) -> BoundingBox:
        """Create a BoundingBox from a 4-element list [x0, y0, x1, y1]."""
        if len(coords) != 4:
            raise ValueError(f"BoundingBox requires exactly 4 coordinates, got {len(coords)}: {coords}")
        return cls(x0=float(coords[0]), y0=float(coords[1]), x1=float(coords[2]), y1=float(coords[3]))

    def to_list(self) -> List[float]:
        """Convert to list format [x0, y0, x1, y1]."""
        return [self.x0, self.y0, self.x1, self.y1]

    @property
    def width(self) -> float:
        """Width of the bounding box."""
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        """Height of the bounding box."""
        return max(0.0, self.y1 - self.y0)

    @property
    def area(self) -> float:
        """Surface area of the bounding box."""
        return self.width * self.height

    def scale(self, scale_x: float, scale_y: float) -> BoundingBox:
        """Return a scaled bounding box."""
        return BoundingBox(
            x0=self.x0 * scale_x,
            y0=self.y0 * scale_y,
            x1=self.x1 * scale_x,
            y1=self.y1 * scale_y,
        )

    def normalize(self, page_width: float, page_height: float) -> BoundingBox:
        """Normalize coordinates to [0, 1] range relative to page dimensions."""
        if page_width <= 0 or page_height <= 0:
            raise ValueError(f"Page dimensions must be positive, got {page_width}x{page_height}")
        return BoundingBox(
            x0=max(0.0, min(1.0, self.x0 / page_width)),
            y0=max(0.0, min(1.0, self.y0 / page_height)),
            x1=max(0.0, min(1.0, self.x1 / page_width)),
            y1=max(0.0, min(1.0, self.y1 / page_height)),
        )


class WordToken(BaseModel):
    """Represents a tokenized word and its bounding box."""

    text: str
    bbox: BoundingBox


class WordsData(BaseModel):
    """Raw word-level tokens and coordinates within a block."""

    word_list: List[str] = Field(default_factory=list)
    bbox_list: List[List[int]] = Field(default_factory=list)

    def get_tokens(self) -> List[WordToken]:
        """Convert word and bbox lists into a list of WordToken objects."""
        tokens = []
        for word, bbox in zip(self.word_list, self.bbox_list):
            tokens.append(WordToken(text=word, bbox=BoundingBox.from_list(bbox)))
        return tokens


class DocumentBlock(BaseModel):
    """Represents an OCR/layout block (paragraph, heading, or table row) in a page."""

    uuid: str
    text: str
    bbox: List[int]
    order: int
    words: WordsData = Field(default_factory=WordsData)

    @property
    def bounding_box(self) -> BoundingBox:
        """Return BoundingBox model for the block."""
        return BoundingBox.from_list(self.bbox)

    def get_bboxes_for_char_span(self, start: int, end: int) -> List[BoundingBox]:
        """Find the word-level bounding boxes intersecting a character span [start, end].

        This enables exact visual citation mapping from TAT-DQA block_mapping spans
        to high-resolution document coordinates.
        """
        # Clamp start and end to valid text bounds (handling annotator edge-case start=-1)
        safe_start = max(0, start)
        safe_end = min(len(self.text), max(safe_start, end))

        if safe_start >= safe_end or not self.words.word_list:
            return [self.bounding_box]

        # Reconstruct word character spans in block text
        matched_bboxes: List[BoundingBox] = []
        current_search_idx = 0

        for word, w_bbox in zip(self.words.word_list, self.words.bbox_list):
            word_start = self.text.find(word, current_search_idx)
            if word_start == -1:
                # Fallback if whitespace or punctuation mismatch occurs
                word_start = current_search_idx
            word_end = word_start + len(word)
            current_search_idx = word_end

            # Check overlap between [word_start, word_end] and [safe_start, safe_end]
            if max(word_start, safe_start) < min(word_end, safe_end):
                matched_bboxes.append(BoundingBox.from_list(w_bbox))

        return matched_bboxes if matched_bboxes else [self.bounding_box]


class DocumentPage(BaseModel):
    """Represents a single document page with its dimensions, blocks, and image path."""

    page_idx: int
    bbox: List[int]
    blocks: List[DocumentBlock] = Field(default_factory=list)
    image_path: Optional[Path] = None

    @property
    def bounding_box(self) -> BoundingBox:
        """Return BoundingBox model for the page dimensions."""
        return BoundingBox.from_list(self.bbox)

    @property
    def width(self) -> float:
        """Page width in coordinate space."""
        return self.bounding_box.width

    @property
    def height(self) -> float:
        """Page height in coordinate space."""
        return self.bounding_box.height

    @property
    def full_text(self) -> str:
        """Concatenate all block texts on this page in reading order."""
        sorted_blocks = sorted(self.blocks, key=lambda b: b.order)
        return "\n".join(b.text for b in sorted_blocks if b.text)

    @property
    def text(self) -> str:
        """Alias for full_text to match Phase 2 normalized schema."""
        return self.full_text

    @property
    def page_number(self) -> int:
        """Alias for page_idx to match Phase 2 normalized schema."""
        return self.page_idx

    @property
    def page_image(self) -> Optional[str]:
        """Path string to page image to match Phase 2 normalized schema."""
        return str(self.image_path) if self.image_path else None

    def get_block(self, uuid: str) -> Optional[DocumentBlock]:
        """Find a block on this page by UUID."""
        for b in self.blocks:
            if b.uuid == uuid:
                return b
        return None


class Document(BaseModel):
    """Represents a complete TAT-DQA document entity."""

    uid: str
    source: Optional[str] = None
    annotated_page: Optional[int] = None
    pdf_path: Optional[Path] = None
    json_path: Optional[Path] = None
    pages: List[DocumentPage] = Field(default_factory=list)

    @property
    def document_id(self) -> str:
        """Alias for uid to match Phase 2 normalized schema."""
        return self.uid

    @property
    def page_count(self) -> int:
        """Total pages in the document."""
        return len(self.pages)

    def get_page(self, page_idx: int) -> Optional[DocumentPage]:
        """Retrieve a page by 1-indexed page index."""
        for p in self.pages:
            if p.page_idx == page_idx:
                return p
        return None

    def find_block(self, block_uuid: str) -> Optional[Tuple[DocumentPage, DocumentBlock]]:
        """Search across all pages to find a block by UUID."""
        for page in self.pages:
            block = page.get_block(block_uuid)
            if block is not None:
                return page, block
        return None

    @property
    def full_text(self) -> str:
        """Concatenate full text across all pages."""
        return "\n\n".join(f"--- Page {p.page_idx} ---\n{p.full_text}" for p in self.pages)

    def to_normalized(self):
        """Convert to Phase 2 NormalizedDocument representation."""
        from auditrag.ingestion.models import (
            NormalizedBlock,
            NormalizedDocument,
            NormalizedPage,
            NormalizedWords,
        )

        norm_pages = []
        for p in self.pages:
            norm_blocks = [
                NormalizedBlock(
                    uuid=b.uuid,
                    text=b.text,
                    bbox=b.bbox,
                    order=b.order,
                    words=NormalizedWords(
                        word_list=b.words.word_list,
                        bbox_list=b.words.bbox_list,
                    ),
                )
                for b in p.blocks
            ]
            page_words = NormalizedWords(
                word_list=[w for b in norm_blocks for w in b.words.word_list],
                bbox_list=[bb for b in norm_blocks for bb in b.words.bbox_list],
            )
            norm_pages.append(
                NormalizedPage(
                    page_number=p.page_idx,
                    text=p.full_text,
                    blocks=norm_blocks,
                    bbox=p.bbox,
                    words=page_words,
                    page_image=str(p.image_path) if p.image_path else None,
                )
            )

        return NormalizedDocument(
            document_id=self.uid,
            source=self.source,
            pdf_path=str(self.pdf_path) if self.pdf_path else None,
            page_count=len(norm_pages),
            pages=norm_pages,
        )



class EvidenceSpan(BaseModel):
    """Represents a resolved evidence annotation linked to a document block and coordinates."""

    block_uuid: str
    start: int
    end: int
    text: Optional[str] = None
    page_idx: Optional[int] = None
    bboxes: List[BoundingBox] = Field(default_factory=list)


class Question(BaseModel):
    """Represents a question and its optional ground truth annotations."""

    uid: str
    doc_uid: str
    order: int
    question: str
    # Gold benchmark annotations (None if evaluating unlabelled test split)
    answer: Optional[Union[List[str], int, float, str]] = None
    derivation: Optional[str] = None
    answer_type: Optional[str] = None
    scale: Optional[str] = None
    req_comparison: Optional[bool] = False
    facts: List[str] = Field(default_factory=list)
    block_mapping: List[Dict[str, List[int]]] = Field(default_factory=list)

    def resolve_evidence_spans(self, doc: Optional[Document] = None) -> List[EvidenceSpan]:
        """Resolve block_mapping entries to concrete EvidenceSpan objects with coordinates."""
        spans: List[EvidenceSpan] = []
        for mapping in self.block_mapping:
            for buuid, offsets in mapping.items():
                start, end = offsets[0], offsets[1]
                safe_start = max(0, start)
                extracted_text = None
                page_idx = None
                bboxes: List[BoundingBox] = []

                if doc is not None:
                    lookup = doc.find_block(buuid)
                    if lookup is not None:
                        page, block = lookup
                        page_idx = page.page_idx
                        safe_end = min(len(block.text), max(safe_start, end))
                        extracted_text = block.text[safe_start:safe_end]
                        bboxes = block.get_bboxes_for_char_span(start, end)

                spans.append(
                    EvidenceSpan(
                        block_uuid=buuid,
                        start=start,
                        end=end,
                        text=extracted_text,
                        page_idx=page_idx,
                        bboxes=bboxes,
                    )
                )
        return spans


class TATDQADataset(BaseModel):
    """Container for the TAT-DQA evaluation corpus with indexing and fast lookups."""

    documents: Dict[str, Document] = Field(default_factory=dict)
    questions: List[Question] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_document(self, doc_uid: str) -> Optional[Document]:
        """Lookup document by UID."""
        return self.documents.get(doc_uid)

    def get_question(self, question_uid: str) -> Optional[Question]:
        """Lookup question by question UID."""
        for q in self.questions:
            if q.uid == question_uid:
                return q
        return None

    def get_questions_for_doc(self, doc_uid: str) -> List[Question]:
        """Lookup all questions associated with a given document."""
        return [q for q in self.questions if q.doc_uid == doc_uid]

    def summary_stats(self) -> Dict[str, Any]:
        """Generate high-level summary statistics of the loaded dataset."""
        answer_types: Dict[str, int] = {}
        scales: Dict[str, int] = {}
        for q in self.questions:
            atype = q.answer_type or "unknown"
            answer_types[atype] = answer_types.get(atype, 0) + 1
            scale = q.scale if q.scale is not None else "unknown"
            scales[scale] = scales.get(scale, 0) + 1

        total_pages = sum(d.page_count for d in self.documents.values())
        return {
            "total_documents": len(self.documents),
            "total_pages": total_pages,
            "total_questions": len(self.questions),
            "answer_types": answer_types,
            "scales": scales,
        }


class TATDQALoader:
    """Loader for the TAT-DQA dataset located in data/raw/tatdqa."""

    def __init__(self, data_dir: Union[str, Path] = "data/raw/tatdqa"):
        self.data_dir = Path(data_dir)
        self.test_dir = self.data_dir / "test"
        self.test_json_path = self.data_dir / "tatdqa_dataset_test.json"
        self.test_gold_path = self.data_dir / "tatdqa_dataset_test_gold.json"

    def load_document(self, doc_uid: str, source: Optional[str] = None, annotated_page: Optional[int] = None) -> Document:
        """Load a single document and all its pages/blocks from the test directory."""
        json_path = self.test_dir / f"{doc_uid}.json"
        pdf_path = self.test_dir / f"{doc_uid}.pdf"

        if not json_path.exists():
            raise FileNotFoundError(f"Document JSON not found: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        pages: List[DocumentPage] = []
        for idx, p_data in enumerate(raw_data.get("pages", []), 1):
            blocks: List[DocumentBlock] = []
            for b_data in p_data.get("blocks", []):
                words_raw = b_data.get("words", {})
                words = WordsData(
                    word_list=words_raw.get("word_list", []),
                    bbox_list=words_raw.get("bbox_list", []),
                )
                block = DocumentBlock(
                    uuid=b_data["uuid"],
                    text=b_data.get("text", ""),
                    bbox=b_data.get("bbox", [0, 0, 0, 0]),
                    order=b_data.get("order", 0),
                    words=words,
                )
                blocks.append(block)

            img_path = self.test_dir / f"{doc_uid}_{idx}.png"
            page = DocumentPage(
                page_idx=idx,
                bbox=p_data.get("bbox", [0, 0, 0, 0]),
                blocks=blocks,
                image_path=img_path if img_path.exists() else None,
            )
            pages.append(page)

        return Document(
            uid=doc_uid,
            source=source,
            annotated_page=annotated_page,
            pdf_path=pdf_path if pdf_path.exists() else None,
            json_path=json_path,
            pages=pages,
        )

    def load_questions(self, use_gold: bool = True) -> Tuple[Dict[str, Dict[str, Any]], List[Question]]:
        """Load questions from either the gold benchmark or unlabelled test split."""
        target_path = self.test_gold_path if use_gold else self.test_json_path
        if not target_path.exists():
            raise FileNotFoundError(f"Question file not found: {target_path}")

        with open(target_path, "r", encoding="utf-8") as f:
            raw_list = json.load(f)

        doc_meta: Dict[str, Dict[str, Any]] = {}
        questions: List[Question] = []

        for item in raw_list:
            d_info = item.get("doc", {})
            d_uid = d_info.get("uid")
            if d_uid:
                doc_meta[d_uid] = d_info

            for q_data in item.get("questions", []):
                q = Question(
                    uid=q_data["uid"],
                    doc_uid=d_uid,
                    order=q_data.get("order", 0),
                    question=q_data.get("question", ""),
                    answer=q_data.get("answer"),
                    derivation=q_data.get("derivation"),
                    answer_type=q_data.get("answer_type"),
                    scale=q_data.get("scale"),
                    req_comparison=q_data.get("req_comparison", False),
                    facts=q_data.get("facts", []),
                    block_mapping=q_data.get("block_mapping", []),
                )
                questions.append(q)

        return doc_meta, questions

    def load_dataset(self, use_gold: bool = True, load_doc_details: bool = True) -> TATDQADataset:
        """Load the complete dataset into the unified TATDQADataset container."""
        doc_meta, questions = self.load_questions(use_gold=use_gold)
        documents: Dict[str, Document] = {}

        if load_doc_details:
            for d_uid, meta in doc_meta.items():
                doc = self.load_document(
                    doc_uid=d_uid,
                    source=meta.get("source"),
                    annotated_page=meta.get("page"),
                )
                documents[d_uid] = doc

        return TATDQADataset(documents=documents, questions=questions)
