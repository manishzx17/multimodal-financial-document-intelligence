"""TAT-DQA raw document parser and asset associator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

from auditrag.config import PROJECT_ROOT, RAW_TEST_DIR
from auditrag.ingestion.models import (
    NormalizedBlock,
    NormalizedDocument,
    NormalizedPage,
    NormalizedWords,
)


class TATDQAParser:
    """Parses raw TAT-DQA document files and associates metadata, PDFs, and PNG page images."""

    def __init__(self, raw_test_dir: Union[str, Path] = RAW_TEST_DIR):
        self.raw_test_dir = Path(raw_test_dir)

    def parse_document(
        self,
        doc_uid: str,
        source: Optional[str] = None,
    ) -> NormalizedDocument:
        """Parse a raw TAT-DQA document into a NormalizedDocument.

        Associates the JSON layout tree with its vector PDF and rendered PNG page images.
        """
        json_path = self.raw_test_dir / f"{doc_uid}.json"
        pdf_path = self.raw_test_dir / f"{doc_uid}.pdf"

        if not json_path.exists():
            raise FileNotFoundError(f"TAT-DQA document JSON not found: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        pages_data = raw_data.get("pages", [])
        normalized_pages = []

        for page_idx, page_item in enumerate(pages_data, start=1):
            raw_blocks = page_item.get("blocks", [])
            normalized_blocks = []

            # Process and sort blocks by reading order
            sorted_raw_blocks = sorted(raw_blocks, key=lambda b: b.get("order", 0))

            for b in sorted_raw_blocks:
                raw_words = b.get("words", {})
                words_model = NormalizedWords(
                    word_list=raw_words.get("word_list", []),
                    bbox_list=raw_words.get("bbox_list", []),
                )
                block_model = NormalizedBlock(
                    uuid=b["uuid"],
                    text=b.get("text", ""),
                    bbox=b.get("bbox", [0, 0, 0, 0]),
                    order=b.get("order", 0),
                    words=words_model,
                )
                normalized_blocks.append(block_model)

            # Concatenate block texts for clean page text
            page_text = "\n".join(b.text for b in normalized_blocks if b.text)

            # Associate page image
            img_file = self.raw_test_dir / f"{doc_uid}_{page_idx}.png"
            img_rel_path: Optional[str] = None
            if img_file.exists():
                try:
                    img_rel_path = img_file.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
                except ValueError:
                    img_rel_path = img_file.as_posix()

            # Page-level words rollup (from all blocks in order)
            page_words = NormalizedWords(
                word_list=[w for b in normalized_blocks for w in b.words.word_list],
                bbox_list=[bb for b in normalized_blocks for bb in b.words.bbox_list],
            )

            page_bbox = page_item.get("bbox", [0, 0, 0, 0])

            page_model = NormalizedPage(
                page_number=page_idx,
                text=page_text,
                blocks=normalized_blocks,
                bbox=page_bbox,
                words=page_words,
                page_image=img_rel_path,
            )
            normalized_pages.append(page_model)

        # Compute relative PDF path if it exists
        pdf_rel_path: Optional[str] = None
        if pdf_path.exists():
            try:
                pdf_rel_path = pdf_path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
            except ValueError:
                pdf_rel_path = pdf_path.as_posix()

        return NormalizedDocument(
            document_id=doc_uid,
            source=source,
            pdf_path=pdf_rel_path,
            page_count=len(normalized_pages),
            pages=normalized_pages,
        )
