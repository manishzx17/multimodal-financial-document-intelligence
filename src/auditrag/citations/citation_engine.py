"""Visual citation engine linking VLM answers to page bounding boxes for AuditRAG Phase 8."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

from auditrag.citations.bounding_box import find_text_boxes_on_page
from auditrag.citations.visualizer import draw_citation_overlay
from auditrag.config import CITATIONS_DIR, PROCESSED_DOCUMENTS_DIR, PROJECT_ROOT
from auditrag.generation.vlm import GeneratedAnswer
from auditrag.ingestion.models import NormalizedDocument
from auditrag.retrieval.hybrid import HybridRetrievalResult


class VisualCitation(BaseModel):
    """Represents a visual bounding-box citation on a document page image."""

    document_id: str
    page_number: int
    text: str
    bbox: List[int]
    word_bboxes: List[List[int]] = Field(default_factory=list)
    block_uuid: Optional[str] = None
    label: Optional[str] = None
    source_image_path: str
    overlay_image_path: Optional[str] = None

    model_config = ConfigDict(extra="ignore")


class AnswerWithCitations(BaseModel):
    """Generated answer paired with visual citations and rendered overlay images."""

    question: str
    answer: str
    citations: List[VisualCitation] = Field(default_factory=list)
    overlay_images: List[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")


class VisualCitationEngine:
    """Orchestrates extraction of visual citations and page overlays for generated answers."""

    def __init__(self, citations_dir: Optional[Path] = None):
        self.citations_dir = citations_dir or CITATIONS_DIR
        self.citations_dir.mkdir(parents=True, exist_ok=True)
        self._doc_cache: Dict[str, NormalizedDocument] = {}

    def _load_document(self, doc_id: str) -> Optional[NormalizedDocument]:
        """Load and cache processed normalized document."""
        if doc_id in self._doc_cache:
            return self._doc_cache[doc_id]

        doc_path = PROCESSED_DOCUMENTS_DIR / f"{doc_id}.json"
        if not doc_path.exists():
            return None

        doc = NormalizedDocument.from_file(doc_path)
        self._doc_cache[doc_id] = doc
        return doc

    def create_citations_for_answer(
        self,
        answer: GeneratedAnswer,
        retrieval_results: List[HybridRetrievalResult],
        claims: Optional[List[Any]] = None,
    ) -> AnswerWithCitations:
        """Link generated answer facts to bounding boxes and render visual citation overlays."""
        # Extract atomic claims if not explicitly passed
        if claims is None:
            try:
                from auditrag.verification.claim_extractor import extract_claims
                claim_objs = extract_claims(answer.answer)
            except Exception:
                claim_objs = []
        else:
            claim_objs = claims

        citations: List[VisualCitation] = []
        page_to_citations: Dict[str, List[VisualCitation]] = {}
        citation_counter = 1

        # 1. Claim-to-Evidence Block Alignment
        aligned_matched = False
        if claim_objs:
            for claim in claim_objs:
                doc_id = getattr(claim, "cited_doc_id", None)
                page_num = getattr(claim, "cited_page_number", None)

                # Fallback to top retrieval doc/page if not in citation
                if (not doc_id or not page_num) and retrieval_results:
                    doc_id = retrieval_results[0].document_id
                    page_num = retrieval_results[0].page_number

                if not doc_id or not page_num:
                    continue

                doc = self._load_document(doc_id)
                if not doc:
                    continue

                page = doc.get_page(page_num)
                if not page or not page.page_image:
                    continue

                image_path = page.page_image
                page_key = f"{doc_id}_{page_num}"

                # Find best matching block on page for this specific claim
                best_block = None
                best_score = 0
                claim_stmt = getattr(claim, "statement", "")
                claim_nums = getattr(claim, "extracted_numbers", [])
                claim_words = [w.lower() for w in re.findall(r"\b[a-zA-Z]{3,}\b", claim_stmt)]

                for b in page.blocks:
                    score = 0
                    b_text_lower = b.text.lower()
                    for num in claim_nums:
                        n_str = f"{num:g}"
                        n_comma = f"{int(num):,}" if num.is_integer() else f"{num:,.2f}"
                        if n_str in b.text or n_comma in b.text:
                            score += 10
                    for w in claim_words:
                        if w in b_text_lower:
                            score += 1
                    if score > best_score:
                        best_score = score
                        best_block = b

                if best_block:
                    # Extract word-level bounding boxes for claim numbers in this block
                    block_hits: List[Dict[str, Any]] = []
                    for num in claim_nums:
                        num_tokens = [f"{int(num):,}" if num.is_integer() else f"{num:,.2f}", f"{num:g}"]
                        for tok in num_tokens:
                            from auditrag.citations.bounding_box import find_word_boxes_in_block
                            hit = find_word_boxes_in_block(best_block, [tok])
                            if hit and hit not in block_hits:
                                block_hits.append(hit)
                                break

                    # If no word boxes hit, use enclosing block bbox
                    if not block_hits:
                        block_hits.append({
                            "bbox": best_block.bbox,
                            "word_bboxes": best_block.words.bbox_list if best_block.words else [best_block.bbox],
                            "text": best_block.text[:80],
                            "block_uuid": best_block.uuid,
                        })

                    for hit in block_hits:
                        cit = VisualCitation(
                            document_id=doc_id,
                            page_number=page_num,
                            text=hit["text"],
                            bbox=hit["bbox"],
                            word_bboxes=hit.get("word_bboxes", []),
                            block_uuid=hit.get("block_uuid"),
                            label=f"[{citation_counter}]",
                            source_image_path=image_path,
                        )
                        citations.append(cit)
                        page_to_citations.setdefault(page_key, []).append(cit)
                        citation_counter += 1
                        aligned_matched = True

        # 2. Fallback to generic token matching if no claim blocks were matched
        if not aligned_matched:
            cited_pairs = re.findall(r"\[Doc:\s*([a-f0-9]{32}),\s*Page:\s*(\d+)\]", answer.answer)
            if not cited_pairs and retrieval_results:
                top_res = retrieval_results[0]
                cited_pairs = [(top_res.document_id, str(top_res.page_number))]

            number_tokens = re.findall(r"\$?\d+(?:,\d+)*(?:\.\d+)?%?", answer.answer)

            for doc_id, page_str in cited_pairs:
                page_num = int(page_str)
                doc = self._load_document(doc_id)
                if not doc:
                    continue

                page = doc.get_page(page_num)
                if not page or not page.page_image:
                    continue

                image_path = page.page_image
                page_key = f"{doc_id}_{page_num}"

                matched_for_page: List[Dict[str, Any]] = []
                for token in number_tokens:
                    clean_num = token.replace("$", "").replace("%", "").strip()
                    if clean_num and len(clean_num) >= 2:
                        hits = find_text_boxes_on_page(page, clean_num)
                        for hit in hits:
                            if hit not in matched_for_page:
                                matched_for_page.append(hit)

                if not matched_for_page:
                    for r in retrieval_results:
                        if r.document_id == doc_id and r.page_number == page_num and r.source_metadata.get("bboxes"):
                            for bbox in r.source_metadata["bboxes"][:2]:
                                matched_for_page.append({
                                    "bbox": bbox,
                                    "word_bboxes": [bbox],
                                    "text": r.chunk_text[:80] if r.chunk_text else "Evidence block",
                                    "block_uuid": r.source_metadata.get("block_uuids", [None])[0],
                                })
                            break

                for match in matched_for_page:
                    cit = VisualCitation(
                        document_id=doc_id,
                        page_number=page_num,
                        text=match["text"],
                        bbox=match["bbox"],
                        word_bboxes=match.get("word_bboxes", []),
                        block_uuid=match.get("block_uuid"),
                        label=f"[{citation_counter}]",
                        source_image_path=image_path,
                    )
                    citations.append(cit)
                    page_to_citations.setdefault(page_key, []).append(cit)
                    citation_counter += 1

        # 3. Render visual overlay images for pages with citations
        overlay_images: List[str] = []
        for page_key, page_cits in page_to_citations.items():
            if not page_cits:
                continue

            src_img = page_cits[0].source_image_path
            out_img = self.citations_dir / f"{page_key}_overlay.png"

            try:
                saved_path = draw_citation_overlay(
                    image_path=src_img,
                    citations=page_cits,
                    output_path=out_img,
                )
                rel_path = str(saved_path)
                try:
                    rel_path = str(saved_path.resolve().relative_to(PROJECT_ROOT.resolve()))
                except ValueError:
                    pass

                overlay_images.append(rel_path)
                for cit in page_cits:
                    cit.overlay_image_path = rel_path
            except Exception as e:
                print(f"Warning: Failed to render visual citation overlay for {page_key}: {e}")

        return AnswerWithCitations(
            question=answer.question,
            answer=answer.answer,
            citations=citations,
            overlay_images=overlay_images,
        )
