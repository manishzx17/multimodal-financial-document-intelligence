"""Unit and integration tests for Phase 8 Visual Bounding-Box Citations."""

from pathlib import Path
from PIL import Image
import pytest

from auditrag.citations import (
    AnswerWithCitations,
    VisualCitation,
    VisualCitationEngine,
    draw_citation_overlay,
    find_text_boxes_on_page,
    merge_bboxes,
)
from auditrag.config import PROCESSED_DOCUMENTS_DIR, RAW_TEST_DIR
from auditrag.generation.vlm import GeneratedAnswer
from auditrag.ingestion.models import NormalizedDocument
from auditrag.retrieval.hybrid import HybridRetrievalResult


@pytest.fixture
def sample_doc() -> NormalizedDocument:
    doc_path = PROCESSED_DOCUMENTS_DIR / "637fab7088ea6c78a5dba55f17e833bd.json"
    if not doc_path.exists():
        pytest.skip(f"Processed document not found: {doc_path}")
    return NormalizedDocument.from_file(doc_path)


def test_merge_bboxes():
    """Verify enclosing bounding box calculation."""
    boxes = [[10, 20, 50, 60], [30, 40, 100, 120]]
    merged = merge_bboxes(boxes)
    assert merged == [10, 20, 100, 120]
    assert merge_bboxes([]) == [0, 0, 0, 0]


def test_find_text_boxes_on_page(sample_doc: NormalizedDocument):
    """Verify locating word-level and block-level bounding boxes for numerical and text tokens."""
    page = sample_doc.pages[0]

    # Test numerical token lookup
    hits = find_text_boxes_on_page(page, "21,224")
    assert len(hits) > 0
    hit = hits[0]
    assert "21,224" in hit["text"]
    assert len(hit["bbox"]) == 4
    x0, y0, x1, y1 = hit["bbox"]
    assert x0 < x1 and y0 < y1
    assert hit["block_uuid"] is not None

    # Test phrase lookup
    phrase_hits = find_text_boxes_on_page(page, "Plexus Corp")
    assert len(phrase_hits) > 0
    assert "Plexus" in phrase_hits[0]["text"]


def test_draw_citation_overlay(tmp_path: Path):
    """Verify drawing highlighted bounding boxes and saving overlay PNG."""
    test_img = RAW_TEST_DIR / "637fab7088ea6c78a5dba55f17e833bd_1.png"
    if not test_img.exists():
        pytest.skip("Test image not found.")

    citation = VisualCitation(
        document_id="637fab7088ea6c78a5dba55f17e833bd",
        page_number=1,
        text="21,224",
        bbox=[200, 300, 350, 340],
        label="[1]",
        source_image_path=str(test_img),
    )

    out_file = tmp_path / "test_overlay.png"
    saved_path = draw_citation_overlay(test_img, [citation], output_path=out_file)

    assert saved_path.exists()
    assert saved_path.stat().st_size > 0

    # Verify image dimensions match original
    with Image.open(test_img) as orig_img, Image.open(saved_path) as out_img:
        assert orig_img.size == out_img.size
        assert out_img.format == "PNG"


def test_visual_citation_engine(sample_doc: NormalizedDocument, tmp_path: Path):
    """Verify end-to-end citation linking and overlay generation for a generated answer."""
    engine = VisualCitationEngine(citations_dir=tmp_path / "citations")

    doc_id = sample_doc.document_id
    answer = GeneratedAnswer(
        question="What was the depreciation for AMER in 2018?",
        answer=f"In 2018, depreciation for AMER was $21,224 [Doc: {doc_id}, Page: 1].",
        sources=[{"document_id": doc_id, "page_number": 1}],
        model="mock-vlm-v1",
        provider="MockVLMProvider",
        evidence_count=1,
    )

    retrieval_results = [
        HybridRetrievalResult(
            document_id=doc_id,
            page_number=1,
            score=0.035,
            sources=["dense", "bm25"],
            chunk_id="chunk_1",
            chunk_text="Depreciation: AMER $ 21,224",
            source_metadata={"bboxes": [[100, 200, 300, 250]]},
            image_path=sample_doc.pages[0].page_image,
        )
    ]

    result = engine.create_citations_for_answer(answer, retrieval_results)

    assert isinstance(result, AnswerWithCitations)
    assert result.question == answer.question
    assert result.answer == answer.answer
    assert len(result.citations) > 0

    # Verify citation fields
    assert any("21,224" in c.text for c in result.citations)
    cit = result.citations[0]
    assert cit.document_id == doc_id
    assert cit.page_number == 1
    assert len(cit.bbox) == 4
    assert cit.overlay_image_path is not None

    # Verify overlay image exists on disk
    assert len(result.overlay_images) == 1
    overlay_path = Path(result.overlay_images[0])
    if not overlay_path.is_absolute():
        from auditrag.config import PROJECT_ROOT
        overlay_path = PROJECT_ROOT / overlay_path
    assert overlay_path.exists()


def test_claim_to_block_alignment(sample_doc: NormalizedDocument, tmp_path: Path):
    """Verify claim-to-evidence alignment accurately matches claims to their originating page blocks."""
    from auditrag.verification.models import Claim, ClaimType
    engine = VisualCitationEngine(citations_dir=tmp_path / "citations")
    doc_id = sample_doc.document_id

    claim = Claim(
        claim_id="claim_1",
        statement="Depreciation for AMER was $21,224",
        claim_type=ClaimType.NUMERICAL,
        extracted_numbers=[21224.0],
        cited_doc_id=doc_id,
        cited_page_number=1,
    )

    answer = GeneratedAnswer(
        question="What was depreciation for AMER?",
        answer=f"Depreciation for AMER was $21,224 [Doc: {doc_id}, Page: 1].",
        sources=[{"document_id": doc_id, "page_number": 1}],
        model="mock-vlm-v1",
        provider="MockVLMProvider",
        evidence_count=1,
    )

    result = engine.create_citations_for_answer(answer, [], claims=[claim])
    assert len(result.citations) >= 1
    first_cit = result.citations[0]
    assert "21,224" in first_cit.text
    assert first_cit.block_uuid is not None

    # Verify that the block containing this citation actually contains the number 21,224
    page = sample_doc.get_page(1)
    matched_block = next((b for b in page.blocks if b.uuid == first_cit.block_uuid), None)
    assert matched_block is not None
    assert "21,224" in matched_block.text
