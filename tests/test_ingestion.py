"""Unit and integration tests for Phase 2 Document Ingestion."""

from pathlib import Path
import pytest

from auditrag.config import RAW_TEST_DIR
from auditrag.data import TATDQALoader
from auditrag.ingestion import (
    DocumentIngestionPipeline,
    NormalizedBlock,
    NormalizedDocument,
    NormalizedPage,
    NormalizedWords,
    TATDQAParser,
)


@pytest.fixture
def parser() -> TATDQAParser:
    return TATDQAParser(raw_test_dir=RAW_TEST_DIR)


@pytest.fixture
def pipeline(tmp_path: Path) -> DocumentIngestionPipeline:
    return DocumentIngestionPipeline(
        raw_test_dir=RAW_TEST_DIR,
        output_dir=tmp_path / "processed" / "documents",
    )


def test_normalize_single_page_document(parser: TATDQAParser):
    """Confirm single-page document ingestion produces the required normalized structure."""
    doc_uid = "637fab7088ea6c78a5dba55f17e833bd"
    doc = parser.parse_document(doc_uid=doc_uid, source="plexus-corp_2019.pdf")

    # Document-level validation
    assert isinstance(doc, NormalizedDocument)
    assert doc.document_id == doc_uid
    assert doc.source == "plexus-corp_2019.pdf"
    assert doc.page_count == 1
    assert len(doc.pages) == 1
    assert doc.pdf_path is not None
    assert Path(doc.pdf_path).exists()

    # Page-level validation: page_number, text, blocks, bbox, words, page_image
    page = doc.pages[0]
    assert isinstance(page, NormalizedPage)
    assert page.page_number == 1
    assert isinstance(page.text, str) and len(page.text) > 0
    assert len(page.bbox) == 4
    assert page.page_image is not None
    assert Path(page.page_image).exists()

    # Block-level validation: uuid, text, bbox, order, words
    assert len(page.blocks) > 0
    for block in page.blocks:
        assert isinstance(block, NormalizedBlock)
        assert isinstance(block.uuid, str) and len(block.uuid) > 0
        assert isinstance(block.text, str)
        assert len(block.bbox) == 4
        assert isinstance(block.order, int)
        assert isinstance(block.words, NormalizedWords)
        assert len(block.words.word_list) == len(block.words.bbox_list)

    # Page-level words
    assert page.words is not None
    assert len(page.words.word_list) == len(page.words.bbox_list)
    assert len(page.words.word_list) > 0


def test_normalize_multipage_document(parser: TATDQAParser):
    """Confirm multi-page document properly associates all pages, images, and numbers."""
    doc_uid = "0b96522d90d672fb9a9575684feab0a6"
    doc = parser.parse_document(doc_uid=doc_uid)

    assert doc.document_id == doc_uid
    assert doc.page_count == 2
    assert len(doc.pages) == 2

    # Check Page 1
    p1 = doc.get_page(1)
    assert p1 is not None
    assert p1.page_number == 1
    assert p1.page_image is not None
    assert "1.png" in p1.page_image
    assert Path(p1.page_image).exists()

    # Check Page 2
    p2 = doc.get_page(2)
    assert p2 is not None
    assert p2.page_number == 2
    assert p2.page_image is not None
    assert "2.png" in p2.page_image
    assert Path(p2.page_image).exists()


def test_processed_document_roundtrip_serialization(parser: TATDQAParser, tmp_path: Path):
    """Confirm a normalized document can be saved to JSON and accurately loaded back."""
    doc_uid = "637fab7088ea6c78a5dba55f17e833bd"
    original_doc = parser.parse_document(doc_uid=doc_uid, source="sample.pdf")

    out_file = tmp_path / f"{doc_uid}.json"
    original_doc.save_to_file(out_file)
    assert out_file.exists()

    loaded_doc = NormalizedDocument.from_file(out_file)
    assert loaded_doc.document_id == original_doc.document_id
    assert loaded_doc.source == original_doc.source
    assert loaded_doc.page_count == original_doc.page_count
    assert len(loaded_doc.pages) == len(original_doc.pages)

    loaded_page = loaded_doc.pages[0]
    orig_page = original_doc.pages[0]
    assert loaded_page.page_number == orig_page.page_number
    assert loaded_page.text == orig_page.text
    assert loaded_page.bbox == orig_page.bbox
    assert loaded_page.page_image == orig_page.page_image
    assert len(loaded_page.blocks) == len(orig_page.blocks)
    assert loaded_page.blocks[0].uuid == orig_page.blocks[0].uuid
    assert loaded_page.blocks[0].text == orig_page.blocks[0].text
    assert loaded_page.blocks[0].bbox == orig_page.blocks[0].bbox


def test_pipeline_ingest_and_save(pipeline: DocumentIngestionPipeline):
    """Confirm DocumentIngestionPipeline ingests, writes, and loads documents."""
    doc_uid = "01fdc2339641d7d4a8b06f32d937cffd"
    doc = pipeline.ingest_document(doc_uid=doc_uid)
    saved_path = pipeline.save_processed_document(doc)

    assert saved_path.exists()
    assert saved_path.name == f"{doc_uid}.json"

    reloaded = pipeline.load_processed_document(doc_uid)
    assert reloaded.document_id == doc_uid
    assert reloaded.page_count == 1
    assert reloaded.pages[0].page_number == 1
    assert len(reloaded.pages[0].blocks) > 0


def test_pipeline_batch_subset_and_manifest(pipeline: DocumentIngestionPipeline):
    """Confirm batch ingestion writes normalized JSONs and a valid manifest.json."""
    summary = pipeline.ingest_all(limit=3)

    assert summary["total_documents"] == 3
    assert summary["total_pages"] >= 3
    assert summary["total_blocks"] > 0

    manifest_path = Path(summary["manifest_path"])
    assert manifest_path.exists()

    all_loaded = pipeline.load_all_processed_documents()
    assert len(all_loaded) == 3


def test_phase1_loader_compatibility():
    """Confirm Phase 1 loader Document converts seamlessly to Phase 2 NormalizedDocument."""
    loader = TATDQALoader()
    doc_uid = "637fab7088ea6c78a5dba55f17e833bd"
    p1_doc = loader.load_document(doc_uid, source="plexus.pdf", annotated_page=1)

    # Test compatibility aliases
    assert p1_doc.document_id == p1_doc.uid
    assert p1_doc.pages[0].page_number == p1_doc.pages[0].page_idx
    assert p1_doc.pages[0].text == p1_doc.pages[0].full_text
    assert p1_doc.pages[0].page_image == str(p1_doc.pages[0].image_path)

    # Test conversion to NormalizedDocument
    norm_doc = p1_doc.to_normalized()
    assert isinstance(norm_doc, NormalizedDocument)
    assert norm_doc.document_id == p1_doc.uid
    assert norm_doc.page_count == p1_doc.page_count
    assert norm_doc.pages[0].page_number == 1
    assert norm_doc.pages[0].text == p1_doc.pages[0].full_text
    assert len(norm_doc.pages[0].blocks) == len(p1_doc.pages[0].blocks)
