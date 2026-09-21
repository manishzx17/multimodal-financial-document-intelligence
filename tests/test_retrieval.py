"""Unit and integration tests for Phase 3 Dense Text Retrieval."""

from pathlib import Path
import pytest

from auditrag.config import PROCESSED_DOCUMENTS_DIR
from auditrag.ingestion.models import NormalizedDocument
from auditrag.retrieval import (
    DenseRetriever,
    RetrievalResult,
    TextChunk,
    chunk_document,
    chunk_page,
)


@pytest.fixture
def sample_doc() -> NormalizedDocument:
    doc_path = PROCESSED_DOCUMENTS_DIR / "637fab7088ea6c78a5dba55f17e833bd.json"
    if not doc_path.exists():
        pytest.skip(f"Processed document not found: {doc_path}")
    return NormalizedDocument.from_file(doc_path)


@pytest.fixture
def multipage_doc() -> NormalizedDocument:
    doc_path = PROCESSED_DOCUMENTS_DIR / "0b96522d90d672fb9a9575684feab0a6.json"
    if not doc_path.exists():
        pytest.skip(f"Processed multipage document not found: {doc_path}")
    return NormalizedDocument.from_file(doc_path)


def test_chunking_preserves_metadata(sample_doc: NormalizedDocument):
    """Verify chunking maintains document_id, page_number, block_uuids, bboxes, and image path."""
    chunks = chunk_document(sample_doc, min_chars=300, max_chars=600)
    assert len(chunks) > 0

    all_doc_block_uuids = {b.uuid for p in sample_doc.pages for b in p.blocks}

    for chunk in chunks:
        assert isinstance(chunk, TextChunk)
        assert chunk.document_id == sample_doc.document_id
        assert chunk.page_number == 1
        assert len(chunk.text) > 0
        assert len(chunk.block_uuids) > 0
        assert len(chunk.bboxes) == len(chunk.block_uuids)

        # Confirm all chunk block uuids exist in the source document
        for buuid in chunk.block_uuids:
            assert buuid in all_doc_block_uuids

        # Check metadata attributes
        assert "source" in chunk.metadata
        assert "page_image" in chunk.metadata
        assert chunk.metadata["page_image"] is not None


def test_chunking_multipage_document(multipage_doc: NormalizedDocument):
    """Verify chunking multi-page documents separates chunks by page."""
    chunks = chunk_document(multipage_doc, min_chars=300, max_chars=600)
    assert len(chunks) > 0

    pages_with_chunks = {c.page_number for c in chunks}
    assert 1 in pages_with_chunks
    assert 2 in pages_with_chunks


def test_dense_retriever_build_and_retrieve(sample_doc: NormalizedDocument, tmp_path: Path):
    """Verify building index, querying, and checking result structure and score ordering."""
    retriever = DenseRetriever(index_dir=tmp_path / "dense")
    total_chunks = retriever.build_index([sample_doc])
    assert total_chunks > 0
    assert retriever.index is not None
    assert retriever.index.ntotal == total_chunks

    # Query the retriever
    results = retriever.retrieve("assets and financial position", top_k=3)
    assert len(results) <= 3
    assert len(results) > 0

    # Validate RetrievalResult structure
    for res in results:
        assert isinstance(res, RetrievalResult)
        assert res.document_id == sample_doc.document_id
        assert res.page_number == 1
        assert isinstance(res.chunk_text, str) and len(res.chunk_text) > 0
        assert isinstance(res.score, float)
        assert -1.0 <= res.score <= 1.0  # Cosine similarity range

        # Verify source metadata
        assert "chunk_id" in res.source_metadata
        assert "block_uuids" in res.source_metadata
        assert len(res.source_metadata["block_uuids"]) > 0
        assert "bboxes" in res.source_metadata
        assert len(res.source_metadata["bboxes"]) > 0
        assert "page_image" in res.source_metadata

    # Validate descending score ordering
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_dense_retriever_persistence(sample_doc: NormalizedDocument, tmp_path: Path):
    """Verify saving FAISS index to disk and reloading without rebuilding."""
    index_dir = tmp_path / "saved_dense_index"
    retriever1 = DenseRetriever(index_dir=index_dir)
    retriever1.build_index([sample_doc])
    saved_path = retriever1.save_index()
    assert (saved_path / "index.faiss").exists()
    assert (saved_path / "chunks.json").exists()
    assert (saved_path / "index_meta.json").exists()

    # Load in a fresh retriever
    retriever2 = DenseRetriever(index_dir=index_dir)
    loaded = retriever2.load_index()
    assert loaded is True
    assert retriever2.index is not None
    assert retriever2.index.ntotal == retriever1.index.ntotal
    assert len(retriever2.chunks) == len(retriever1.chunks)

    # Compare query outputs
    query = "What was the total assets in 2018?"
    res1 = retriever1.retrieve(query, top_k=2)
    res2 = retriever2.retrieve(query, top_k=2)

    assert len(res1) == len(res2)
    for r1, r2 in zip(res1, res2):
        assert r1.document_id == r2.document_id
        assert r1.chunk_text == r2.chunk_text
        assert pytest.approx(r1.score, rel=1e-4) == r2.score


def test_dense_retriever_empty_query(sample_doc: NormalizedDocument, tmp_path: Path):
    """Verify empty query handling."""
    retriever = DenseRetriever(index_dir=tmp_path / "dense")
    retriever.build_index([sample_doc])
    assert retriever.retrieve("") == []
    assert retriever.retrieve("   ") == []
