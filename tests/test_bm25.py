"""Unit and integration tests for Phase 4 BM25 Lexical Retrieval."""

from pathlib import Path
import pytest

from auditrag.config import PROCESSED_DOCUMENTS_DIR
from auditrag.ingestion.models import NormalizedDocument
from auditrag.retrieval import (
    BM25Retriever,
    RetrievalResult,
    TextChunk,
    chunk_document,
    tokenize_text,
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


def test_tokenize_text():
    """Verify regex tokenization handles numbers, years, decimals, and financial acronyms."""
    sample = "In 2018, AMER assets were 645,791 with a 15.4% growth in EBITDA."
    tokens = tokenize_text(sample)

    assert "2018" in tokens
    assert "amer" in tokens
    assert "assets" in tokens
    assert "645" in tokens
    assert "791" in tokens
    assert "15.4" in tokens
    assert "ebitda" in tokens


def test_bm25_build_and_retrieve(sample_doc: NormalizedDocument, multipage_doc: NormalizedDocument, tmp_path: Path):
    """Verify building index, querying, score ordering, and result structure."""
    retriever = BM25Retriever(index_dir=tmp_path / "bm25", k1=1.5, b=0.75)
    total_chunks = retriever.build_index(documents=[sample_doc, multipage_doc])

    assert total_chunks > 0
    assert retriever.bm25 is not None
    assert len(retriever.chunks) == total_chunks

    # Query for exact keywords present in sample_doc
    results = retriever.retrieve("AMER 2018 depreciation", top_k=3)
    assert len(results) > 0
    assert len(results) <= 3

    # Check top result
    top_res = results[0]
    assert isinstance(top_res, RetrievalResult)
    assert top_res.document_id == sample_doc.document_id
    assert top_res.page_number == 1
    assert isinstance(top_res.chunk_text, str) and len(top_res.chunk_text) > 0
    assert isinstance(top_res.score, float)
    assert top_res.score > 0.0

    # Verify descending score ordering
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)



def test_bm25_metadata_preservation(sample_doc: NormalizedDocument, tmp_path: Path):
    """Verify that BM25 retrieval results preserve block UUIDs, coordinates, and images."""
    retriever = BM25Retriever(index_dir=tmp_path / "bm25")
    retriever.build_index(documents=[sample_doc])

    results = retriever.retrieve("Plexus Corp Notes", top_k=1)
    assert len(results) == 1

    res = results[0]
    meta = res.source_metadata
    assert "chunk_id" in meta
    assert "block_uuids" in meta
    assert len(meta["block_uuids"]) > 0
    assert "bboxes" in meta
    assert len(meta["bboxes"]) == len(meta["block_uuids"])
    assert "page_image" in meta
    assert meta["page_image"] is not None
    assert "source" in meta


def test_bm25_persistence_roundtrip(sample_doc: NormalizedDocument, tmp_path: Path):
    """Verify BM25 index save to disk and reload without rebuilding."""
    index_dir = tmp_path / "saved_bm25"
    retriever1 = BM25Retriever(index_dir=index_dir)
    retriever1.build_index(documents=[sample_doc])
    saved_path = retriever1.save_index()

    assert (saved_path / "bm25.pkl").exists()
    assert (saved_path / "chunks.json").exists()
    assert (saved_path / "index_meta.json").exists()

    # Load in new retriever instance
    retriever2 = BM25Retriever(index_dir=index_dir)
    loaded = retriever2.load_index()
    assert loaded is True
    assert retriever2.bm25 is not None
    assert len(retriever2.chunks) == len(retriever1.chunks)

    # Compare query outputs
    query = "Capital expenditures AMER"
    res1 = retriever1.retrieve(query, top_k=2)
    res2 = retriever2.retrieve(query, top_k=2)

    assert len(res1) == len(res2)
    for r1, r2 in zip(res1, res2):
        assert r1.document_id == r2.document_id
        assert r1.chunk_text == r2.chunk_text
        assert pytest.approx(r1.score, rel=1e-4) == r2.score


def test_bm25_empty_query(sample_doc: NormalizedDocument, tmp_path: Path):
    """Verify empty and punctuation-only queries return an empty list."""
    retriever = BM25Retriever(index_dir=tmp_path / "bm25")
    retriever.build_index(documents=[sample_doc])

    assert retriever.retrieve("") == []
    assert retriever.retrieve("   ") == []
    assert retriever.retrieve("??? !!! ,,,") == []
