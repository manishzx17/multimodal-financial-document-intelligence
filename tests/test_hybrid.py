"""Unit and integration tests for Phase 6 Multimodal Hybrid Retrieval."""

from pathlib import Path
import pytest

from auditrag.retrieval import (
    HybridRetrievalResult,
    HybridRetriever,
    RetrievalResult,
    VisualRetrievalResult,
    reciprocal_rank_fusion,
)


def test_rrf_scoring_and_ranking():
    """Verify exact mathematical calculation of RRF scores, weights, and ranking order."""
    # Chunk A: Dense rank 1, BM25 rank 2
    # Chunk B: Dense rank 2, BM25 rank 1
    # Page X: ColPali rank 1
    chunk_a_dense = RetrievalResult(
        document_id="doc_a",
        page_number=1,
        chunk_text="Text of Chunk A",
        score=0.95,
        source_metadata={"chunk_id": "chunk_a", "page_image": "doc_a_1.png"},
    )
    chunk_b_dense = RetrievalResult(
        document_id="doc_b",
        page_number=1,
        chunk_text="Text of Chunk B",
        score=0.85,
        source_metadata={"chunk_id": "chunk_b", "page_image": "doc_b_1.png"},
    )
    chunk_b_bm25 = RetrievalResult(
        document_id="doc_b",
        page_number=1,
        chunk_text="Text of Chunk B",
        score=15.0,
        source_metadata={"chunk_id": "chunk_b", "page_image": "doc_b_1.png"},
    )
    chunk_a_bm25 = RetrievalResult(
        document_id="doc_a",
        page_number=1,
        chunk_text="Text of Chunk A",
        score=12.0,
        source_metadata={"chunk_id": "chunk_a", "page_image": "doc_a_1.png"},
    )
    colpali_page_a = VisualRetrievalResult(
        document_id="doc_a",
        page_number=1,
        relevance_score=14.2,
        image_path="doc_a_1.png",
    )

    # Weights: equal 1.0, k = 60
    # Expected score for chunk_a:
    # dense rank 1 -> 1 / (60 + 1) = 1/61
    # bm25 rank 2 -> 1 / (60 + 2) = 1/62
    # colpali page a rank 1 -> 1 / (60 + 1) = 1/61
    # total = 1/61 + 1/62 + 1/61 = 2/61 + 1/62 ≈ 0.032786885 + 0.016129032 ≈ 0.0489159
    expected_score_a = (1.0 / 61.0) + (1.0 / 62.0) + (1.0 / 61.0)

    # Expected score for chunk_b:
    # dense rank 2 -> 1/62
    # bm25 rank 1 -> 1/61
    # total = 1/62 + 1/61 ≈ 0.0325232
    expected_score_b = (1.0 / 62.0) + (1.0 / 61.0)

    fused = reciprocal_rank_fusion(
        dense_results=[chunk_a_dense, chunk_b_dense],
        bm25_results=[chunk_b_bm25, chunk_a_bm25],
        colpali_results=[colpali_page_a],
        rrf_k=60,
        top_k=5,
    )

    assert len(fused) == 2
    assert fused[0].chunk_id == "chunk_a"
    assert pytest.approx(fused[0].score, rel=1e-5) == expected_score_a
    assert fused[1].chunk_id == "chunk_b"
    assert pytest.approx(fused[1].score, rel=1e-5) == expected_score_b
    assert fused[0].score > fused[1].score


def test_hybrid_retriever_source_preservation():
    """Verify that single-source and multi-source items preserve sources and individual scores."""
    chunk_dense = RetrievalResult(
        document_id="doc_1",
        page_number=1,
        chunk_text="Dense only text",
        score=0.91,
        source_metadata={"chunk_id": "chunk_dense_only"},
    )
    chunk_bm25 = RetrievalResult(
        document_id="doc_2",
        page_number=2,
        chunk_text="BM25 only text",
        score=18.4,
        source_metadata={"chunk_id": "chunk_bm25_only"},
    )
    colpali_visual = VisualRetrievalResult(
        document_id="doc_3",
        page_number=1,
        relevance_score=15.3,
        image_path="doc_3_1.png",
    )

    fused = reciprocal_rank_fusion(
        dense_results=[chunk_dense],
        bm25_results=[chunk_bm25],
        colpali_results=[colpali_visual],
        rrf_k=60,
    )

    by_id = {res.chunk_id or f"{res.document_id}_{res.page_number}": res for res in fused}

    # Dense only item
    res_dense = by_id["chunk_dense_only"]
    assert res_dense.sources == ["dense"]
    assert res_dense.source_ranks == {"dense": 1}
    assert res_dense.source_scores == {"dense": 0.91}
    assert res_dense.visual_score is None

    # BM25 only item
    res_bm25 = by_id["chunk_bm25_only"]
    assert res_bm25.sources == ["bm25"]
    assert res_bm25.source_ranks == {"bm25": 1}
    assert res_bm25.source_scores == {"bm25": 18.4}
    assert res_bm25.visual_score is None

    # ColPali visual-only item
    res_visual = by_id["doc_3_1"]
    assert res_visual.sources == ["colpali"]
    assert res_visual.source_ranks == {"colpali": 1}
    assert res_visual.source_scores == {"colpali": 15.3}
    assert res_visual.visual_score == 15.3
    assert res_visual.image_path == "doc_3_1.png"
    assert res_visual.chunk_text is None


def test_hybrid_retriever_metadata_preservation():
    """Verify text chunk metadata (bboxes, block UUIDs) and visual metadata are preserved."""
    bboxes = [[10.0, 20.0, 100.0, 200.0], [30.0, 40.0, 120.0, 220.0]]
    block_uuids = ["uuid-1", "uuid-2"]
    chunk = RetrievalResult(
        document_id="doc_meta",
        page_number=2,
        chunk_text="Segment revenue for APAC",
        score=0.88,
        source_metadata={
            "chunk_id": "chunk_meta_1",
            "bboxes": bboxes,
            "block_uuids": block_uuids,
            "page_image": "doc_meta_2.png",
            "source": "financial_report.pdf",
        },
    )
    colpali = VisualRetrievalResult(
        document_id="doc_meta",
        page_number=2,
        relevance_score=16.8,
        image_path="doc_meta_2.png",
    )

    fused = reciprocal_rank_fusion(
        dense_results=[chunk],
        bm25_results=[chunk],
        colpali_results=[colpali],
        rrf_k=60,
    )

    assert len(fused) == 1
    res = fused[0]
    assert set(res.sources) == {"dense", "bm25", "colpali"}
    assert res.document_id == "doc_meta"
    assert res.page_number == 2
    assert res.chunk_id == "chunk_meta_1"
    assert res.chunk_text == "Segment revenue for APAC"
    assert res.source_metadata["bboxes"] == bboxes
    assert res.source_metadata["block_uuids"] == block_uuids
    assert res.source_metadata["source"] == "financial_report.pdf"
    assert res.image_path == "doc_meta_2.png"
    assert res.visual_score == 16.8


def test_hybrid_retriever_empty_query():
    """Verify that empty and whitespace queries return an empty list without computing."""
    class DummyRetriever:
        def retrieve(self, q, top_k=5):
            raise AssertionError("Should not be called for empty query")

    retriever = HybridRetriever(
        dense_retriever=DummyRetriever(),
        bm25_retriever=DummyRetriever(),
        colpali_retriever=DummyRetriever(),
    )

    assert retriever.retrieve("") == []
    assert retriever.retrieve("   ") == []
    assert reciprocal_rank_fusion([], [], []) == []


def test_hybrid_retriever_integration_with_persisted_indices():
    """Verify live query retrieval from persisted Dense, BM25, and ColPali indices."""
    retriever = HybridRetriever()

    # Query with a real financial query matching the test dataset
    results = retriever.retrieve("What was the total assets from AMER in 2018?", top_k=5)

    assert len(results) > 0
    assert len(results) <= 5

    # Verify score ordering
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)

    # Verify result structure and provenance
    for r in results:
        assert isinstance(r, HybridRetrievalResult)
        assert len(r.document_id) == 32
        assert r.page_number in (1, 2)
        assert r.score > 0.0
        assert len(r.sources) > 0
        for s in r.sources:
            assert s in ("dense", "bm25", "colpali")
            assert s in r.source_ranks
            assert s in r.source_scores

        if "dense" in r.sources or "bm25" in r.sources:
            assert r.chunk_text is not None and len(r.chunk_text) > 0
            assert r.chunk_id is not None
            assert "chunk_id" in r.source_metadata

        if "colpali" in r.sources:
            assert r.visual_score is not None
            assert r.image_path is not None
            assert Path(r.image_path).exists()
