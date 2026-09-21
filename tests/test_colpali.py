"""Unit and integration tests for Phase 5 ColPali Visual Retrieval."""

from pathlib import Path
import pytest
import torch
from colpali_engine.models import ColPaliProcessor

from auditrag.config import COLPALI_INDEX_DIR, RAW_TEST_DIR
from auditrag.retrieval import ColPaliRetriever, VisualRetrievalResult


def test_discover_pages():
    """Verify discovery and parsing of all 312 TAT-DQA rendered page PNGs."""
    pages = ColPaliRetriever.discover_pages(raw_test_dir=RAW_TEST_DIR)
    assert len(pages) == 312

    for p in pages:
        assert len(p["document_id"]) == 32
        assert p["page_number"] in (1, 2)
        assert Path(p["image_path"]).exists()
        assert p["page_id"] == f"{p['document_id']}_p{p['page_number']}"


def test_colpali_score_multi_vector():
    """Verify ColPali late-interaction MaxSim scoring via ColPaliProcessor."""
    processor = ColPaliProcessor.from_pretrained("vidore/colpali-v1.2")

    # 1 query with 5 tokens of dim 128
    query_emb = torch.randn(1, 5, 128)
    # 2 passages, each with 16 patch tokens of dim 128
    passage_embs = [torch.randn(16, 128), torch.randn(16, 128)]

    scores = processor.score_multi_vector([query_emb[0]], passage_embs, device="cpu")
    assert isinstance(scores, torch.Tensor)
    assert scores.shape == (1, 2)
    assert isinstance(scores[0, 0].item(), float)



def test_colpali_persistence_roundtrip(tmp_path: Path):
    """Verify saving and loading ColPali multi-vector representations and metadata."""
    index_dir = tmp_path / "test_colpali_index"
    retriever = ColPaliRetriever(index_dir=index_dir)

    pages = retriever.discover_pages()[:2]
    # Create sample multi-vector patch embeddings for testing persistence
    dummy_embs = [torch.randn(1031, 128) for _ in pages]

    retriever.pages = pages
    retriever.embeddings = dummy_embs
    saved_path = retriever.save_index()

    assert (saved_path / "page_embeddings.pt").exists()
    assert (saved_path / "pages.json").exists()
    assert (saved_path / "index_meta.json").exists()

    # Load in new retriever instance
    new_retriever = ColPaliRetriever(index_dir=index_dir)
    loaded = new_retriever.load_index()
    assert loaded is True
    assert len(new_retriever.pages) == 2
    assert len(new_retriever.embeddings) == 2
    assert new_retriever.embeddings[0].shape == (1031, 128)


def test_colpali_retriever_search_on_persisted_index():
    """Verify query-time retrieval and result structure from the persisted index."""
    retriever = ColPaliRetriever(index_dir=COLPALI_INDEX_DIR)
    loaded = retriever.load_index()
    assert loaded is True
    assert len(retriever.pages) == 312
    assert len(retriever.embeddings) == 312

    # Query using multi-vector scoring directly with pre-computed query tensor
    query_tensor = torch.randn(1, 10, 128)
    scores = retriever.processor.score_multi_vector([query_tensor[0]], retriever.embeddings[:5], device="cpu")
    assert scores.shape == (1, 5)



    # Test top-k result ranking
    top_scores, top_indices = torch.topk(scores[0], k=3)
    results = [
        VisualRetrievalResult(
            document_id=retriever.pages[idx]["document_id"],
            page_number=retriever.pages[idx]["page_number"],
            relevance_score=float(score),
            image_path=retriever.pages[idx]["image_path"],
        )
        for score, idx in zip(top_scores.tolist(), top_indices.tolist())
    ]

    assert len(results) == 3
    for res in results:
        assert isinstance(res, VisualRetrievalResult)
        assert len(res.document_id) == 32
        assert res.page_number in (1, 2)
        assert Path(res.image_path).exists()
        assert isinstance(res.relevance_score, float)

    # Verify descending score ordering
    assert [r.relevance_score for r in results] == sorted([r.relevance_score for r in results], reverse=True)


def test_colpali_empty_query():
    """Verify empty query returns empty list."""
    retriever = ColPaliRetriever(index_dir=COLPALI_INDEX_DIR)
    retriever.load_index()
    assert retriever.retrieve("") == []
    assert retriever.retrieve("   ") == []
