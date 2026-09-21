"""Multimodal Hybrid Retriever combining Dense, BM25, and ColPali for AuditRAG Phase 6."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, ConfigDict, Field

from auditrag.retrieval.dense import DenseRetriever, RetrievalResult
from auditrag.retrieval.bm25 import BM25Retriever
from auditrag.retrieval.colpali import ColPaliRetriever, VisualRetrievalResult


class HybridRetrievalResult(BaseModel):
    """Unified multimodal retrieval result produced by Reciprocal Rank Fusion (RRF)."""

    document_id: str
    page_number: int
    score: float  # Fused RRF score
    sources: List[str] = Field(default_factory=list)  # ["dense", "bm25", "colpali", ...]
    source_ranks: Dict[str, int] = Field(default_factory=dict)  # {"dense": 1, "bm25": 2, ...}
    source_scores: Dict[str, float] = Field(default_factory=dict)  # {"dense": 0.82, ...}

    # Text chunk fields (populated if retrieved via dense or bm25, or linked)
    chunk_id: Optional[str] = None
    chunk_text: Optional[str] = None
    source_metadata: Dict[str, Any] = Field(default_factory=dict)

    # Visual fields (populated if retrieved via colpali or page image present)
    image_path: Optional[str] = None
    visual_score: Optional[float] = None

    model_config = ConfigDict(extra="ignore")


def reciprocal_rank_fusion(
    dense_results: Optional[List[RetrievalResult]] = None,
    bm25_results: Optional[List[RetrievalResult]] = None,
    colpali_results: Optional[List[VisualRetrievalResult]] = None,
    rrf_k: int = 60,
    weights: Optional[Dict[str, float]] = None,
    top_k: int = 5,
) -> List[HybridRetrievalResult]:
    """Combine dense, BM25, and ColPali candidates using Reciprocal Rank Fusion (RRF).

    Formula:
        RRF_Score(item) = sum_{m in modalities} (weight_m / (rrf_k + rank_m(item)))

    Text results are deduplicated by chunk_id.
    ColPali results are deduplicated by (document_id, page_number).
    Cross-modal page enrichment:
      If a text chunk's page was also retrieved by ColPali, the chunk is enriched
      with ColPali's page rank, visual score, and image path, receiving the combined
      RRF score from all matching modalities.
      Any ColPali page that has no matching retrieved text chunk is retained as
      a standalone visual result.
    """
    dense_list = dense_results or []
    bm25_list = bm25_results or []
    colpali_list = colpali_results or []

    if not dense_list and not bm25_list and not colpali_list:
        return []

    w = weights or {"dense": 1.0, "bm25": 1.0, "colpali": 1.0}
    w_dense = w.get("dense", 1.0)
    w_bm25 = w.get("bm25", 1.0)
    w_colpali = w.get("colpali", 1.0)

    # 1. Process ColPali visual candidates: key is (document_id, page_number)
    # Preserve highest rank per page
    colpali_by_page: Dict[Tuple[str, int], Tuple[int, VisualRetrievalResult]] = {}
    for rank, v_res in enumerate(colpali_list, start=1):
        page_key = (v_res.document_id, v_res.page_number)
        if page_key not in colpali_by_page:
            colpali_by_page[page_key] = (rank, v_res)

    # 2. Process Text candidates (Dense and BM25): key is chunk_id
    text_candidates: Dict[str, Dict[str, Any]] = {}

    # Helper to ingest a text RetrievalResult
    def _add_text_candidate(res: RetrievalResult, source: str, rank: int, weight: float) -> None:
        cid = res.source_metadata.get("chunk_id")
        if not cid:
            cid = f"{res.document_id}_p{res.page_number}_{rank}"

        rrf_contrib = weight / (rrf_k + rank)

        if cid not in text_candidates:
            img = res.source_metadata.get("page_image")
            text_candidates[cid] = {
                "document_id": res.document_id,
                "page_number": res.page_number,
                "chunk_id": cid,
                "chunk_text": res.chunk_text,
                "source_metadata": dict(res.source_metadata),
                "image_path": img,
                "score": rrf_contrib,
                "sources": [source],
                "source_ranks": {source: rank},
                "source_scores": {source: float(res.score)},
                "visual_score": None,
            }
        else:
            cand = text_candidates[cid]
            cand["score"] += rrf_contrib
            if source not in cand["sources"]:
                cand["sources"].append(source)
            cand["source_ranks"][source] = rank
            cand["source_scores"][source] = float(res.score)

    for rank, res in enumerate(dense_list, start=1):
        _add_text_candidate(res, source="dense", rank=rank, weight=w_dense)

    for rank, res in enumerate(bm25_list, start=1):
        _add_text_candidate(res, source="bm25", rank=rank, weight=w_bm25)

    # 3. Cross-modal page enrichment:
    # Check which text chunks match a retrieved ColPali page
    pages_matched_by_text: set = set()
    for cid, cand in text_candidates.items():
        page_key = (cand["document_id"], cand["page_number"])
        if page_key in colpali_by_page:
            c_rank, v_res = colpali_by_page[page_key]
            pages_matched_by_text.add(page_key)
            rrf_contrib = w_colpali / (rrf_k + c_rank)
            cand["score"] += rrf_contrib
            if "colpali" not in cand["sources"]:
                cand["sources"].append("colpali")
            cand["source_ranks"]["colpali"] = c_rank
            cand["source_scores"]["colpali"] = float(v_res.relevance_score)
            cand["visual_score"] = float(v_res.relevance_score)
            if v_res.image_path:
                cand["image_path"] = v_res.image_path

    # 4. Collect final candidates
    all_results: List[HybridRetrievalResult] = []

    for cid, cand in text_candidates.items():
        all_results.append(
            HybridRetrievalResult(
                document_id=cand["document_id"],
                page_number=cand["page_number"],
                score=float(cand["score"]),
                sources=cand["sources"],
                source_ranks=cand["source_ranks"],
                source_scores=cand["source_scores"],
                chunk_id=cand["chunk_id"],
                chunk_text=cand["chunk_text"],
                source_metadata=cand["source_metadata"],
                image_path=cand["image_path"],
                visual_score=cand["visual_score"],
            )
        )

    # 5. Add visual-only ColPali pages (those without any retrieved text chunk)
    for page_key, (c_rank, v_res) in colpali_by_page.items():
        if page_key not in pages_matched_by_text:
            rrf_contrib = w_colpali / (rrf_k + c_rank)
            all_results.append(
                HybridRetrievalResult(
                    document_id=v_res.document_id,
                    page_number=v_res.page_number,
                    score=float(rrf_contrib),
                    sources=["colpali"],
                    source_ranks={"colpali": c_rank},
                    source_scores={"colpali": float(v_res.relevance_score)},
                    chunk_id=None,
                    chunk_text=None,
                    source_metadata={},
                    image_path=v_res.image_path,
                    visual_score=float(v_res.relevance_score),
                )
            )

    # 6. Sort by fused score descending
    all_results.sort(key=lambda x: x.score, reverse=True)

    return all_results[:top_k]


class HybridRetriever:
    """Multimodal hybrid retriever fusing Dense, BM25, and ColPali via RRF."""

    def __init__(
        self,
        dense_retriever: Optional[DenseRetriever] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
        colpali_retriever: Optional[ColPaliRetriever] = None,
        rrf_k: int = 60,
        weights: Optional[Dict[str, float]] = None,
    ):
        self._dense = dense_retriever
        self._bm25 = bm25_retriever
        self._colpali = colpali_retriever
        self.rrf_k = rrf_k
        self.weights = weights or {"dense": 1.0, "bm25": 1.0, "colpali": 1.0}

    @property
    def dense_retriever(self) -> DenseRetriever:
        """Lazy loader for Dense retriever."""
        if self._dense is None:
            self._dense = DenseRetriever()
            self._dense.load_index()
        return self._dense

    @property
    def bm25_retriever(self) -> BM25Retriever:
        """Lazy loader for BM25 retriever."""
        if self._bm25 is None:
            self._bm25 = BM25Retriever()
            self._bm25.load_index()
        return self._bm25

    @property
    def colpali_retriever(self) -> ColPaliRetriever:
        """Lazy loader for ColPali retriever."""
        if self._colpali is None:
            self._colpali = ColPaliRetriever()
            self._colpali.load_index()
        return self._colpali

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        candidate_pool_size: Optional[int] = None,
    ) -> List[HybridRetrievalResult]:
        """Retrieve top_k multimodal results for question using Reciprocal Rank Fusion."""
        if not question or not question.strip():
            return []

        pool_k = candidate_pool_size or max(top_k * 4, 20)

        # Retrieve candidate pools from each individual engine
        dense_candidates = self.dense_retriever.retrieve(question, top_k=pool_k)
        bm25_candidates = self.bm25_retriever.retrieve(question, top_k=pool_k)
        colpali_candidates = self.colpali_retriever.retrieve(question, top_k=pool_k)

        # Fuse rankings using RRF
        return reciprocal_rank_fusion(
            dense_results=dense_candidates,
            bm25_results=bm25_candidates,
            colpali_results=colpali_candidates,
            rrf_k=self.rrf_k,
            weights=self.weights,
            top_k=top_k,
        )
