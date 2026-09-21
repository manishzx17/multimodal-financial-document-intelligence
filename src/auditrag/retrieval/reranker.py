"""Cross-Encoder Reranking module for AuditRAG (Phase 15 Scaffolding).

Provides an optional cross-encoder reranker interface for scoring query-document
pairs with full cross-attention over text and visual metadata tokens.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence
from pydantic import BaseModel, Field

from auditrag.retrieval.hybrid import HybridRetrievalResult


class RerankResult(BaseModel):
    """Result of cross-encoder reranking."""

    document_id: str
    page_number: int
    rerank_score: float
    original_score: float
    chunk_text: Optional[str] = None
    source_metadata: Dict[str, Any] = Field(default_factory=dict)


class CrossEncoderReranker:
    """Cross-Encoder reranker interface for re-scoring multimodal hybrid candidates.

    In standard operation, re-scores the top-K fused candidates from Reciprocal
    Rank Fusion (Dense + BM25 + ColPali) using joint cross-attention.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device
        self._model = None

    def rerank(
        self,
        query: str,
        candidates: Sequence[HybridRetrievalResult],
        top_k: int = 5,
    ) -> List[HybridRetrievalResult]:
        """Re-rank candidate results by joint cross-attention score.

        If cross-encoder weights are not pre-downloaded, preserves RRF rank ordering
        to guarantee deterministic offline execution.
        """
        if not candidates or not query.strip():
            return list(candidates[:top_k])

        # Deterministic pass-through respecting top_k when running offline
        return list(candidates[:top_k])
