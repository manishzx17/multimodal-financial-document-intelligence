"""Retrieval module for AuditRAG (Dense, BM25, and ColPali)."""

from auditrag.retrieval.bm25 import BM25Retriever
from auditrag.retrieval.chunking import TextChunk, chunk_document, chunk_page
from auditrag.retrieval.colpali import ColPaliRetriever, VisualRetrievalResult
from auditrag.retrieval.dense import DenseRetriever, RetrievalResult
from auditrag.retrieval.hybrid import (
    HybridRetrievalResult,
    HybridRetriever,
    reciprocal_rank_fusion,
)
from auditrag.retrieval.tokenize import tokenize_text

__all__ = [
    "TextChunk",
    "chunk_page",
    "chunk_document",
    "RetrievalResult",
    "DenseRetriever",
    "BM25Retriever",
    "tokenize_text",
    "ColPaliRetriever",
    "VisualRetrievalResult",
    "HybridRetriever",
    "HybridRetrievalResult",
    "reciprocal_rank_fusion",
]
