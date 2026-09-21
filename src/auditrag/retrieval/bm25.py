"""BM25 lexical text retriever for AuditRAG Phase 4."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
from typing import Any, Dict, List, Optional, Union

import numpy as np
from rank_bm25 import BM25Okapi

from auditrag.config import BM25_INDEX_DIR, DENSE_INDEX_DIR
from auditrag.ingestion.models import NormalizedDocument
from auditrag.retrieval.chunking import TextChunk, chunk_document
from auditrag.retrieval.dense import RetrievalResult
from auditrag.retrieval.tokenize import tokenize_text


class BM25Retriever:
    """Lexical retriever using BM25Okapi."""

    def __init__(
        self,
        index_dir: Union[str, Path] = BM25_INDEX_DIR,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.index_dir = Path(index_dir)
        self.k1 = k1
        self.b = b
        self.bm25: Optional[BM25Okapi] = None
        self.chunks: List[TextChunk] = []

    def build_index(
        self,
        documents: Optional[List[NormalizedDocument]] = None,
        chunks: Optional[List[TextChunk]] = None,
        min_chars: int = 300,
        max_chars: int = 600,
    ) -> int:
        """Tokenize text chunks and build the BM25Okapi index.

        If chunks are provided directly (e.g. reused from Phase 3), they are used directly.
        Otherwise, documents are chunked using the standard block-window chunker.
        If neither is provided, attempts to load existing chunks from Phase 3 dense index.
        """
        if chunks is not None:
            all_chunks = list(chunks)
        elif documents is not None:
            all_chunks = []
            for doc in documents:
                all_chunks.extend(chunk_document(doc, min_chars=min_chars, max_chars=max_chars))
        else:
            # Fallback: attempt to load exact chunks saved in dense index
            dense_chunks_path = DENSE_INDEX_DIR / "chunks.json"
            if dense_chunks_path.exists():
                with open(dense_chunks_path, "r", encoding="utf-8") as f:
                    raw_chunks = json.load(f)
                all_chunks = [TextChunk.model_validate(c) for c in raw_chunks]
            else:
                raise ValueError("No documents or chunks provided to build BM25 index.")

        if not all_chunks:
            raise ValueError("No text chunks available to build BM25 index.")

        tokenized_corpus = [tokenize_text(c.text) for c in all_chunks]
        self.bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
        self.chunks = all_chunks
        return len(all_chunks)

    def save_index(self, index_dir: Optional[Union[str, Path]] = None) -> Path:
        """Persist BM25 index, chunk metadata, and index metadata to disk."""
        if self.bm25 is None or not self.chunks:
            raise ValueError("No BM25 index has been built or loaded to save.")

        target_dir = Path(index_dir) if index_dir else self.index_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save BM25 pickled model
        bm25_path = target_dir / "bm25.pkl"
        with open(bm25_path, "wb") as f:
            pickle.dump(self.bm25, f)

        # 2. Save Chunks JSON
        chunks_path = target_dir / "chunks.json"
        chunks_data = [chunk.model_dump() for chunk in self.chunks]
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)

        # 3. Save Index Metadata
        meta_path = target_dir / "index_meta.json"
        metadata = {
            "retriever": "BM25Okapi",
            "k1": self.k1,
            "b": self.b,
            "total_chunks": len(self.chunks),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        return target_dir

    def load_index(self, index_dir: Optional[Union[str, Path]] = None) -> bool:
        """Load persisted BM25 model and chunk metadata from disk."""
        target_dir = Path(index_dir) if index_dir else self.index_dir
        bm25_path = target_dir / "bm25.pkl"
        chunks_path = target_dir / "chunks.json"

        if not bm25_path.exists() or not chunks_path.exists():
            return False

        with open(bm25_path, "rb") as f:
            self.bm25 = pickle.load(f)

        with open(chunks_path, "r", encoding="utf-8") as f:
            raw_chunks = json.load(f)
        self.chunks = [TextChunk.model_validate(c) for c in raw_chunks]
        return True

    def retrieve(self, question: str, top_k: int = 5) -> List[RetrievalResult]:
        """Retrieve top_k most relevant text chunks using BM25 lexical matching."""
        if self.bm25 is None or not self.chunks:
            loaded = self.load_index()
            if not loaded or self.bm25 is None:
                raise RuntimeError(
                    "No BM25 index available. Call build_index() or load_index() first."
                )

        if not question or not question.strip():
            return []

        query_tokens = tokenize_text(question)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        k = min(top_k, len(self.chunks))

        # Sort chunk indices by descending score
        top_indices = np.argsort(scores)[::-1][:k]

        results: List[RetrievalResult] = []
        for idx in top_indices:
            score = max(0.0, float(scores[idx]))
            chunk = self.chunks[idx]


            source_meta: Dict[str, Any] = {
                "chunk_id": chunk.chunk_id,
                "block_uuids": chunk.block_uuids,
                "bboxes": chunk.bboxes,
                "page_image": chunk.metadata.get("page_image"),
                "source": chunk.metadata.get("source"),
                "order_range": chunk.metadata.get("order_range"),
                "block_count": chunk.metadata.get("block_count"),
            }

            result = RetrievalResult(
                document_id=chunk.document_id,
                page_number=chunk.page_number,
                chunk_text=chunk.text,
                score=score,
                source_metadata=source_meta,
            )
            results.append(result)

        return results
