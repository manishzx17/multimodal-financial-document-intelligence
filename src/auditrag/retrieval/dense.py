"""Dense semantic text retriever for AuditRAG Phase 3."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import faiss
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from sentence_transformers import SentenceTransformer

from auditrag.config import DEFAULT_EMBEDDING_MODEL, DENSE_INDEX_DIR
from auditrag.ingestion.models import NormalizedDocument
from auditrag.retrieval.chunking import TextChunk, chunk_document


class RetrievalResult(BaseModel):
    """Result returned by the semantic retriever."""

    document_id: str
    page_number: int
    chunk_text: str
    score: float
    source_metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="ignore")


class DenseRetriever:
    """Dense retriever using SentenceTransformers and FAISS IndexFlatIP."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        index_dir: Union[str, Path] = DENSE_INDEX_DIR,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.index_dir = Path(index_dir)
        self.device = device
        self._model: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks: List[TextChunk] = []

    @property
    def model(self) -> SentenceTransformer:
        """Lazy loader for SentenceTransformer model."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def build_index(
        self,
        documents: List[NormalizedDocument],
        min_chars: int = 300,
        max_chars: int = 600,
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> int:
        """Chunk normalized documents, generate dense embeddings, and build FAISS index."""
        all_chunks: List[TextChunk] = []
        for doc in documents:
            doc_chunks = chunk_document(doc, min_chars=min_chars, max_chars=max_chars)
            all_chunks.extend(doc_chunks)

        if not all_chunks:
            raise ValueError("No text chunks generated from the provided documents.")

        texts = [chunk.text for chunk in all_chunks]
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        dimension = embeddings.shape[1]
        faiss_index = faiss.IndexFlatIP(dimension)
        faiss_index.add(embeddings)

        self.index = faiss_index
        self.chunks = all_chunks
        return len(all_chunks)

    def save_index(self, index_dir: Optional[Union[str, Path]] = None) -> Path:
        """Save FAISS index, chunk metadata, and index specification to disk."""
        if self.index is None or not self.chunks:
            raise ValueError("No index has been built or loaded to save.")

        target_dir = Path(index_dir) if index_dir else self.index_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save FAISS index binary
        faiss_path = target_dir / "index.faiss"
        faiss.write_index(self.index, str(faiss_path))

        # 2. Save Chunks JSON
        chunks_path = target_dir / "chunks.json"
        chunks_data = [chunk.model_dump() for chunk in self.chunks]
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)

        # 3. Save Index Metadata
        meta_path = target_dir / "index_meta.json"
        metadata = {
            "model_name": self.model_name,
            "dimension": self.index.d,
            "total_chunks": len(self.chunks),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        return target_dir

    def load_index(self, index_dir: Optional[Union[str, Path]] = None) -> bool:
        """Load FAISS index and chunk metadata from disk."""
        target_dir = Path(index_dir) if index_dir else self.index_dir
        faiss_path = target_dir / "index.faiss"
        chunks_path = target_dir / "chunks.json"

        if not faiss_path.exists() or not chunks_path.exists():
            return False

        self.index = faiss.read_index(str(faiss_path))
        with open(chunks_path, "r", encoding="utf-8") as f:
            raw_chunks = json.load(f)
        self.chunks = [TextChunk.model_validate(c) for c in raw_chunks]
        return True

    def retrieve(self, question: str, top_k: int = 5) -> List[RetrievalResult]:
        """Retrieve top_k most semantically similar text chunks for a query."""
        if self.index is None or not self.chunks:
            # Attempt to load existing index from disk if not explicitly initialized
            loaded = self.load_index()
            if not loaded or self.index is None:
                raise RuntimeError(
                    "No index available for retrieval. Call build_index() or load_index() first."
                )

        if not question or not question.strip():
            return []

        # Encode query with unit normalization for exact cosine similarity
        query_emb = self.model.encode(
            [question],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        k = min(top_k, len(self.chunks))
        scores, indices = self.index.search(query_emb, k)

        results: List[RetrievalResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]

            source_meta = {
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
                score=float(score),
                source_metadata=source_meta,
            )
            results.append(result)

        return results
