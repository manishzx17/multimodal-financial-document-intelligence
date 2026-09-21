"""ColPali visual page retrieval for AuditRAG Phase 5."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Union

# Disable tqdm background monitor thread to prevent thread conflicts with MPS
os.environ["TQDM_DISABLE_MONITOR"] = "1"
try:
    from tqdm import tqdm
    tqdm.monitor_interval = 0
except ImportError:
    pass

from PIL import Image
from pydantic import BaseModel, ConfigDict
import torch


from auditrag.config import (
    COLPALI_INDEX_DIR,
    DEFAULT_COLPALI_MODEL,
    PROJECT_ROOT,
    RAW_TEST_DIR,
)


class VisualRetrievalResult(BaseModel):
    """Result returned by the ColPali visual retriever."""

    document_id: str
    page_number: int
    relevance_score: float
    image_path: str

    model_config = ConfigDict(extra="ignore")


class ColPaliRetriever:
    """ColPali visual retriever using patch-level multi-vector late-interaction scoring."""

    def __init__(
        self,
        model_name: str = DEFAULT_COLPALI_MODEL,
        index_dir: Union[str, Path] = COLPALI_INDEX_DIR,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        self.model_name = model_name
        self.index_dir = Path(index_dir)
        if device is not None:
            self.device = device
        elif torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        if torch_dtype is not None:
            self.torch_dtype = torch_dtype
        elif self.device in ("mps", "cuda"):
            self.torch_dtype = torch.bfloat16
        else:
            self.torch_dtype = torch.float32

        self._model = None
        self._processor = None
        self.pages: List[Dict[str, Any]] = []
        self.embeddings: List[torch.Tensor] = []

    @property
    def processor(self):
        """Lazy loader for ColPaliProcessor."""
        if self._processor is None:
            from colpali_engine.models import ColPaliProcessor
            self._processor = ColPaliProcessor.from_pretrained(self.model_name)
        return self._processor

    @property
    def model(self):
        """Lazy loader for ColPali model."""
        if self._model is None:
            from colpali_engine.models import ColPali
            self._model = ColPali.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                device_map=self.device,
            )
            self._model.eval()
        return self._model

    @staticmethod
    def discover_pages(raw_test_dir: Union[str, Path] = RAW_TEST_DIR) -> List[Dict[str, Any]]:
        """Discover and catalog all TAT-DQA rendered page PNGs in test directory."""
        test_dir = Path(raw_test_dir)
        page_pattern = re.compile(r"^([a-f0-9]{32})_(\d+)\.png$")
        discovered: List[Dict[str, Any]] = []

        for img_path in sorted(test_dir.glob("*.png")):
            match = page_pattern.match(img_path.name)
            if match:
                doc_uid = match.group(1)
                page_num = int(match.group(2))
                try:
                    rel_path = img_path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
                except ValueError:
                    rel_path = img_path.as_posix()

                discovered.append({
                    "document_id": doc_uid,
                    "page_number": page_num,
                    "page_id": f"{doc_uid}_p{page_num}",
                    "image_path": rel_path,
                    "absolute_path": str(img_path.resolve()),
                })

        return discovered

    def build_index(
        self,
        page_records: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 4,
        show_progress: bool = False,
    ) -> int:
        """Encode all page PNGs into multi-vector visual patch embeddings."""
        if page_records is None:
            page_records = self.discover_pages()

        if not page_records:
            raise ValueError("No page image records found to index.")

        encoded_embeddings: List[torch.Tensor] = []
        indexed_pages: List[Dict[str, Any]] = []

        total = len(page_records)
        for i in range(0, total, batch_size):
            batch = page_records[i : i + batch_size]
            batch_imgs = []
            valid_batch_meta = []

            for p_meta in batch:
                rel = p_meta.get("image_path")
                if rel and (PROJECT_ROOT / rel).exists():
                    p_path = PROJECT_ROOT / rel
                else:
                    p_path = Path(p_meta.get("absolute_path", rel or ""))
                    if not p_path.is_absolute():
                        p_path = PROJECT_ROOT / p_path
                if p_path.exists():
                    img = Image.open(p_path).convert("RGB")
                    batch_imgs.append(img)
                    valid_batch_meta.append(p_meta)

            if not batch_imgs:
                continue

            batch_inputs = self.processor.process_images(batch_imgs).to(self.device)
            # Remove labels if injected by processor to avoid 4GB+ cross-entropy logits allocation
            batch_inputs.pop("labels", None)

            with torch.no_grad():
                batch_embs = self.model(**batch_inputs)

            # Move embeddings to CPU float32 for compact persistence
            for j in range(batch_embs.shape[0]):
                single_emb = batch_embs[j].detach().cpu().to(torch.float32)
                encoded_embeddings.append(single_emb)
                indexed_pages.append(valid_batch_meta[j])

            if self.device == "mps":
                torch.mps.empty_cache()

            if show_progress:
                print(f"Indexed {min(i + batch_size, total)}/{total} pages...")


        self.embeddings = encoded_embeddings
        self.pages = indexed_pages
        return len(self.pages)

    def save_index(self, index_dir: Optional[Union[str, Path]] = None) -> Path:
        """Persist multi-vector page embeddings and catalog metadata under data/indices/colpali/."""
        if not self.embeddings or not self.pages:
            raise ValueError("No ColPali index built or loaded to save.")

        target_dir = Path(index_dir) if index_dir else self.index_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save multi-vector embeddings tensor list
        emb_path = target_dir / "page_embeddings.pt"
        torch.save(
            {
                "embeddings": self.embeddings,
                "page_ids": [p["page_id"] for p in self.pages],
            },
            emb_path,
        )

        # 2. Save Pages Metadata JSON
        pages_path = target_dir / "pages.json"
        with open(pages_path, "w", encoding="utf-8") as f:
            json.dump(self.pages, f, indent=2, ensure_ascii=False)

        # 3. Save Index Metadata
        meta_path = target_dir / "index_meta.json"
        num_patches = self.embeddings[0].shape[0] if self.embeddings else 0
        emb_dim = self.embeddings[0].shape[1] if self.embeddings else 0
        metadata = {
            "model_name": self.model_name,
            "total_pages": len(self.pages),
            "embedding_dim": emb_dim,
            "num_patches_per_page": num_patches,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        return target_dir

    def load_index(self, index_dir: Optional[Union[str, Path]] = None) -> bool:
        """Load persisted ColPali index and page catalog from disk."""
        target_dir = Path(index_dir) if index_dir else self.index_dir
        emb_path = target_dir / "page_embeddings.pt"
        pages_path = target_dir / "pages.json"

        if not emb_path.exists() or not pages_path.exists():
            return False

        saved_data = torch.load(emb_path, map_location="cpu", weights_only=True)
        self.embeddings = saved_data["embeddings"]

        with open(pages_path, "r", encoding="utf-8") as f:
            self.pages = json.load(f)

        return True

    def retrieve(self, question: str, top_k: int = 5) -> List[VisualRetrievalResult]:
        """Retrieve top_k most relevant document pages using late-interaction MaxSim scoring."""
        if not self.embeddings or not self.pages:
            loaded = self.load_index()
            if not loaded or not self.embeddings:
                raise RuntimeError(
                    "No ColPali index available. Call build_index() or load_index() first."
                )

        if not question or not question.strip():
            return []

        batch_query = self.processor.process_queries([question]).to(self.device)
        batch_query.pop("labels", None)
        with torch.no_grad():
            query_embs = self.model(**batch_query)

        if self.device == "mps":
            torch.mps.empty_cache()

        # Convert query embedding to list of 2D CPU float32 tensors for late-interaction scoring
        query_embs_list = [query_embs[i].detach().cpu().to(torch.float32) for i in range(query_embs.shape[0])]

        # Compute late interaction MaxSim scores using ColPaliProcessor on CPU
        scores = self.processor.score_multi_vector(query_embs_list, self.embeddings, device="cpu")
        q_scores = scores[0]



        k = min(top_k, len(self.pages))
        top_scores, top_indices = torch.topk(q_scores, k=k)

        results: List[VisualRetrievalResult] = []
        for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
            page_meta = self.pages[idx]
            results.append(
                VisualRetrievalResult(
                    document_id=page_meta["document_id"],
                    page_number=page_meta["page_number"],
                    relevance_score=float(score),
                    image_path=page_meta["image_path"],
                )
            )

        return results
