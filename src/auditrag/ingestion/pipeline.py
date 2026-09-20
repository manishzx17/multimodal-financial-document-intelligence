"""Document ingestion pipeline for AuditRAG Phase 2."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from auditrag.config import (
    PROCESSED_DOCUMENTS_DIR,
    RAW_DATA_DIR,
    RAW_TEST_DIR,
    RAW_TEST_GOLD_JSON,
    RAW_TEST_JSON,
)
from auditrag.ingestion.models import NormalizedDocument
from auditrag.ingestion.pdf_parser import TATDQAParser


class DocumentIngestionPipeline:
    """Pipeline for converting raw TAT-DQA data into normalized document representations."""

    def __init__(
        self,
        raw_test_dir: Union[str, Path] = RAW_TEST_DIR,
        output_dir: Union[str, Path] = PROCESSED_DOCUMENTS_DIR,
        metadata_file: Optional[Union[str, Path]] = None,
    ):
        self.raw_test_dir = Path(raw_test_dir)
        self.output_dir = Path(output_dir)
        self.metadata_file = Path(metadata_file) if metadata_file else self._find_metadata_file()
        self.parser = TATDQAParser(raw_test_dir=self.raw_test_dir)

    def _find_metadata_file(self) -> Path:
        """Locate raw dataset metadata JSON file."""
        if RAW_TEST_GOLD_JSON.exists():
            return RAW_TEST_GOLD_JSON
        if RAW_TEST_JSON.exists():
            return RAW_TEST_JSON
        return RAW_DATA_DIR / "tatdqa_dataset_test.json"

    def get_document_catalog(self) -> Dict[str, Dict[str, Any]]:
        """Retrieve catalog of document metadata (doc_uid -> metadata) from raw dataset."""
        catalog: Dict[str, Dict[str, Any]] = {}
        if self.metadata_file.exists():
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                raw_items = json.load(f)
            for item in raw_items:
                d_info = item.get("doc", {})
                d_uid = d_info.get("uid")
                if d_uid and d_uid not in catalog:
                    catalog[d_uid] = d_info

        # Supplement with any standalone JSONs found in test directory
        if self.raw_test_dir.exists():
            for json_path in sorted(self.raw_test_dir.glob("*.json")):
                doc_uid = json_path.stem
                if doc_uid not in catalog:
                    catalog[doc_uid] = {"uid": doc_uid}

        return catalog

    def ingest_document(
        self,
        doc_uid: str,
        source: Optional[str] = None,
    ) -> NormalizedDocument:
        """Ingest a single document and return its normalized representation."""
        if source is None:
            catalog = self.get_document_catalog()
            source = catalog.get(doc_uid, {}).get("source")
        return self.parser.parse_document(doc_uid=doc_uid, source=source)

    def save_processed_document(
        self,
        doc: NormalizedDocument,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Save a normalized document to JSON format in data/processed/documents/."""
        target_dir = Path(output_dir) if output_dir else self.output_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        file_path = target_dir / f"{doc.document_id}.json"
        doc.save_to_file(file_path)
        return file_path

    def load_processed_document(
        self,
        doc_id: str,
        input_dir: Optional[Union[str, Path]] = None,
    ) -> NormalizedDocument:
        """Load a processed document JSON by document ID."""
        target_dir = Path(input_dir) if input_dir else self.output_dir
        file_path = target_dir / f"{doc_id}.json"
        return NormalizedDocument.from_file(file_path)

    def load_all_processed_documents(
        self,
        input_dir: Optional[Union[str, Path]] = None,
    ) -> List[NormalizedDocument]:
        """Load all processed documents from the processed directory."""
        target_dir = Path(input_dir) if input_dir else self.output_dir
        if not target_dir.exists():
            return []

        docs: List[NormalizedDocument] = []
        for file_path in sorted(target_dir.glob("*.json")):
            if file_path.name == "manifest.json":
                continue
            docs.append(NormalizedDocument.from_file(file_path))
        return docs

    def ingest_all(
        self,
        limit: Optional[int] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Batch ingest raw documents, serialize normalized JSONs, and write manifest.json."""
        target_dir = Path(output_dir) if output_dir else self.output_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        catalog = self.get_document_catalog()
        doc_uids = sorted(catalog.keys())
        if limit is not None:
            doc_uids = doc_uids[:limit]

        processed_docs_meta: List[Dict[str, Any]] = []
        total_pages = 0
        total_blocks = 0

        for doc_uid in doc_uids:
            source = catalog.get(doc_uid, {}).get("source")
            norm_doc = self.ingest_document(doc_uid=doc_uid, source=source)
            self.save_processed_document(norm_doc, output_dir=target_dir)

            doc_blocks = sum(len(p.blocks) for p in norm_doc.pages)
            total_pages += norm_doc.page_count
            total_blocks += doc_blocks

            processed_docs_meta.append({
                "document_id": norm_doc.document_id,
                "source": norm_doc.source,
                "page_count": norm_doc.page_count,
                "block_count": doc_blocks,
                "pdf_path": norm_doc.pdf_path,
                "pages": [
                    {
                        "page_number": p.page_number,
                        "block_count": len(p.blocks),
                        "page_image": p.page_image,
                    }
                    for p in norm_doc.pages
                ],
                "file_name": f"{norm_doc.document_id}.json",
            })

        manifest = {
            "version": "1.0.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_documents": len(processed_docs_meta),
            "total_pages": total_pages,
            "total_blocks": total_blocks,
            "documents": processed_docs_meta,
        }

        manifest_path = target_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        return {
            "total_documents": len(processed_docs_meta),
            "total_pages": total_pages,
            "total_blocks": total_blocks,
            "output_dir": str(target_dir),
            "manifest_path": str(manifest_path),
        }
