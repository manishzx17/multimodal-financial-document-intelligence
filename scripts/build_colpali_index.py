"""Script to build and verify ColPali visual index across all 312 pages."""

import os
import sys
import time

os.environ["TQDM_DISABLE_MONITOR"] = "1"

import torch
from auditrag.config import COLPALI_INDEX_DIR
from auditrag.retrieval import ColPaliRetriever

def main():
    print("=== ColPali Visual Indexing ===")
    retriever = ColPaliRetriever(index_dir=COLPALI_INDEX_DIR)
    pages = retriever.discover_pages()
    print(f"Discovered {len(pages)} page images.")

    print("Loading ColPali model onto device:", retriever.device)
    _ = retriever.model
    print("Model loaded.")

    print("Encoding pages in batches...")
    t0 = time.time()
    count = retriever.build_index(page_records=pages, batch_size=2, show_progress=True)
    elapsed = time.time() - t0

    print(f"Successfully encoded {count} pages in {elapsed:.2f}s (avg {elapsed/count:.2f}s/page).")

    save_path = retriever.save_index()
    print(f"ColPali index saved to {save_path}")

    query = "What was the total assets from AMER in 2018?"
    print(f"\nRunning example retrieval for query: '{query}'")
    results = retriever.retrieve(query, top_k=5)

    for i, res in enumerate(results, 1):
        print(f"\n--- Result {i} (Relevance Score: {res.relevance_score:.4f}) ---")
        print(f"Document ID: {res.document_id}")
        print(f"Page Number: {res.page_number}")
        print(f"Image Path: {res.image_path}")

if __name__ == "__main__":
    main()
