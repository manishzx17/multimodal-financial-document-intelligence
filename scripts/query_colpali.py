"""Query test for saved ColPali index."""

import os
import sys
import time

os.environ["TQDM_DISABLE_MONITOR"] = "1"

import torch
from auditrag.config import COLPALI_INDEX_DIR
from auditrag.retrieval import ColPaliRetriever


def main():
    retriever = ColPaliRetriever(index_dir=COLPALI_INDEX_DIR)
    print("Pre-loading ColPali model on device:", retriever.device)
    _ = retriever.model
    print("Model ready. Loading index from", COLPALI_INDEX_DIR)

    loaded = retriever.load_index()
    if not loaded:
        print("Error: Could not load index from", COLPALI_INDEX_DIR)
        sys.exit(1)

    print(f"Loaded {len(retriever.pages)} pages and {len(retriever.embeddings)} multi-vector embeddings.")
    query = "What was the total assets from AMER in 2018?"
    print(f"Querying: '{query}'...")
    t0 = time.time()
    results = retriever.retrieve(query, top_k=5)
    elapsed = time.time() - t0
    print(f"Retrieved {len(results)} results in {elapsed:.3f}s:\n")

    for i, res in enumerate(results, 1):
        print(f"--- Result {i} (Score: {res.relevance_score:.4f}) ---")
        print(f"Document ID: {res.document_id}")
        print(f"Page Number: {res.page_number}")
        print(f"Image Path: {res.image_path}")

if __name__ == "__main__":
    main()
