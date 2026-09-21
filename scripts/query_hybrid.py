import os
import sys
import time

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TQDM_DISABLE_MONITOR"] = "1"

import torch
torch.set_num_threads(1)

import argparse
from auditrag.retrieval import HybridRetriever


def main():
    parser = argparse.ArgumentParser(description="Query AuditRAG Multimodal Hybrid Retriever.")
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default="What was the total assets from AMER in 2018?",
        help="Query string to search.",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=5,
        help="Number of hybrid results to return (default: 5).",
    )
    args = parser.parse_args()

    print(f"=== Initializing Hybrid Retriever (Dense + BM25 + ColPali) ===")
    start_init = time.time()
    retriever = HybridRetriever()
    # Eagerly load indices
    _ = retriever.dense_retriever
    _ = retriever.bm25_retriever
    _ = retriever.colpali_retriever
    init_time = time.time() - start_init
    print(f"All indices loaded in {init_time:.2f}s.")

    print(f"\nQuery: {args.query}")
    print(f"Executing RRF fusion across modalities...")
    start_q = time.time()
    results = retriever.retrieve(args.query, top_k=args.top_k)
    query_time = time.time() - start_q
    print(f"Retrieval completed in {query_time:.3f}s. Top {len(results)} results:\n")

    for rank, res in enumerate(results, start=1):
        sources_str = ", ".join(res.sources)
        print(f"--- Rank {rank} (Fused RRF Score: {res.score:.6f}) ---")
        print(f"Document ID:   {res.document_id}")
        print(f"Page Number:   {res.page_number}")
        print(f"Sources:       [{sources_str}]")
        print(f"Source Ranks:  {res.source_ranks}")
        print(f"Source Scores: {res.source_scores}")
        if res.image_path:
            print(f"Image Path:    {res.image_path}")
        if res.visual_score is not None:
            print(f"Visual Score:  {res.visual_score:.4f}")
        if res.chunk_text:
            text_preview = res.chunk_text.replace("\n", " ")[:200]
            print(f"Text Preview:  {text_preview}...")
        print()


if __name__ == "__main__":
    main()
