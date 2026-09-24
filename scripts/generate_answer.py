#!/usr/bin/env python3
"""Run end-to-end grounded answer generation using HybridRetriever and VLMGenerator."""

import os
import sys
import time

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TQDM_DISABLE_MONITOR"] = "1"

import torch
torch.set_num_threads(1)

import argparse
from auditrag.generation import VLMGenerator
from auditrag.retrieval import HybridRetriever


def main():
    parser = argparse.ArgumentParser(description="AuditRAG Grounded VLM Answer Generation CLI.")
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default="What was the total assets from AMER in 2018?",
        help="Question to answer.",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=5,
        help="Number of hybrid retrieval candidates to supply to VLM (default: 5).",
    )
    parser.add_argument(
        "--provider",
        "-p",
        type=str,
        default=None,
        help="VLM provider to use ('mock', 'openai'). Defaults to config.",
    )
    args = parser.parse_args()

    print(f"=== AuditRAG — VLM Answer Generation ===")
    print(f"Question: {args.query}\n")

    # 1. Retrieval
    print(f"Step 1: Running Multimodal Hybrid Retrieval (Dense + BM25 + ColPali with RRF)...")
    t0 = time.time()
    retriever = HybridRetriever()
    results = retriever.retrieve(args.query, top_k=args.top_k)
    retrieval_time = time.time() - t0
    print(f"Retrieved {len(results)} hybrid candidates in {retrieval_time:.2f}s.\n")

    print(f"Retrieved Top Candidates:")
    for idx, r in enumerate(results, start=1):
        sources = ", ".join(r.sources)
        print(f"  [{idx}] Doc: {r.document_id[:12]}... | Page: {r.page_number} | Score: {r.score:.5f} | Sources: [{sources}]")
    print()

    # 2. VLM Generation
    print(f"Step 2: Generating Grounded VLM Answer...")
    t1 = time.time()
    generator = VLMGenerator(provider_name=args.provider)
    answer = generator.generate_answer(args.query, results)
    generation_time = time.time() - t1
    print(f"Generation completed in {generation_time:.2f}s using {answer.provider} ({answer.model}).\n")

    print(f"=== Grounded Answer ===")
    print(answer.answer)
    print()

    print(f"=== Evidence Summary ===")
    print(f"Total Evidence Items: {answer.evidence_count}")
    print(f"Images Provided:      {len(answer.images_provided)}")
    for img in answer.images_provided:
        print(f"  - {img}")


if __name__ == "__main__":
    main()
