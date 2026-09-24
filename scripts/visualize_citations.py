#!/usr/bin/env python3
"""CLI script demonstrating Visual Bounding-Box Citations for AuditRAG."""

import os
import sys
import time

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TQDM_DISABLE_MONITOR"] = "1"

import torch
torch.set_num_threads(1)

import argparse
from auditrag.citations import VisualCitationEngine
from auditrag.generation import VLMGenerator
from auditrag.retrieval import HybridRetriever


def main():
    parser = argparse.ArgumentParser(description="AuditRAG — Visual Bounding-Box Citations CLI.")
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default="What was the total assets from AMER in 2018?",
        help="Question to answer and cite.",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=5,
        help="Top-k retrieval candidates to use (default: 5).",
    )
    args = parser.parse_args()

    print(f"=== AuditRAG — Visual Bounding-Box Citations ===")
    print(f"Question: {args.query}\n")

    # 1. Retrieve
    print(f"Step 1: Retrieving multimodal candidates (Hybrid Dense + BM25 + ColPali)...")
    t0 = time.time()
    retriever = HybridRetriever()
    results = retriever.retrieve(args.query, top_k=args.top_k)
    print(f"Retrieved {len(results)} candidates in {time.time() - t0:.2f}s.\n")

    # 2. Generate Grounded Answer
    print(f"Step 2: Generating grounded answer with VLM...")
    t1 = time.time()
    generator = VLMGenerator()
    answer = generator.generate_answer(args.query, results)
    print(f"Generated answer in {time.time() - t1:.2f}s.\n")
    print(f"Answer: {answer.answer}\n")

    # 3. Extract Visual Citations & Render Overlay
    print(f"Step 3: Mapping answer facts to TAT-DQA bounding boxes & rendering overlay...")
    t2 = time.time()
    engine = VisualCitationEngine()
    cited_answer = engine.create_citations_for_answer(answer, results)
    print(f"Extracted {len(cited_answer.citations)} visual citations in {time.time() - t2:.2f}s.\n")

    print(f"=== Visual Citations ===")
    for cit in cited_answer.citations:
        print(f"Citation {cit.label}:")
        print(f"  Document: {cit.document_id}")
        print(f"  Page:     {cit.page_number}")
        print(f"  Text:     '{cit.text}'")
        print(f"  BBox:     {cit.bbox} (x0, y0, x1, y1)")
        if cit.overlay_image_path:
            print(f"  Overlay:  {cit.overlay_image_path}")
        print()

    print(f"=== Generated Overlay Images ===")
    for img in cited_answer.overlay_images:
        print(f"  Annotated Page Image: {img}")


if __name__ == "__main__":
    main()
