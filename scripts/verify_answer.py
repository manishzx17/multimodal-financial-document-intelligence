#!/usr/bin/env python3
"""CLI script demonstrating Evidence & Numerical Verification for AuditRAG."""

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
from auditrag.verification import VerifierEngine


def main():
    parser = argparse.ArgumentParser(description="AuditRAG — Evidence & Numerical Verification CLI.")
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default="What was the total assets from AMER in 2018?",
        help="Question to answer, cite, and verify.",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=5,
        help="Top-k retrieval candidates (default: 5).",
    )
    args = parser.parse_args()

    print(f"=== AuditRAG — Evidence & Numerical Verification ===")
    print(f"Question: {args.query}\n")

    # 1. Retrieval
    print(f"Step 1: Retrieving multimodal evidence (Hybrid Dense + BM25 + ColPali)...")
    t0 = time.time()
    retriever = HybridRetriever()
    results = retriever.retrieve(args.query, top_k=args.top_k)
    print(f"Retrieved {len(results)} candidates in {time.time() - t0:.2f}s.\n")

    # 2. VLM Generation
    print(f"Step 2: Generating grounded answer with VLM...")
    t1 = time.time()
    generator = VLMGenerator()
    answer = generator.generate_answer(args.query, results)
    print(f"Generated answer in {time.time() - t1:.2f}s.\n")
    print(f"Answer: {answer.answer}\n")

    # 3. Visual Citations
    print(f"Step 3: Extracting visual citations & generating overlay...")
    t2 = time.time()
    citation_engine = VisualCitationEngine()
    cited_answer = citation_engine.create_citations_for_answer(answer, results)
    print(f"Extracted {len(cited_answer.citations)} visual citations in {time.time() - t2:.2f}s.\n")

    # 4. Evidence & Numerical Verification
    print(f"Step 4: Deconstructing claims and auditing against evidence...")
    t3 = time.time()
    verifier = VerifierEngine()
    report = verifier.verify_answer(answer, results, cited_answer.citations)
    print(f"Verification completed in {time.time() - t3:.2f}s.\n")

    print(f"=== Verification Audit Report ===")
    print(f"Overall Status:            {report.overall_status.value}")
    print(f"Factual Consistency Score: {report.factual_consistency_score * 100:.1f}%")
    print(f"Supported Claims:          {report.supported_count}/{len(report.claims)}")
    print(f"Contradicted Claims:       {report.contradicted_count}/{len(report.claims)}")
    print(f"Not Supported Claims:      {report.not_supported_count}/{len(report.claims)}\n")

    print(f"=== Detailed Claim Audits ===")
    for idx, res in enumerate(report.results, start=1):
        claim = res.claim
        badge = f" [Citation: {res.citation_label}]" if res.citation_label else ""
        print(f"--- Claim {idx}: {res.status.value}{badge} ---")
        print(f"Statement:   '{claim.statement}'")
        print(f"Type:        {claim.claim_type.value}")
        if claim.extracted_numbers:
            print(f"Numbers:     {claim.extracted_numbers}")
        print(f"Explanation: {res.explanation}")
        if res.arithmetic_check:
            print(f"Arithmetic:  {res.arithmetic_check.get('expression')} (Match: {res.arithmetic_check.get('verified')})")
        if res.evidence_snippet:
            snippet_preview = res.evidence_snippet.replace('\n', ' ')[:140]
            print(f"Evidence:    \"{snippet_preview}...\"")
        print()


if __name__ == "__main__":
    main()
