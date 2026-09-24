#!/usr/bin/env python3
"""CLI script demonstrating the integrated AuditRAG pipeline with Hallucination Protection."""

import os
import sys
import time

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TQDM_DISABLE_MONITOR"] = "1"

import torch
torch.set_num_threads(1)

import argparse
from auditrag.generation.vlm import GeneratedAnswer
from auditrag.pipeline import AuditRAGPipeline


def main():
    parser = argparse.ArgumentParser(description="AuditRAG — Pipeline & Hallucination Protection CLI.")
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default="What was the total assets from AMER in 2018?",
        help="Question to query through AuditRAG.",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=5,
        help="Top-k retrieval candidates (default: 5).",
    )
    parser.add_argument(
        "--provider",
        "-p",
        type=str,
        default=None,
        choices=["openai", "mock"],
        help="VLM Provider to use (default: openai if OPENAI_API_KEY set, else mock).",
    )
    args = parser.parse_args()

    from auditrag.generation.vlm import VLMGenerator
    gen = VLMGenerator(provider_name=args.provider)
    pipeline = AuditRAGPipeline(generator=gen)

    provider_name = gen.provider.__class__.__name__
    has_openai_key = bool(os.getenv("OPENAI_API_KEY"))

    print(f"================================================================")
    print(f"       AuditRAG Grounded Pipeline & Protection Engine           ")
    print(f"================================================================")
    print(f"VLM Provider Active: {provider_name}")
    if provider_name == "MockVLMProvider" and not has_openai_key:
        print(f"[Note: OPENAI_API_KEY not set. Using offline MockVLMProvider.]")
        print(f"[To use real OpenAI Vision: export OPENAI_API_KEY=your_key --provider openai]\n")
    else:
        print()

    # CASE 1: Standard / Accepted execution
    print(f"--- DEMO 1: Grounded & Verified Query (Expected: ACCEPTED) ---")
    print(f"Query: {args.query}")
    t0 = time.time()
    resp1 = pipeline.run(args.query, top_k=args.top_k)
    print(f"Pipeline executed in {time.time() - t0:.2f}s.\n")

    print(f"Decision:         {'[ABSTAINED]' if resp1.is_abstained else '[ACCEPTED]'}")
    if resp1.is_abstained:
        print(f"Abstention Cause: {resp1.abstention_reason}")
    print(f"Final Answer:     {resp1.final_answer}")
    print(f"Consistency:      {resp1.verification_report.factual_consistency_score * 100:.1f}% "
          f"({resp1.verification_report.supported_count}/{len(resp1.verification_report.claims)} supported)")
    print(f"Visual Citations: {len(resp1.answer_with_citations.citations)}")
    print(f"Overlay Images:   {resp1.answer_with_citations.overlay_images}\n")

    # CASE 2: Hallucination / Contradicted execution
    print(f"--- DEMO 2: Hallucination Protection Gate (Expected: ABSTAINED) ---")
    contradictory_query = "What was the total assets from AMER in 2018?"
    print(f"Query: {contradictory_query}")
    print(f"(Injecting hallucinated/contradictory number to test safety gating...)")

    # Mock an unfaithful VLM output for demonstration of safety protection
    class HallucinatingVLM:
        def generate_answer(self, q, results):
            top_doc = results[0].document_id if results else "637fab7088ea6c78a5dba55f17e833bd"
            return GeneratedAnswer(
                question=q,
                answer=f"Based on financial disclosures, in 2018 depreciation for AMER was $999,999 thousand [Doc: {top_doc}, Page: 1].",
                sources=[{"document_id": top_doc, "page_number": 1}],
                model="mock-hallucinating-vlm",
                provider="Mock",
                evidence_count=len(results),
            )

    protected_pipeline = AuditRAGPipeline(generator=HallucinatingVLM())
    t1 = time.time()
    resp2 = protected_pipeline.run(contradictory_query, top_k=args.top_k)
    print(f"Pipeline executed in {time.time() - t1:.2f}s.\n")

    print(f"Decision:         {'[ABSTAINED]' if resp2.is_abstained else '[ACCEPTED]'}")
    print(f"Abstention Cause: {resp2.abstention_reason}")
    print(f"Final Answer:     \"{resp2.final_answer}\"")
    print(f"Consistency:      {resp2.verification_report.factual_consistency_score * 100:.1f}% "
          f"({resp2.verification_report.supported_count}/{len(resp2.verification_report.claims)} supported)")
    print(f"Audit Status:     {resp2.verification_report.overall_status.value}")
    print(f"Report Preserved: Yes (Claims: {len(resp2.verification_report.claims)}, "
          f"Contradicted: {resp2.verification_report.contradicted_count}, "
          f"Unsupported: {resp2.verification_report.not_supported_count})")
    print(f"\n================================================================")


if __name__ == "__main__":
    main()
