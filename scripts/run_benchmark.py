#!/usr/bin/env python3
"""CLI script to run AuditRAG Evaluation & Benchmarking."""

import argparse
import os
import sys
import time

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TQDM_DISABLE_MONITOR"] = "1"

import torch
torch.set_num_threads(1)

from auditrag.evaluation import BenchmarkRunner


def main():
    parser = argparse.ArgumentParser(description="AuditRAG Evaluation & Benchmark CLI.")
    parser.add_argument(
        "--sample-size",
        "-n",
        type=int,
        default=20,
        help="Number of representative questions from TAT-DQA test split to evaluate (default: 20).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="reports",
        help="Directory to save reports and figures (default: reports).",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=5,
        help="Top-k retrieval cut-off (default: 5).",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Evaluate retrieval systems only.",
    )
    parser.add_argument(
        "--pipeline-only",
        action="store_true",
        help="Evaluate end-to-end pipeline only.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Suffix for output report files (e.g. '_improved').",
    )
    args = parser.parse_args()

    print("================================================================")
    print("             AuditRAG: Evaluation & Benchmarking                ")
    print("================================================================\n")

    runner = BenchmarkRunner(output_dir=args.output_dir)
    questions = runner.select_sample_questions(sample_size=args.sample_size)
    print(f"Selected {len(questions)} representative test questions from TAT-DQA test split.\n")

    run_all = not (args.retrieval_only or args.pipeline_only)

    # 1. Retrieval Benchmark
    if run_all or args.retrieval_only:
        print("--- Section 1: Multimodal Retrieval Benchmark ---")
        print("Comparing Dense, BM25, ColPali, and Hybrid RRF...")
        t0 = time.time()
        retrieval_report = runner.evaluate_retrieval(questions, top_k=args.top_k)
        print(f"Retrieval evaluation finished in {time.time() - t0:.2f}s.\n")

        print(f"{'Retriever':<15} | {'Doc-Hit@1':<9} | {'Doc-Hit@5':<9} | {'Doc-MRR':<8} | {'Page-Hit@1':<10} | {'Page-Hit@5':<10} | {'Page-MRR':<8} | {'Latency':<8}")
        print("-" * 95)
        for r in retrieval_report.results:
            print(
                f"{r.retriever_name:<15} | "
                f"{r.doc_hit_at_1*100:>8.1f}% | "
                f"{r.doc_hit_at_5*100:>8.1f}% | "
                f"{r.doc_mrr:>8.4f} | "
                f"{r.page_hit_at_1*100:>9.1f}% | "
                f"{r.page_hit_at_5*100:>9.1f}% | "
                f"{r.page_mrr:>8.4f} | "
                f"{r.avg_latency_ms:>6.1f}ms"
            )
        print("\n")

    # 2. Pipeline Benchmark
    if run_all or args.pipeline_only:
        print("--- Section 2: End-to-End Pipeline & Protection Benchmark ---")
        print("Running full pipeline (Hybrid → VLM → Citations → Verification → Protection)...")
        t1 = time.time()
        pipeline_report = runner.evaluate_pipeline(questions, top_k=args.top_k, suffix=args.suffix)
        print(f"Pipeline evaluation finished in {time.time() - t1:.2f}s.\n")

        print("Grounding & Quality Metrics:")
        print(f"  Exact Match (EM):    {pipeline_report.avg_exact_match * 100:.1f}%")
        print(f"  Token F1:            {pipeline_report.avg_token_f1 * 100:.1f}%")
        print(f"  Numeric Match:       {pipeline_report.avg_numeric_match * 100:.1f}%")
        print(f"  Average Latency:     {pipeline_report.avg_latency_ms:.1f} ms\n")

        v = pipeline_report.verification_summary
        print("Atomic Claim Verification Metrics:")
        print(f"  Total Claims Extracted:    {v.total_claims}")
        print(f"  Supported Claims:          {v.supported_claims} ({v.claim_support_rate * 100:.1f}%)")
        print(f"  Contradicted Claims:       {v.contradicted_claims} ({v.claim_contradiction_rate * 100:.1f}%)")
        print(f"  Unsupported Claims:        {v.not_supported_claims} ({v.claim_unsupported_rate * 100:.1f}%)")
        print(f"  Mean Factual Consistency:  {v.average_factual_consistency * 100:.1f}%\n")

        a = pipeline_report.abstention_summary
        print("Hallucination Protection & Abstention:")
        print(f"  Total Queries:       {a.total_evaluated}")
        print(f"  Accepted Answers:    {a.accepted_count}")
        print(f"  Abstained Answers:   {a.abstained_count}")
        print(f"  Abstention Rate:     {a.abstention_rate * 100:.1f}%")
        if a.reasons_breakdown:
            print("  Abstention Reasons:")
            for reason, count in a.reasons_breakdown.items():
                print(f"    - {reason}: {count}")

    print("\n================================================================")
    print(f"Artifacts successfully written to '{args.output_dir}/':")
    print(f"  - Tables:  {args.output_dir}/tables/retrieval_benchmark.md (.json)")
    print(f"             {args.output_dir}/tables/pipeline_benchmark.md (.json)")
    print(f"  - Figures: {args.output_dir}/figures/retrieval_comparison.png")
    print(f"             {args.output_dir}/figures/verification_abstention_breakdown.png")
    print("================================================================\n")


if __name__ == "__main__":
    main()
