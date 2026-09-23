"""Evaluation & Benchmarking module for AuditRAG Phase 11."""

from auditrag.evaluation.benchmark import (
    BenchmarkRunner,
    PipelineBenchmarkReport,
    PipelineSampleResult,
    RetrievalBenchmarkReport,
    RetrieverMetrics,
)
from auditrag.evaluation.metrics import (
    AbstentionMetricsSummary,
    RetrievalItem,
    VerificationMetricsSummary,
    compute_abstention_metrics,
    compute_hit_at_k,
    compute_mrr,
    compute_verification_metrics,
    exact_match_score,
    extract_numbers,
    gold_to_strings,
    normalize_text,
    numeric_match_score,
    token_f1_score,
)

__all__ = [
    "RetrievalItem",
    "compute_hit_at_k",
    "compute_mrr",
    "normalize_text",
    "extract_numbers",
    "gold_to_strings",
    "exact_match_score",
    "token_f1_score",
    "numeric_match_score",
    "VerificationMetricsSummary",
    "compute_verification_metrics",
    "AbstentionMetricsSummary",
    "compute_abstention_metrics",
    "RetrieverMetrics",
    "RetrievalBenchmarkReport",
    "PipelineSampleResult",
    "PipelineBenchmarkReport",
    "BenchmarkRunner",
]
