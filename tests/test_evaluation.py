"""Unit and integration tests for Phase 11 Evaluation & Benchmarking."""

from pathlib import Path
import pytest
from pydantic import BaseModel

from auditrag.evaluation.metrics import (
    RetrievalItem,
    compute_abstention_metrics,
    compute_hit_at_k,
    compute_mrr,
    compute_verification_metrics,
    exact_match_score,
    extract_numbers,
    normalize_text,
    numeric_match_score,
    token_f1_score,
)
from auditrag.evaluation.benchmark import BenchmarkRunner, RetrieverMetrics, RetrievalBenchmarkReport
from auditrag.verification.models import VerificationReport, VerificationStatus


def test_retrieval_hit_at_k():
    """Verify document and page level hit@k calculations."""
    gold_doc = "doc_123"
    gold_pages = {2, 4}

    retrieved = [
        RetrievalItem(document_id="doc_other", page_number=1),
        RetrievalItem(document_id="doc_123", page_number=1),   # Match doc, not page
        RetrievalItem(document_id="doc_123", page_number=2),   # Match doc and page
        RetrievalItem(document_id="doc_xyz", page_number=3),
    ]

    # At k=1: neither doc nor page hit
    d1, p1 = compute_hit_at_k(retrieved, gold_doc, gold_pages, k=1)
    assert d1 == 0.0
    assert p1 == 0.0

    # At k=2: doc hit, but page not hit
    d2, p2 = compute_hit_at_k(retrieved, gold_doc, gold_pages, k=2)
    assert d2 == 1.0
    assert p2 == 0.0

    # At k=3: both doc and page hit
    d3, p3 = compute_hit_at_k(retrieved, gold_doc, gold_pages, k=3)
    assert d3 == 1.0
    assert p3 == 1.0


def test_retrieval_mrr():
    """Verify Mean Reciprocal Rank (MRR) computation."""
    gold_doc = "doc_target"
    gold_pages = {3}

    retrieved = [
        RetrievalItem(document_id="doc_other", page_number=1),
        RetrievalItem(document_id="doc_target", page_number=1),  # rank 2 for doc
        RetrievalItem(document_id="doc_target", page_number=3),  # rank 3 for page
    ]

    doc_mrr, page_mrr = compute_mrr(retrieved, gold_doc, gold_pages)
    assert doc_mrr == pytest.approx(0.5)  # 1/2
    assert page_mrr == pytest.approx(1.0 / 3.0)  # 1/3


def test_answer_exact_match_score():
    """Verify Exact Match normalization across currency, punctuation, and casing."""
    assert exact_match_score("In 2018, total assets were $645,791.", ["645,791"]) == 1.0
    assert exact_match_score("Revenue was $1,200", 1200) == 1.0
    assert exact_match_score("No match here", ["645,791"]) == 0.0


def test_token_f1_score():
    """Verify token-level precision/recall/F1 calculation."""
    pred = "depreciation for AMER was 21224 thousand"
    gold = "21224 thousand"
    f1 = token_f1_score(pred, gold)
    assert f1 >= 0.5

    # Completely disjoint
    assert token_f1_score("apple orange", "banana grape") == 0.0


def test_numeric_match_score():
    """Verify numeric match extraction and tolerance."""
    pred = "Depreciation was $21,224.50 in 2018 and total was $48,095."
    # Matches single target number
    assert numeric_match_score(pred, 21224.5) == 1.0
    # Matches target within percentage tolerance
    assert numeric_match_score(pred, 21224.0, tolerance=1.0) == 1.0
    # Multiple target numbers
    assert numeric_match_score(pred, [21224.5, 48095.0]) == 1.0
    # Missing number
    assert numeric_match_score(pred, 999999.0) == 0.0


def test_verification_and_abstention_metrics():
    """Verify aggregation of verification reports and abstention decisions."""
    r1 = VerificationReport(
        question="q1",
        answer="a1",
        claims=[],
        results=[],
        overall_status=VerificationStatus.SUPPORTED,
        supported_count=2,
        contradicted_count=0,
        not_supported_count=0,
        factual_consistency_score=1.0,
    )
    r2 = VerificationReport(
        question="q2",
        answer="a2",
        claims=[],
        results=[],
        overall_status=VerificationStatus.CONTRADICTED,
        supported_count=0,
        contradicted_count=1,
        not_supported_count=1,
        factual_consistency_score=0.0,
    )

    v_summary = compute_verification_metrics([r1, r2])
    assert v_summary.total_claims == 4
    assert v_summary.supported_claims == 2
    assert v_summary.contradicted_claims == 1
    assert v_summary.not_supported_claims == 1
    assert v_summary.claim_support_rate == 0.5
    assert v_summary.average_factual_consistency == 0.5

    # Test abstention metrics
    class MockResp(BaseModel):
        is_abstained: bool
        abstention_reason: str

    responses = [
        MockResp(is_abstained=False, abstention_reason=""),
        MockResp(is_abstained=True, abstention_reason="CONTRADICTED_CLAIM"),
        MockResp(is_abstained=True, abstention_reason="CONTRADICTED_CLAIM"),
        MockResp(is_abstained=True, abstention_reason="FAILED_ARITHMETIC_CHECK"),
    ]

    a_summary = compute_abstention_metrics(responses)
    assert a_summary.total_evaluated == 4
    assert a_summary.accepted_count == 1
    assert a_summary.abstained_count == 3
    assert a_summary.abstention_rate == 0.75
    assert a_summary.reasons_breakdown == {
        "CONTRADICTED_CLAIM": 2,
        "FAILED_ARITHMETIC_CHECK": 1,
    }


def test_benchmark_runner_mock(tmp_path: Path):
    """Verify BenchmarkRunner executes and saves report tables and plots using mock components."""
    runner = BenchmarkRunner(output_dir=str(tmp_path))

    # Mock retrieval report
    mock_retrieval_report = RetrievalBenchmarkReport(
        sample_size=2,
        results=[
            RetrieverMetrics(
                retriever_name="MockRetriever",
                sample_size=2,
                doc_hit_at_1=1.0,
                doc_hit_at_3=1.0,
                doc_hit_at_5=1.0,
                doc_mrr=1.0,
                page_hit_at_1=0.5,
                page_hit_at_3=1.0,
                page_hit_at_5=1.0,
                page_mrr=0.75,
                avg_latency_ms=12.5,
            )
        ]
    )

    runner._save_retrieval_reports(mock_retrieval_report)
    runner._plot_retrieval_comparison(mock_retrieval_report)

    assert (tmp_path / "tables" / "retrieval_benchmark.md").exists()
    assert (tmp_path / "tables" / "retrieval_benchmark.json").exists()
    assert (tmp_path / "figures" / "retrieval_comparison.png").exists()
