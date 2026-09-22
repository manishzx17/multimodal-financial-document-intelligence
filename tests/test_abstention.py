"""Unit and integration tests for Phase 10 Hallucination Protection & Abstention."""

import pytest

from auditrag.config import DEFAULT_ABSTENTION_MESSAGE, PROCESSED_DOCUMENTS_DIR
from auditrag.generation.vlm import GeneratedAnswer
from auditrag.ingestion.models import NormalizedDocument
from auditrag.pipeline import AuditRAGPipeline, AuditRAGResponse
from auditrag.retrieval.hybrid import HybridRetrievalResult
from auditrag.verification import (
    AbstentionDecision,
    AbstentionReason,
    Claim,
    ClaimType,
    HallucinationProtectionEngine,
    VerificationReport,
    VerificationResult,
    VerificationStatus,
)


@pytest.fixture
def sample_doc() -> NormalizedDocument:
    doc_path = PROCESSED_DOCUMENTS_DIR / "637fab7088ea6c78a5dba55f17e833bd.json"
    if not doc_path.exists():
        pytest.skip(f"Processed document not found: {doc_path}")
    return NormalizedDocument.from_file(doc_path)


def test_abstention_accepted_supported():
    """Verify that fully supported verification results are accepted without modification."""
    engine = HallucinationProtectionEngine()

    claim = Claim(
        claim_id="c1",
        statement="Depreciation for AMER was $21,224",
        claim_type=ClaimType.NUMERICAL,
        extracted_numbers=[21224.0],
    )
    result = VerificationResult(
        claim=claim,
        status=VerificationStatus.SUPPORTED,
        confidence=1.0,
        explanation="Fact verified in disclosure.",
    )
    report = VerificationReport(
        question="What was depreciation for AMER in 2018?",
        answer="Depreciation for AMER was $21,224.",
        claims=[claim],
        results=[result],
        overall_status=VerificationStatus.SUPPORTED,
        supported_count=1,
        contradicted_count=0,
        not_supported_count=0,
        factual_consistency_score=1.0,
    )

    decision = engine.evaluate_and_protect(report, overlay_images=["overlay1.png"])

    assert decision.is_abstained is False
    assert decision.final_answer == report.answer
    assert decision.abstention_reason == AbstentionReason.NONE
    assert decision.confidence_score == 1.0
    assert decision.overlay_images == ["overlay1.png"]
    assert decision.verification_report == report


def test_abstention_contradicted_claim():
    """Verify that contradicted claims cause abstention and preserve the report."""
    engine = HallucinationProtectionEngine()

    claim = Claim(
        claim_id="c1",
        statement="Depreciation for AMER was 999,999",
        claim_type=ClaimType.NUMERICAL,
        extracted_numbers=[999999.0],
    )
    result = VerificationResult(
        claim=claim,
        status=VerificationStatus.CONTRADICTED,
        confidence=0.9,
        explanation="Contradicted by financial statement.",
    )
    report = VerificationReport(
        question="What was depreciation for AMER in 2018?",
        answer="Depreciation for AMER was $999,999.",
        claims=[claim],
        results=[result],
        overall_status=VerificationStatus.CONTRADICTED,
        supported_count=0,
        contradicted_count=1,
        not_supported_count=0,
        factual_consistency_score=0.0,
    )

    decision = engine.evaluate_and_protect(report, overlay_images=["overlay1.png"])

    assert decision.is_abstained is True
    assert decision.final_answer == DEFAULT_ABSTENTION_MESSAGE
    assert decision.abstention_reason == AbstentionReason.CONTRADICTED_CLAIM
    assert decision.verification_report == report
    assert decision.overlay_images == ["overlay1.png"]


def test_abstention_unsupported_numerical_claim():
    """Verify that unsupported numerical figures trigger abstention."""
    engine = HallucinationProtectionEngine()

    claim = Claim(
        claim_id="c1",
        statement="Solar energy revenue reached 77,888",
        claim_type=ClaimType.NUMERICAL,
        extracted_numbers=[77888.0],
    )
    result = VerificationResult(
        claim=claim,
        status=VerificationStatus.NOT_SUPPORTED,
        confidence=0.3,
        explanation="Stated numerical figures could not be located in retrieved evidence.",
    )
    report = VerificationReport(
        question="What was solar energy revenue?",
        answer="Solar energy revenue reached 77,888.",
        claims=[claim],
        results=[result],
        overall_status=VerificationStatus.NOT_SUPPORTED,
        supported_count=0,
        contradicted_count=0,
        not_supported_count=1,
        factual_consistency_score=0.0,
    )

    decision = engine.evaluate_and_protect(report)

    assert decision.is_abstained is True
    assert decision.final_answer == DEFAULT_ABSTENTION_MESSAGE
    assert decision.abstention_reason == AbstentionReason.UNSUPPORTED_NUMERICAL_FACT


def test_abstention_failed_arithmetic_check():
    """Verify that failed arithmetic checks trigger abstention."""
    engine = HallucinationProtectionEngine()

    claim = Claim(
        claim_id="c1",
        statement="Operating income increased by 50.0 from 38.6 to 57.8",
        claim_type=ClaimType.CALCULATION,
        extracted_numbers=[50.0, 38.6, 57.8],
    )
    result = VerificationResult(
        claim=claim,
        status=VerificationStatus.SUPPORTED,
        confidence=0.8,
        arithmetic_check={
            "verified": False,
            "operation": "variance_check",
            "expression": "57.8 - 38.6 != 50.0",
            "explanation": "Arithmetic error.",
        },
        explanation="Calculation mismatch.",
    )
    report = VerificationReport(
        question="How much did operating income increase?",
        answer="Operating income increased by 50.0 from 38.6 to 57.8.",
        claims=[claim],
        results=[result],
        overall_status=VerificationStatus.SUPPORTED,
        supported_count=1,
        contradicted_count=0,
        not_supported_count=0,
        factual_consistency_score=1.0,
    )

    decision = engine.evaluate_and_protect(report)

    assert decision.is_abstained is True
    assert decision.final_answer == DEFAULT_ABSTENTION_MESSAGE
    assert decision.abstention_reason == AbstentionReason.FAILED_ARITHMETIC_CHECK


def test_abstention_insufficient_evidence():
    """Verify that empty or zero-supported reports trigger abstention."""
    engine = HallucinationProtectionEngine()
    report = VerificationReport(
        question="What were the 2025 forecasts?",
        answer="Forecasts are unannounced.",
        claims=[],
        results=[],
        overall_status=VerificationStatus.NOT_SUPPORTED,
        supported_count=0,
        contradicted_count=0,
        not_supported_count=0,
        factual_consistency_score=0.0,
    )

    decision = engine.evaluate_and_protect(report)

    assert decision.is_abstained is True
    assert decision.final_answer == DEFAULT_ABSTENTION_MESSAGE
    assert decision.abstention_reason == AbstentionReason.INSUFFICIENT_EVIDENCE


def test_auditrag_pipeline_accepted_and_abstained(sample_doc: NormalizedDocument):
    """Verify integrated pipeline execution for both accepted and abstained cases."""
    pipeline = AuditRAGPipeline()
    doc_id = sample_doc.document_id

    # 1. Accepted scenario: Supported facts from sample_doc
    class MockSupportedGenerator:
        def generate_answer(self, q, results):
            return GeneratedAnswer(
                question=q,
                answer=f"In 2018 depreciation for AMER was $21,224 [Doc: {doc_id}, Page: 1].",
                sources=[{"document_id": doc_id, "page_number": 1}],
                model="mock-vlm",
                provider="Mock",
                evidence_count=1,
            )

    pipeline.generator = MockSupportedGenerator()
    resp_accepted = pipeline.run("What was the depreciation for AMER in 2018?", top_k=2)

    assert isinstance(resp_accepted, AuditRAGResponse)
    assert resp_accepted.is_abstained is False
    assert resp_accepted.abstention_reason is None
    assert "$21,224" in resp_accepted.final_answer
    assert resp_accepted.verification_report.supported_count >= 1

    # 2. Abstained scenario: Hallucinated/contradicted claim
    class MockHallucinatingGenerator:
        def generate_answer(self, q, results):
            return GeneratedAnswer(
                question=q,
                answer=f"In 2018 depreciation for AMER was $999,999 [Doc: {doc_id}, Page: 1].",
                sources=[{"document_id": doc_id, "page_number": 1}],
                model="mock-vlm",
                provider="Mock",
                evidence_count=1,
            )

    pipeline.generator = MockHallucinatingGenerator()
    resp_abstained = pipeline.run("What was the depreciation for AMER in 2018?", top_k=2)

    assert isinstance(resp_abstained, AuditRAGResponse)
    assert resp_abstained.is_abstained is True
    assert resp_abstained.final_answer == DEFAULT_ABSTENTION_MESSAGE
    assert resp_abstained.abstention_reason in ("CONTRADICTED_CLAIM", "UNSUPPORTED_NUMERICAL_FACT")
    assert resp_abstained.verification_report is not None
    assert resp_abstained.answer_with_citations is not None


def test_relevance_gate_wrong_metric_supported_abstains():
    """Test A from audit: Supported facts about wrong metric must be rejected with IRRELEVANT_ANSWER."""
    from auditrag.verification.engine import VerifierEngine

    question = "What was the total assets from AMER in 2018?"
    answer_text = "Depreciation for AMER was $21,224 thousand."
    gen_answer = GeneratedAnswer(
        question=question,
        answer=answer_text,
        model="mock-vlm",
        provider="mock",
    )
    retrieval_results = [
        HybridRetrievalResult(
            document_id="doc_amer_1",
            page_number=1,
            score=1.0,
            chunk_text="Depreciation for AMER was $21,224 thousand and capital expenditures were $17,690 thousand.",
            sources=["dense"],
        )
    ]

    verifier = VerifierEngine()
    report = verifier.verify_answer(gen_answer, retrieval_results)

    # The claim is factually supported in the evidence
    assert report.supported_count >= 1
    assert report.contradicted_count == 0
    # But does not address the question (total assets vs depreciation)
    assert report.is_question_addressed is False

    engine = HallucinationProtectionEngine()
    decision = engine.evaluate_and_protect(report)

    assert decision.is_abstained is True
    assert decision.abstention_reason == AbstentionReason.IRRELEVANT_ANSWER
    assert decision.final_answer == DEFAULT_ABSTENTION_MESSAGE


def test_relevance_gate_matching_metric_accepted():
    """Test B from audit: Supported facts addressing the exact question must be accepted."""
    from auditrag.verification.engine import VerifierEngine

    question = "What was the depreciation for AMER in 2018?"
    answer_text = "Depreciation for AMER was $21,224 thousand."
    gen_answer = GeneratedAnswer(
        question=question,
        answer=answer_text,
        model="mock-vlm",
        provider="mock",
    )
    retrieval_results = [
        HybridRetrievalResult(
            document_id="doc_amer_1",
            page_number=1,
            score=1.0,
            chunk_text="Depreciation for AMER was $21,224 thousand.",
            sources=["dense"],
        )
    ]

    verifier = VerifierEngine()
    report = verifier.verify_answer(gen_answer, retrieval_results)

    assert report.supported_count >= 1
    assert report.contradicted_count == 0
    assert report.is_question_addressed is True

    engine = HallucinationProtectionEngine()
    decision = engine.evaluate_and_protect(report)

    assert decision.is_abstained is False
    assert decision.abstention_reason == AbstentionReason.NONE
    assert decision.final_answer == answer_text


def test_relevance_gate_year_mismatch():
    """Year mismatch between question and answer must abstain with IRRELEVANT_ANSWER."""
    from auditrag.verification.engine import VerifierEngine

    question = "What was the depreciation for AMER in 2018?"
    answer_text = "In 2016 depreciation for AMER was $21,224 thousand."
    gen_answer = GeneratedAnswer(
        question=question,
        answer=answer_text,
        model="mock-vlm",
        provider="mock",
    )
    retrieval_results = [
        HybridRetrievalResult(
            document_id="doc_amer_1",
            page_number=1,
            score=1.0,
            chunk_text="In 2016 depreciation for AMER was $21,224 thousand.",
            sources=["dense"],
        )
    ]

    verifier = VerifierEngine()
    report = verifier.verify_answer(gen_answer, retrieval_results)

    assert report.supported_count >= 1
    assert report.is_question_addressed is False

    engine = HallucinationProtectionEngine()
    decision = engine.evaluate_and_protect(report)

    assert decision.is_abstained is True
    assert decision.abstention_reason == AbstentionReason.IRRELEVANT_ANSWER


def test_relevance_gate_metric_mismatch_goodwill_vs_depreciation():
    """Metric mismatch (goodwill vs depreciation) must abstain with IRRELEVANT_ANSWER."""
    from auditrag.verification.engine import VerifierEngine

    question = "What are the respective goodwill at 2018 and 2019?"
    answer_text = "Depreciation for the years ended June 30, 2019, 2018 and 2017 was $108.1 million."
    gen_answer = GeneratedAnswer(
        question=question,
        answer=answer_text,
        model="mock-vlm",
        provider="mock",
    )
    retrieval_results = [
        HybridRetrievalResult(
            document_id="doc_goodwill_1",
            page_number=1,
            score=1.0,
            chunk_text="Depreciation for the years ended June 30, 2019, 2018 and 2017 was $108.1 million.",
            sources=["dense"],
        )
    ]

    verifier = VerifierEngine()
    report = verifier.verify_answer(gen_answer, retrieval_results)

    assert report.supported_count >= 1
    assert report.is_question_addressed is False

    engine = HallucinationProtectionEngine()
    decision = engine.evaluate_and_protect(report)

    assert decision.is_abstained is True
    assert decision.abstention_reason == AbstentionReason.IRRELEVANT_ANSWER
