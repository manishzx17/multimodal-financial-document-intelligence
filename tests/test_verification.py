"""Unit and integration tests for Phase 9 Evidence & Numerical Verification."""

import pytest

from auditrag.citations.citation_engine import VisualCitation
from auditrag.config import PROCESSED_DOCUMENTS_DIR
from auditrag.generation.vlm import GeneratedAnswer
from auditrag.ingestion.models import NormalizedDocument
from auditrag.retrieval.hybrid import HybridRetrievalResult
from auditrag.verification import (
    Claim,
    ClaimType,
    EvidenceVerifier,
    VerificationReport,
    VerificationResult,
    VerificationStatus,
    VerifierEngine,
    extract_claims,
    extract_numbers_from_text,
    parse_citation,
    verify_arithmetic_calculation,
)


@pytest.fixture
def sample_doc() -> NormalizedDocument:
    doc_path = PROCESSED_DOCUMENTS_DIR / "637fab7088ea6c78a5dba55f17e833bd.json"
    if not doc_path.exists():
        pytest.skip(f"Processed document not found: {doc_path}")
    return NormalizedDocument.from_file(doc_path)


def test_extract_numbers_from_text():
    """Verify numeric extraction handling currency, commas, and floats."""
    text = "AMER depreciation was $22,531 in 2019 and $21,224.50 in 2018, up 15.4%."
    nums = extract_numbers_from_text(text)
    assert nums == [22531.0, 2019.0, 21224.5, 2018.0, 15.4]


def test_parse_citation():
    """Verify parsing document ID and page number from citation strings."""
    doc_id = "637fab7088ea6c78a5dba55f17e833bd"
    text = f"Depreciation was $21,224 [Doc: {doc_id}, Page: 1]."
    clean, d, p, raw = parse_citation(text)
    assert d == doc_id
    assert p == 1
    assert raw == f"[Doc: {doc_id}, Page: 1]"
    assert clean == "Depreciation was $21,224"


def test_extract_claims():
    """Verify atomic claim extraction and classification."""
    doc_id = "637fab7088ea6c78a5dba55f17e833bd"
    answer = (
        f"In 2018 depreciation for AMER was $21,224 thousand and capital expenditures were $17,690 thousand "
        f"[Doc: {doc_id}, Page: 1]."
    )
    claims = extract_claims(answer)
    assert len(claims) >= 2

    # Check first claim (depreciation)
    assert "21,224" in claims[0].statement
    assert claims[0].claim_type == ClaimType.NUMERICAL
    assert 21224.0 in claims[0].extracted_numbers
    assert claims[0].cited_doc_id == doc_id
    assert claims[0].cited_page_number == 1

    # Check second claim (capital expenditures)
    assert "17,690" in claims[1].statement
    assert claims[1].claim_type == ClaimType.NUMERICAL
    assert 17690.0 in claims[1].extracted_numbers


def test_numerical_arithmetic_verifier():
    """Verify deterministic arithmetic checks for variance, sums, and percentages."""
    # Difference / Variance check
    claim_diff = Claim(
        claim_id="c1",
        statement="Operating income increased by 19.2 from 38.6 to 57.8",
        claim_type=ClaimType.CALCULATION,
        extracted_numbers=[19.2, 38.6, 57.8],
    )
    res_diff = verify_arithmetic_calculation(claim_diff)
    assert res_diff is not None
    assert res_diff["verified"] is True
    assert "57.8 - 38.6 = 19.2" in res_diff["expression"]

    # Sum / Total check
    claim_sum = Claim(
        claim_id="c2",
        statement="Total depreciation was 48,095 totaling from 21,224, 15,954, 6,054, and 4,863",
        claim_type=ClaimType.CALCULATION,
        extracted_numbers=[48095.0, 21224.0, 15954.0, 6054.0, 4863.0],
    )
    res_sum = verify_arithmetic_calculation(claim_sum)
    assert res_sum is not None
    assert res_sum["verified"] is True

    # No calculation present
    claim_none = Claim(
        claim_id="c3",
        statement="Depreciation was 21,224",
        claim_type=ClaimType.NUMERICAL,
        extracted_numbers=[21224.0],
    )
    assert verify_arithmetic_calculation(claim_none) is None


def test_evidence_verifier_supported(sample_doc: NormalizedDocument):
    """Verify that an atomic claim matching the document is marked SUPPORTED."""
    verifier = EvidenceVerifier()
    doc_id = sample_doc.document_id

    claim = Claim(
        claim_id="c_supp",
        statement="Depreciation for AMER in 2018 was $21,224",
        claim_type=ClaimType.NUMERICAL,
        extracted_numbers=[2018.0, 21224.0],
        cited_doc_id=doc_id,
        cited_page_number=1,
    )

    res = verifier.verify_claim(claim, [])
    assert res.status == VerificationStatus.SUPPORTED
    assert res.confidence == 1.0
    assert "21224" in res.explanation
    assert res.matching_doc_id == doc_id
    assert res.matching_page_number == 1


def test_evidence_verifier_contradicted(sample_doc: NormalizedDocument):
    """Verify that a claim with a contradicted number is marked CONTRADICTED."""
    verifier = EvidenceVerifier()
    doc_id = sample_doc.document_id

    # Claim that AMER depreciation was 999,999 (entity exists on page, but number contradicts)
    claim = Claim(
        claim_id="c_contra",
        statement="Depreciation for AMER was 999,999",
        claim_type=ClaimType.NUMERICAL,
        extracted_numbers=[999999.0],
        cited_doc_id=doc_id,
        cited_page_number=1,
    )

    res = verifier.verify_claim(claim, [])
    assert res.status == VerificationStatus.CONTRADICTED
    assert "not confirmed" in res.explanation


def test_evidence_verifier_not_supported():
    """Verify that a claim with unknown entities and numbers is NOT_SUPPORTED."""
    verifier = EvidenceVerifier()
    claim = Claim(
        claim_id="c_not",
        statement="Solar energy revenue reached 77,888 in Antarctica",
        claim_type=ClaimType.NUMERICAL,
        extracted_numbers=[77888.0],
    )
    res = verifier.verify_claim(claim, [])
    assert res.status == VerificationStatus.NOT_SUPPORTED


def test_verifier_engine_full_report(sample_doc: NormalizedDocument):
    """Verify complete verification report generation and visual citation linking."""
    engine = VerifierEngine()
    doc_id = sample_doc.document_id

    answer = GeneratedAnswer(
        question="What was the depreciation for AMER in 2018?",
        answer=f"In 2018 depreciation for AMER was $21,224 [Doc: {doc_id}, Page: 1].",
        sources=[{"document_id": doc_id, "page_number": 1}],
        model="mock-vlm-v1",
        provider="MockVLMProvider",
        evidence_count=1,
    )

    retrieval_results = [
        HybridRetrievalResult(
            document_id=doc_id,
            page_number=1,
            score=0.035,
            sources=["dense", "bm25"],
            chunk_id="chunk_1",
            chunk_text="Depreciation: AMER $ 21,224",
            source_metadata={"bboxes": [[100, 200, 300, 250]]},
            image_path=sample_doc.pages[0].page_image,
        )
    ]

    citations = [
        VisualCitation(
            document_id=doc_id,
            page_number=1,
            text="21,224",
            bbox=[100, 200, 300, 250],
            label="[1]",
            source_image_path=sample_doc.pages[0].page_image or "",
        )
    ]

    report = engine.verify_answer(answer, retrieval_results, citations)

    assert isinstance(report, VerificationReport)
    assert report.question == answer.question
    assert report.answer == answer.answer
    assert len(report.claims) > 0
    assert len(report.results) == len(report.claims)
    assert report.overall_status == VerificationStatus.SUPPORTED
    assert report.supported_count >= 1
    assert report.contradicted_count == 0
    assert report.factual_consistency_score == 1.0

    # Verify visual citation linking
    assert any(r.citation_label == "[1]" for r in report.results)
