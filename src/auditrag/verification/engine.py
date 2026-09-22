"""Verification orchestrator engine for AuditRAG Phase 9."""

from __future__ import annotations

from typing import List, Optional

from auditrag.citations.citation_engine import VisualCitation
from auditrag.generation.vlm import GeneratedAnswer
from auditrag.retrieval.hybrid import HybridRetrievalResult
from auditrag.verification.claim_extractor import extract_claims
from auditrag.verification.evidence_verifier import EvidenceVerifier
from auditrag.verification.models import (
    Claim,
    VerificationReport,
    VerificationResult,
    VerificationStatus,
)
from auditrag.verification.relevance_verifier import RelevanceVerifier


class VerifierEngine:
    """End-to-end factuality verification engine executing post-generation auditing."""

    def __init__(
        self,
        evidence_verifier: Optional[EvidenceVerifier] = None,
        relevance_verifier: Optional[RelevanceVerifier] = None,
    ):
        self.evidence_verifier = evidence_verifier or EvidenceVerifier()
        self.relevance_verifier = relevance_verifier or RelevanceVerifier()

    def verify_answer(
        self,
        answer: GeneratedAnswer,
        retrieval_results: List[HybridRetrievalResult],
        citations: Optional[List[VisualCitation]] = None,
        question: Optional[str] = None,
    ) -> VerificationReport:
        """Extract atomic claims, verify each against evidence, and produce a verification report."""
        claims = extract_claims(answer.answer)
        results: List[VerificationResult] = []

        citations_list = citations or []

        for claim in claims:
            res = self.evidence_verifier.verify_claim(claim, retrieval_results)

            # Link visual citation label if matching doc and page
            for cit in citations_list:
                if (
                    cit.document_id == res.matching_doc_id
                    and cit.page_number == res.matching_page_number
                ):
                    # Check if citation text matches any claimed numbers or snippet
                    if any(str(int(n)) in cit.text for n in claim.extracted_numbers if n.is_integer()) or cit.text in claim.statement:
                        res.citation_label = cit.label
                        break

            results.append(res)

        # Aggregate counts
        supported = sum(1 for r in results if r.status == VerificationStatus.SUPPORTED)
        contradicted = sum(1 for r in results if r.status == VerificationStatus.CONTRADICTED)
        not_supported = sum(1 for r in results if r.status == VerificationStatus.NOT_SUPPORTED)

        total = len(results)
        score = supported / total if total > 0 else 0.0

        if contradicted > 0:
            overall = VerificationStatus.CONTRADICTED
        elif supported > 0 and not_supported == 0:
            overall = VerificationStatus.SUPPORTED
        else:
            overall = VerificationStatus.NOT_SUPPORTED

        # Question-Answer Relevance Gating
        target_question = question or answer.question
        is_addressed, rel_score, _ = self.relevance_verifier.verify_relevance(
            target_question, answer.answer, claims
        )

        return VerificationReport(
            question=target_question,
            answer=answer.answer,
            claims=claims,
            results=results,
            overall_status=overall,
            supported_count=supported,
            contradicted_count=contradicted,
            not_supported_count=not_supported,
            factual_consistency_score=round(score, 4),
            is_question_addressed=is_addressed,
            relevance_score=rel_score,
        )
