"""Integrated AuditRAG Pipeline with Hallucination Protection for Phase 10."""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field

from auditrag.citations import AnswerWithCitations, VisualCitationEngine
from auditrag.generation import GeneratedAnswer, VLMGenerator
from auditrag.retrieval import HybridRetrievalResult, HybridRetriever
from auditrag.verification import (
    AbstentionDecision,
    HallucinationProtectionEngine,
    VerificationReport,
    VerifierEngine,
)


class AuditRAGResponse(BaseModel):
    """End-to-end response produced by the AuditRAG pipeline."""

    question: str
    final_answer: str
    is_abstained: bool
    abstention_reason: Optional[str] = None
    decision: AbstentionDecision
    answer_with_citations: AnswerWithCitations
    verification_report: VerificationReport
    retrieval_results: List[HybridRetrievalResult] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")


class AuditRAGPipeline:
    """Unified multimodal financial RAG pipeline with verification and hallucination protection."""

    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        generator: Optional[VLMGenerator] = None,
        citation_engine: Optional[VisualCitationEngine] = None,
        verifier: Optional[VerifierEngine] = None,
        protection_engine: Optional[HallucinationProtectionEngine] = None,
    ):
        self.retriever = retriever or HybridRetriever()
        self.generator = generator or VLMGenerator()
        self.citation_engine = citation_engine or VisualCitationEngine()
        self.verifier = verifier or VerifierEngine()
        self.protection_engine = protection_engine or HallucinationProtectionEngine()

    def run(self, question: str, top_k: int = 5) -> AuditRAGResponse:
        """Execute end-to-end retrieval, VLM generation, visual citations, verification, and protection."""
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        # Step 1: Multimodal Hybrid Retrieval (Dense + BM25 + ColPali with RRF)
        retrieval_results = self.retriever.retrieve(question, top_k=top_k)

        # Step 2: Grounded VLM Answer Generation
        generated_answer = self.generator.generate_answer(question, retrieval_results)

        # Step 3: Extract atomic claims for block-level alignment
        from auditrag.verification.claim_extractor import extract_claims
        claims = extract_claims(generated_answer.answer)

        # Step 4: Visual Bounding-Box Citations & Overlay Generation (Claim-to-Block Aligned)
        answer_with_citations = self.citation_engine.create_citations_for_answer(
            generated_answer, retrieval_results, claims=claims
        )

        # Step 5: Atomic Evidence & Numerical Verification & Relevance Gating
        verification_report = self.verifier.verify_answer(
            generated_answer, retrieval_results, answer_with_citations.citations, question=question
        )

        # Step 6: Hallucination Protection & Abstention Gating
        decision = self.protection_engine.evaluate_and_protect(
            verification_report, answer_with_citations.overlay_images
        )

        return AuditRAGResponse(
            question=question,
            final_answer=decision.final_answer,
            is_abstained=decision.is_abstained,
            abstention_reason=decision.abstention_reason.value if decision.is_abstained else None,
            decision=decision,
            answer_with_citations=answer_with_citations,
            verification_report=verification_report,
            retrieval_results=retrieval_results,
        )
