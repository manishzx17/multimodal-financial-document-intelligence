"""Deterministic Hallucination Protection and Abstention module for AuditRAG Phase 10."""

from __future__ import annotations

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field

from auditrag.config import ABSTENTION_THRESHOLD, DEFAULT_ABSTENTION_MESSAGE
from auditrag.verification.models import ClaimType, VerificationReport, VerificationStatus


class AbstentionReason(str, Enum):
    """Specific cause for answer rejection / abstention."""

    NONE = "NONE"
    CONTRADICTED_CLAIM = "CONTRADICTED_CLAIM"
    UNSUPPORTED_NUMERICAL_FACT = "UNSUPPORTED_NUMERICAL_FACT"
    FAILED_ARITHMETIC_CHECK = "FAILED_ARITHMETIC_CHECK"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    IRRELEVANT_ANSWER = "IRRELEVANT_ANSWER"


class AbstentionDecision(BaseModel):
    """Final gated decision produced by the hallucination protection layer."""

    is_abstained: bool
    final_answer: str
    abstention_reason: AbstentionReason = AbstentionReason.NONE
    confidence_score: float = 1.0
    verification_report: VerificationReport
    overlay_images: List[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")


class HallucinationProtectionEngine:
    """Deterministic safety gating engine evaluating Phase 9 verification results."""

    def __init__(
        self,
        threshold: float = ABSTENTION_THRESHOLD,
        default_message: str = DEFAULT_ABSTENTION_MESSAGE,
    ):
        self.threshold = threshold
        self.default_message = default_message

    def evaluate_and_protect(
        self,
        report: VerificationReport,
        overlay_images: Optional[List[str]] = None,
    ) -> AbstentionDecision:
        """Evaluate verification results and gate the answer against hallucinations."""
        overlays = overlay_images or []

        # 1. Reject if any claim is contradicted by evidence
        if report.contradicted_count > 0:
            return AbstentionDecision(
                is_abstained=True,
                final_answer=self.default_message,
                abstention_reason=AbstentionReason.CONTRADICTED_CLAIM,
                confidence_score=report.factual_consistency_score,
                verification_report=report,
                overlay_images=overlays,
            )

        # 2. Reject if any deterministic arithmetic check explicitly failed
        for res in report.results:
            if res.arithmetic_check and res.arithmetic_check.get("verified") is False:
                return AbstentionDecision(
                    is_abstained=True,
                    final_answer=self.default_message,
                    abstention_reason=AbstentionReason.FAILED_ARITHMETIC_CHECK,
                    confidence_score=report.factual_consistency_score,
                    verification_report=report,
                    overlay_images=overlays,
                )

        # 3. Reject if unsupported numerical facts exist
        has_unsupported_numbers = any(
            res.status == VerificationStatus.NOT_SUPPORTED
            and res.claim.claim_type in (ClaimType.NUMERICAL, ClaimType.CALCULATION)
            for res in report.results
        )
        if has_unsupported_numbers:
            return AbstentionDecision(
                is_abstained=True,
                final_answer=self.default_message,
                abstention_reason=AbstentionReason.UNSUPPORTED_NUMERICAL_FACT,
                confidence_score=report.factual_consistency_score,
                verification_report=report,
                overlay_images=overlays,
            )

        # 4. Reject if there are zero supported claims or empty results
        if report.supported_count == 0:
            return AbstentionDecision(
                is_abstained=True,
                final_answer=self.default_message,
                abstention_reason=AbstentionReason.INSUFFICIENT_EVIDENCE,
                confidence_score=0.0,
                verification_report=report,
                overlay_images=overlays,
            )

        # 5. Reject if factual consistency is below the required threshold
        if report.factual_consistency_score < self.threshold:
            return AbstentionDecision(
                is_abstained=True,
                final_answer=self.default_message,
                abstention_reason=AbstentionReason.LOW_CONFIDENCE,
                confidence_score=report.factual_consistency_score,
                verification_report=report,
                overlay_images=overlays,
            )

        # 6. Reject if the question is not addressed by the answer (Relevance Gate)
        if not report.is_question_addressed:
            return AbstentionDecision(
                is_abstained=True,
                final_answer=self.default_message,
                abstention_reason=AbstentionReason.IRRELEVANT_ANSWER,
                confidence_score=report.relevance_score,
                verification_report=report,
                overlay_images=overlays,
            )

        # 7. ACCEPT: All claims supported and verified, and question is addressed
        return AbstentionDecision(
            is_abstained=False,
            final_answer=report.answer,
            abstention_reason=AbstentionReason.NONE,
            confidence_score=report.factual_consistency_score,
            verification_report=report,
            overlay_images=overlays,
        )
