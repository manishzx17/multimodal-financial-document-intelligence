"""Verification module for AuditRAG Phase 9 (Atomic Claims & Factuality Verification)."""

from auditrag.verification.abstention import (
    AbstentionDecision,
    AbstentionReason,
    HallucinationProtectionEngine,
)
from auditrag.verification.claim_extractor import (
    classify_claim,
    extract_claims,
    extract_numbers_from_text,
    parse_citation,
)
from auditrag.verification.engine import VerifierEngine
from auditrag.verification.evidence_verifier import EvidenceVerifier
from auditrag.verification.models import (
    Claim,
    ClaimType,
    VerificationReport,
    VerificationResult,
    VerificationStatus,
)
from auditrag.verification.numerical_verifier import verify_arithmetic_calculation
from auditrag.verification.relevance_verifier import RelevanceVerifier, verify_question_relevance

__all__ = [
    "ClaimType",
    "VerificationStatus",
    "Claim",
    "VerificationResult",
    "VerificationReport",
    "AbstentionReason",
    "AbstentionDecision",
    "HallucinationProtectionEngine",
    "extract_claims",
    "extract_numbers_from_text",
    "parse_citation",
    "classify_claim",
    "verify_arithmetic_calculation",
    "EvidenceVerifier",
    "VerifierEngine",
    "RelevanceVerifier",
    "verify_question_relevance",
]
