"""Data models for AuditRAG Phase 9 Evidence & Numerical Verification."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class ClaimType(str, Enum):
    """Type of extracted atomic claim."""

    TEXTUAL = "textual"
    NUMERICAL = "numerical"
    CALCULATION = "calculation"


class VerificationStatus(str, Enum):
    """Verdict of verification check."""

    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    NOT_SUPPORTED = "NOT_SUPPORTED"


class Claim(BaseModel):
    """An atomic, verifiable proposition extracted from a generated answer."""

    claim_id: str
    statement: str
    claim_type: ClaimType = ClaimType.TEXTUAL
    extracted_numbers: List[float] = Field(default_factory=list)
    cited_doc_id: Optional[str] = None
    cited_page_number: Optional[int] = None
    raw_citation: Optional[str] = None

    model_config = ConfigDict(extra="ignore")


class VerificationResult(BaseModel):
    """Verification assessment for a single atomic claim."""

    claim: Claim
    status: VerificationStatus
    confidence: float = 1.0
    evidence_snippet: Optional[str] = None
    matching_doc_id: Optional[str] = None
    matching_page_number: Optional[int] = None
    arithmetic_check: Optional[Dict[str, Any]] = None
    explanation: str
    citation_label: Optional[str] = None

    model_config = ConfigDict(extra="ignore")


class VerificationReport(BaseModel):
    """Complete verification audit report for a generated answer."""

    question: str
    answer: str
    claims: List[Claim] = Field(default_factory=list)
    results: List[VerificationResult] = Field(default_factory=list)
    overall_status: VerificationStatus
    supported_count: int = 0
    contradicted_count: int = 0
    not_supported_count: int = 0
    factual_consistency_score: float = 0.0
    is_question_addressed: bool = True
    relevance_score: float = 1.0

    model_config = ConfigDict(extra="ignore")
