"""Explainable Numerical Reasoning & Verification Trace for AuditRAG Streamlit UI (Addition 2).

Extracts and structures deterministic calculation traces, source evidence values,
mathematical formulas, and verification verdicts from VerificationReports.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from auditrag.verification.models import ClaimType, VerificationReport, VerificationResult, VerificationStatus


class NumericalTraceItem(BaseModel):
    """Structured deterministic calculation and numerical verification trace item."""

    claim_id: str
    claim_statement: str
    claim_type: str  # "calculation" or "numerical"
    source_values: List[str] = Field(default_factory=list)
    formula_name: Optional[str] = None
    formula_template: Optional[str] = None
    formula_expression: Optional[str] = None
    calculated_value: Optional[str] = None
    reported_value: Optional[str] = None
    status: str  # "VERIFIED", "FAILED_ARITHMETIC_CHECK", "SUPPORTED", "UNSUPPORTED_NUMERICAL_FACT", "CONTRADICTED"
    is_verified: bool = False
    explanation: str = ""
    citation_label: Optional[str] = None
    document_id: Optional[str] = None
    page_number: Optional[int] = None
    evidence_snippet: Optional[str] = None


def extract_numerical_traces(report: Optional[VerificationReport]) -> List[NumericalTraceItem]:
    """Extract all numerical and calculation verification items from a VerificationReport.

    Returns:
        List of NumericalTraceItem objects. Empty if only textual claims exist.
    """
    if not report or not report.results:
        return []

    traces: List[NumericalTraceItem] = []

    for res in report.results:
        claim = res.claim
        is_calc = claim.claim_type == ClaimType.CALCULATION or res.arithmetic_check is not None
        is_num = claim.claim_type == ClaimType.NUMERICAL

        if not (is_calc or is_num):
            continue

        arith = res.arithmetic_check

        # -------------------------------------------------------------------
        # 1. Calculation Claims (with deterministic arithmetic check)
        # -------------------------------------------------------------------
        if is_calc and arith:
            verified = arith.get("verified", False)
            op = arith.get("operation", "arithmetic_check")
            expr = arith.get("expression", "")
            computed = arith.get("computed_value")
            claimed = arith.get("claimed_value")
            expl = arith.get("explanation", res.explanation)

            # Determine human-readable formula name and template
            if op == "percentage_growth_check":
                formula_name = "Percentage Change / Growth"
                formula_template = "((New_Value - Old_Value) / |Old_Value|) × 100"
            elif op == "variance_check":
                formula_name = "Variance / Difference"
                formula_template = "End_Value - Start_Value = Variance"
            elif op == "sum_check":
                formula_name = "Summation / Aggregate"
                formula_template = "Sum(Component_Values) = Total"
            else:
                formula_name = "Deterministic Arithmetic Check"
                formula_template = expr

            # Determine source values vs target reported value
            claimed_str = f"{claimed:g}" if isinstance(claimed, (int, float)) else str(claimed)
            computed_str = f"{computed:g}" if isinstance(computed, (int, float)) else str(computed)

            source_nums = []
            for n in claim.extracted_numbers:
                n_str = f"{int(n):,}" if n.is_integer() else f"{n:g}"
                if claimed is None or not (abs(n - float(claimed)) < 1e-4):
                    source_nums.append(n_str)

            if not source_nums and claim.extracted_numbers:
                source_nums = [f"{n:g}" for n in claim.extracted_numbers]

            status_str = "VERIFIED" if verified else "FAILED_ARITHMETIC_CHECK"

            traces.append(
                NumericalTraceItem(
                    claim_id=claim.claim_id,
                    claim_statement=claim.statement,
                    claim_type="calculation",
                    source_values=source_nums,
                    formula_name=formula_name,
                    formula_template=formula_template,
                    formula_expression=expr,
                    calculated_value=computed_str,
                    reported_value=claimed_str,
                    status=status_str,
                    is_verified=verified,
                    explanation=expl,
                    citation_label=res.citation_label,
                    document_id=res.matching_doc_id or claim.cited_doc_id,
                    page_number=res.matching_page_number or claim.cited_page_number,
                    evidence_snippet=res.evidence_snippet,
                )
            )

        # -------------------------------------------------------------------
        # 2. Calculation Claim where arithmetic check was attempted but failed
        # -------------------------------------------------------------------
        elif is_calc and not arith:
            nums = [f"{int(n):,}" if n.is_integer() else f"{n:g}" for n in claim.extracted_numbers]
            traces.append(
                NumericalTraceItem(
                    claim_id=claim.claim_id,
                    claim_statement=claim.statement,
                    claim_type="calculation",
                    source_values=nums[:-1] if len(nums) > 1 else nums,
                    formula_name="Calculation Verification",
                    formula_template="Arithmetic consistency evaluation",
                    formula_expression="Operands did not verify via deterministic arithmetic",
                    calculated_value="Mismatch",
                    reported_value=nums[-1] if nums else "N/A",
                    status="FAILED_ARITHMETIC_CHECK",
                    is_verified=False,
                    explanation="Deterministic arithmetic check could not verify the calculation from extracted numbers.",
                    citation_label=res.citation_label,
                    document_id=res.matching_doc_id or claim.cited_doc_id,
                    page_number=res.matching_page_number or claim.cited_page_number,
                    evidence_snippet=res.evidence_snippet,
                )
            )

        # -------------------------------------------------------------------
        # 3. Direct Numerical Fact Claims (Grounding in financial disclosures)
        # -------------------------------------------------------------------
        elif is_num:
            nums = [f"{int(n):,}" if n.is_integer() else f"{n:g}" for n in claim.extracted_numbers]
            is_sup = res.status == VerificationStatus.SUPPORTED
            is_contra = res.status == VerificationStatus.CONTRADICTED

            if is_sup:
                status_str = "SUPPORTED"
            elif is_contra:
                status_str = "CONTRADICTED"
            else:
                status_str = "UNSUPPORTED_NUMERICAL_FACT"

            traces.append(
                NumericalTraceItem(
                    claim_id=claim.claim_id,
                    claim_statement=claim.statement,
                    claim_type="numerical",
                    source_values=nums,
                    formula_name="Direct Financial Disclosure Grounding",
                    formula_template="Disclosed Amount = Stated Fact",
                    formula_expression=f"Reported value(s): {', '.join(nums)} matched against evidence disclosure",
                    calculated_value=nums[0] if nums else "N/A",
                    reported_value=nums[0] if nums else "N/A",
                    status=status_str,
                    is_verified=is_sup,
                    explanation=res.explanation,
                    citation_label=res.citation_label,
                    document_id=res.matching_doc_id or claim.cited_doc_id,
                    page_number=res.matching_page_number or claim.cited_page_number,
                    evidence_snippet=res.evidence_snippet,
                )
            )

    return traces
