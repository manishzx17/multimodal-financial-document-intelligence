"""Evidence-based claim verification for AuditRAG Phase 9."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

from auditrag.config import PROCESSED_DOCUMENTS_DIR
from auditrag.ingestion.models import NormalizedDocument, NormalizedPage
from auditrag.retrieval.hybrid import HybridRetrievalResult
from auditrag.verification.models import Claim, ClaimType, VerificationResult, VerificationStatus
from auditrag.verification.numerical_verifier import verify_arithmetic_calculation


class EvidenceVerifier:
    """Verifies atomic claims against retrieved evidence blocks and document text."""

    def __init__(self):
        self._doc_cache: Dict[str, NormalizedDocument] = {}

    def _load_document(self, doc_id: str) -> Optional[NormalizedDocument]:
        if doc_id in self._doc_cache:
            return self._doc_cache[doc_id]
        path = PROCESSED_DOCUMENTS_DIR / f"{doc_id}.json"
        if not path.exists():
            return None
        doc = NormalizedDocument.from_file(path)
        self._doc_cache[doc_id] = doc
        return doc

    def verify_claim(
        self,
        claim: Claim,
        retrieval_results: List[HybridRetrievalResult],
    ) -> VerificationResult:
        """Verify an atomic claim against evidence context."""
        # 1. Run deterministic arithmetic check if calculation-oriented
        arithmetic_info = verify_arithmetic_calculation(claim)

        # 2. Gather candidate evidence text
        evidence_snippets: List[Tuple[str, int, str]] = []  # (doc_id, page_num, text)

        # Prioritize cited document and page if present
        if claim.cited_doc_id:
            doc = self._load_document(claim.cited_doc_id)
            if doc:
                pages_to_check = (
                    [doc.get_page(claim.cited_page_number)]
                    if claim.cited_page_number
                    else doc.pages
                )
                for p in pages_to_check:
                    if p:
                        if p.text:
                            evidence_snippets.append((doc.document_id, p.page_number, p.text))
                        for b in p.blocks:
                            evidence_snippets.append((doc.document_id, p.page_number, b.text))

        # Also add text from hybrid retrieval results
        for r in retrieval_results:
            if r.chunk_text:
                evidence_snippets.append((r.document_id, r.page_number, r.chunk_text))

        if not evidence_snippets:
            return VerificationResult(
                claim=claim,
                status=VerificationStatus.NOT_SUPPORTED,
                confidence=0.0,
                explanation="No evidence available for document verification.",
            )

        # 3. Numerical Verification
        if claim.claim_type in (ClaimType.NUMERICAL, ClaimType.CALCULATION) and claim.extracted_numbers:
            # Determine critical numbers that must be supported (financial metrics take precedence over year references)
            non_year_numbers = [
                n for n in claim.extracted_numbers
                if not (1900 <= n <= 2099 and n.is_integer())
            ]
            required_numbers = non_year_numbers if non_year_numbers else claim.extracted_numbers

            best_match: List[float] = []
            best_snippet_info: Optional[Tuple[str, int, str]] = None

            for doc_id, page_num, snippet in evidence_snippets:
                matched_nums = []
                for num in claim.extracted_numbers:
                    num_str_no_comma = f"{num:g}"
                    num_str_comma = f"{int(num):,}" if num.is_integer() else f"{num:,.2f}"
                    pattern = rf"(?<![\d])({re.escape(num_str_no_comma)}|{re.escape(num_str_comma)})(?![\d])"
                    if re.search(pattern, snippet):
                        matched_nums.append(num)

                # Prioritize snippets matching more numbers
                if len(matched_nums) > len(best_match):
                    best_match = matched_nums
                    best_snippet_info = (doc_id, page_num, snippet)

            all_required_found = all(req in best_match for req in required_numbers)
            if best_match and best_snippet_info and all_required_found:
                doc_id, page_num, snippet = best_snippet_info
                status = VerificationStatus.SUPPORTED
                explanation = (
                    f"Numerical fact verified: {best_match} found in document disclosure "
                    f"[Doc: {doc_id[:12]}..., Page: {page_num}]."
                )
                if arithmetic_info and arithmetic_info.get("verified"):
                    explanation += f" {arithmetic_info.get('explanation')}"

                return VerificationResult(
                    claim=claim,
                    status=status,
                    confidence=1.0,
                    evidence_snippet=snippet.strip()[:250],
                    matching_doc_id=doc_id,
                    matching_page_number=page_num,
                    arithmetic_check=arithmetic_info,
                    explanation=explanation,
                )

            # If numbers were not found: check if entity is mentioned with contradicting number
            stop_words = {"the", "for", "and", "was", "were", "are", "from", "with", "that", "this", "total", "were"}
            claim_tokens = [
                t.lower()
                for t in re.findall(r"\b[a-zA-Z]{3,}\b", claim.statement)
                if t.lower() not in stop_words
            ]
            for doc_id, page_num, snippet in evidence_snippets:
                snippet_lower = snippet.lower()
                entity_hits = sum(1 for t in claim_tokens if t in snippet_lower)
                if entity_hits >= 2 and len(claim_tokens) >= 2:
                    # Entity mentioned in context, but claimed numbers absent
                    return VerificationResult(
                        claim=claim,
                        status=VerificationStatus.CONTRADICTED,
                        confidence=0.85,
                        evidence_snippet=snippet.strip()[:250],
                        matching_doc_id=doc_id,
                        matching_page_number=page_num,
                        arithmetic_check=arithmetic_info,
                        explanation=f"Entity context found in disclosure but stated numbers {claim.extracted_numbers} were not confirmed.",
                    )

            return VerificationResult(
                claim=claim,
                status=VerificationStatus.NOT_SUPPORTED,
                confidence=0.5,
                arithmetic_check=arithmetic_info,
                explanation=f"Stated numerical figures {claim.extracted_numbers} could not be located in retrieved evidence.",
            )

        # 4. Textual Verification
        # Check token containment in evidence
        words = [w.lower() for w in re.findall(r"\b[a-zA-Z]{3,}\b", claim.statement)]
        for doc_id, page_num, snippet in evidence_snippets:
            snippet_lower = snippet.lower()
            matching_words = [w for w in words if w in snippet_lower]
            ratio = len(matching_words) / max(len(words), 1)

            if ratio >= 0.7:
                return VerificationResult(
                    claim=claim,
                    status=VerificationStatus.SUPPORTED,
                    confidence=ratio,
                    evidence_snippet=snippet.strip()[:250],
                    matching_doc_id=doc_id,
                    matching_page_number=page_num,
                    explanation=f"Textual assertion supported by disclosed context ({int(ratio*100)}% keyword alignment).",
                )

        return VerificationResult(
            claim=claim,
            status=VerificationStatus.NOT_SUPPORTED,
            confidence=0.3,
            explanation="Claim text could not be verified against the retrieved evidence.",
        )
