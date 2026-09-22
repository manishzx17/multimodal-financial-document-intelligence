"""Atomic claim extraction for AuditRAG Phase 9."""

from __future__ import annotations

import re
from typing import List, Tuple

from auditrag.verification.models import Claim, ClaimType


def extract_numbers_from_text(text: str) -> List[float]:
    """Extract numeric values from a string, handling commas, currency, and percentages."""
    matches = re.findall(r"(?<![a-zA-Z])[-+]?\$?\d+(?:,\d+)*(?:\.\d+)?%?", text)
    numbers: List[float] = []
    for m in matches:
        clean = m.replace("$", "").replace("%", "").replace(",", "").strip()
        if not clean or clean in ("-", "+"):
            continue
        try:
            val = float(clean)
            numbers.append(val)
        except ValueError:
            pass
    return numbers


def parse_citation(text: str) -> Tuple[str, Optional[str], Optional[int], Optional[str]]:
    """Extract [Doc: <doc_id>, Page: <page_num>] from text and return cleaned text, doc_id, page_num, raw_citation."""
    citation_match = re.search(r"\[Doc:\s*([a-f0-9]{32}),\s*Page:\s*(\d+)\]", text)
    if citation_match:
        doc_id = citation_match.group(1)
        page_num = int(citation_match.group(2))
        raw_cit = citation_match.group(0)
        clean_text = text.replace(raw_cit, "").strip()
        # Remove trailing period if it was left from citation at end of sentence
        clean_text = clean_text.rstrip(".").strip()
        return clean_text, doc_id, page_num, raw_cit
    return text.strip().rstrip(".").strip(), None, None, None


CALCULATION_KEYWORDS = {
    "increase",
    "decreased",
    "decrease",
    "growth",
    "difference",
    "sum",
    "total of",
    "totaled",
    "totaling",
    "margin",
    "percentage",
    "ratio",
    "change",
    "variance",
}


def classify_claim(statement: str, numbers: List[float]) -> ClaimType:
    """Determine whether a claim is calculation-based, numerical, or textual."""
    lower = statement.lower()
    has_calc_word = any(k in lower for k in CALCULATION_KEYWORDS) or "%" in statement

    if has_calc_word and len(numbers) >= 2:
        return ClaimType.CALCULATION
    elif len(numbers) > 0:
        return ClaimType.NUMERICAL
    return ClaimType.TEXTUAL


def extract_claims(answer_text: str) -> List[Claim]:
    """Deconstruct generated answer into atomic verifiable Claim objects."""
    if not answer_text or not answer_text.strip():
        return []

    # First, find overall document and page citation if present
    global_clean, global_doc, global_page, global_cit = parse_citation(answer_text)

    # Split into candidate statement segments
    # Split on sentence boundaries, colons, or semicolons
    raw_segments = re.split(r"(?<=[.?!;])\s+|\n+", answer_text)

    claims: List[Claim] = []
    claim_idx = 1

    for seg in raw_segments:
        seg = seg.strip()
        if not seg:
            continue

        clean_seg, doc_id, page_num, raw_cit = parse_citation(seg)
        if not clean_seg:
            continue

        # Inherit global citation if segment doesn't have its own
        final_doc_id = doc_id or global_doc
        final_page_num = page_num or global_page
        final_cit = raw_cit or global_cit

        # If a segment contains multiple distinct numerical clauses separated by 'and', split them
        sub_clauses = re.split(r",?\s+and\s+(?=[a-zA-Z\s]*\$\d)", clean_seg)
        if len(sub_clauses) > 1:
            for sub in sub_clauses:
                sub = sub.strip().rstrip(".")
                if not sub:
                    continue
                numbers = extract_numbers_from_text(sub)
                c_type = classify_claim(sub, numbers)
                claims.append(
                    Claim(
                        claim_id=f"claim_{claim_idx}",
                        statement=sub,
                        claim_type=c_type,
                        extracted_numbers=numbers,
                        cited_doc_id=final_doc_id,
                        cited_page_number=final_page_num,
                        raw_citation=final_cit,
                    )
                )
                claim_idx += 1
        else:
            clean_stmt = clean_seg.rstrip(".")
            numbers = extract_numbers_from_text(clean_stmt)
            c_type = classify_claim(clean_stmt, numbers)
            claims.append(
                Claim(
                    claim_id=f"claim_{claim_idx}",
                    statement=clean_stmt,
                    claim_type=c_type,
                    extracted_numbers=numbers,
                    cited_doc_id=final_doc_id,
                    cited_page_number=final_page_num,
                    raw_citation=final_cit,
                )
            )
            claim_idx += 1

    return claims
