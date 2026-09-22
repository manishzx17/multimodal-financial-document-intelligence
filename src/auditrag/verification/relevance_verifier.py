"""Deterministic Question-Answer Relevance Gate for AuditRAG Verification.

Verifies that a generated answer and its extracted claims genuinely address
the user's question, preventing factually supported answers to the wrong
financial metric, entity, or time period from being accepted.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple

from auditrag.verification.models import Claim

# Canonical financial concepts and synonyms/aliases
FINANCIAL_METRICS: Dict[str, List[str]] = {
    "total_assets": [
        "total assets",
        "total asset",
        "assets",
        "asset",
    ],
    "depreciation_amortization": [
        "depreciation",
        "amortization",
        "depreciation and amortization",
        "depreciation expense",
        "amortization expense",
    ],
    "capital_expenditures": [
        "capital expenditures",
        "capital expenditure",
        "capex",
        "additions to property",
        "property and equipment additions",
    ],
    "goodwill": [
        "goodwill",
        "negative goodwill",
    ],
    "revenue": [
        "revenue",
        "revenues",
        "sales",
        "net sales",
        "turnover",
        "total revenue",
        "total revenues",
    ],
    "operating_income": [
        "operating income",
        "operating profit",
        "operating earnings",
        "operating loss",
        "income from operations",
    ],
    "net_income": [
        "net income",
        "net earnings",
        "net loss",
        "net profit",
    ],
    "operating_expense": [
        "operating expense",
        "operating expenses",
        "opex",
        "total operating expenses",
        "selling, general and administrative",
    ],
    "cash_equivalents": [
        "cash and cash equivalents",
        "cash equivalents",
        "cash",
    ],
    "liabilities": [
        "liabilities",
        "total liabilities",
        "current liabilities",
        "long-term debt",
        "loans payable",
        "debt",
        "borrowings",
    ],
    "equity": [
        "equity",
        "shareholders' equity",
        "stockholders' equity",
        "total equity",
        "retained earnings",
    ],
    "earnings_per_share": [
        "net income per share",
        "earnings per share",
        "basic net income per share",
        "diluted earnings per share",
        "diluted net income per share",
        "eps",
    ],
    "shares": [
        "number of shares",
        "shares to be increased",
        "common shares",
        "shares outstanding",
        "weighted-average shares",
    ],
    "income_tax": [
        "provision for income taxes",
        "income tax",
        "income taxes",
        "tax rate",
        "effective tax rate",
        "tax provision",
    ],
    "dividends": [
        "dividends",
        "dividend",
        "shareholder distributions",
        "distributions",
        "dividends paid",
    ],
    "arpu": [
        "apru",
        "arpu",
        "average revenue per user",
    ],
    "capital_intensity": [
        "capital intensity",
        "capital intensity ratio",
    ],
    "payables": [
        "trade payables",
        "accounts payable",
        "payables",
    ],
    "license_agreement": [
        "license agreement",
        "license agreements",
        "licensing agreement",
        "licensed",
        "license",
    ],
}

STOP_ACRONYMS: Set[str] = {
    "US", "USA", "SEC", "GAAP", "CEO", "CFO", "VLM", "RAG", "EM", "F1",
    "TAT", "DQA", "APR", "APRU", "ARPU", "USD", "CAD", "EUR", "GBP", "NOTE", "TABLE",
}

STOP_WORDS: Set[str] = {
    "what", "was", "were", "is", "are", "the", "for", "from", "in", "of", "and",
    "a", "an", "does", "did", "do", "to", "table", "which", "how", "show",
    "that", "this", "these", "those", "provide", "provides", "provided",
    "information", "respective", "respectively", "report", "reported", "included",
    "item", "items", "note", "notes", "based", "on", "disclosed", "disclosures",
    "financial", "statement", "statements", "at", "by", "with", "as", "be",
    "been", "under", "per", "or", "had", "has", "have", "would", "could", "should",
}


def verify_question_relevance(
    question: str,
    answer: str,
    claims: Optional[List[Claim]] = None,
) -> Tuple[bool, float, str]:
    """Deterministically assess whether the generated answer addresses the question.

    Returns:
        (is_question_addressed: bool, relevance_score: float, explanation: str)
    """
    if not question or not question.strip():
        return False, 0.0, "Question is empty."

    if not answer or not answer.strip():
        return False, 0.0, "Answer is empty."

    if "The provided evidence does not contain sufficient information" in answer:
        return False, 0.0, "Answer indicates insufficient evidence."

    q_clean = question.strip()
    q_low = q_clean.lower()
    a_clean = answer.strip()
    a_low = a_clean.lower()

    # Combine answer text and claim statements for coverage
    claims_list = claims or []
    claims_text = " ".join(c.statement for c in claims_list)
    combined_ans = f"{a_clean} {claims_text}"
    combined_low = combined_ans.lower()

    # 1. Financial Metric Alignment
    q_metrics: Set[str] = set()
    for cat, aliases in FINANCIAL_METRICS.items():
        for alias in sorted(aliases, key=len, reverse=True):
            if re.search(rf"\b{re.escape(alias)}\b", q_low):
                q_metrics.add(cat)
                break

    a_metrics: Set[str] = set()
    for cat, aliases in FINANCIAL_METRICS.items():
        for alias in sorted(aliases, key=len, reverse=True):
            if re.search(rf"\b{re.escape(alias)}\b", combined_low):
                a_metrics.add(cat)
                break

    if q_metrics:
        # Require that at least one metric concept requested in the question is present in the answer
        metric_matched = bool(q_metrics & a_metrics)
    else:
        metric_matched = True

    # 2. Entity / Company / Segment Alignment
    q_entities: Set[str] = set()
    for acr in re.findall(r"\b[A-Z]{2,}\b", q_clean):
        if acr not in STOP_ACRONYMS:
            q_entities.add(acr)

    for w in re.findall(r"(?<!^)(?<!\.\s)\b[A-Z][a-z]{2,}\b", q_clean):
        w_low = w.lower()
        if w_low not in STOP_WORDS and w_low not in ["table", "note", "december", "january", "june"]:
            if not any(w_low in alias for aliases in FINANCIAL_METRICS.values() for alias in aliases):
                q_entities.add(w)

    if q_entities:
        entity_matched = any(re.search(rf"\b{re.escape(e)}\b", combined_ans, re.IGNORECASE) for e in q_entities)
    else:
        entity_matched = True

    # 3. Year / Period Alignment
    q_years = set(re.findall(r"\b(19\d\d|20\d\d)\b", q_clean))
    a_years = set(re.findall(r"\b(19\d\d|20\d\d)\b", combined_ans))
    for c in claims_list:
        for n in getattr(c, "extracted_numbers", []):
            if 1900 <= n <= 2099 and n.is_integer():
                a_years.add(str(int(n)))

    if q_years:
        if a_years:
            year_matched = bool(q_years & a_years)
        else:
            # Answer does not state a conflicting year
            year_matched = True
    else:
        year_matched = True

    # 4. Content Token Overlap
    q_tokens = [w for w in re.findall(r"\b[a-zA-Z]{3,}\b", q_low) if w not in STOP_WORDS]
    a_tokens = set(w for w in re.findall(r"\b[a-zA-Z]{3,}\b", combined_low) if w not in STOP_WORDS)

    if q_tokens:
        overlap = sum(1 for t in q_tokens if t in a_tokens) / len(q_tokens)
    else:
        overlap = 1.0

    # Decision
    is_addressed = (
        metric_matched
        and entity_matched
        and year_matched
        and (overlap >= 0.15 or len(q_tokens) < 3)
    )

    score = (
        (0.35 if metric_matched else 0.0)
        + (0.25 if entity_matched else 0.0)
        + (0.20 if year_matched else 0.0)
        + min(1.0, overlap) * 0.20
    )

    if not is_addressed:
        score = min(score, 0.45)

    relevance_score = round(score, 4)

    reasons: List[str] = []
    if not metric_matched:
        reasons.append(f"Metric mismatch: question asks for {q_metrics}, answer discussed {a_metrics}")
    if not entity_matched:
        reasons.append(f"Entity mismatch: question mentions {q_entities}, not found in answer")
    if not year_matched:
        reasons.append(f"Year mismatch: question specifies {q_years}, answer specified {a_years}")
    if overlap < 0.15 and len(q_tokens) >= 3:
        reasons.append(f"Low content overlap: {overlap:.2f}")

    if is_addressed:
        explanation = f"Question addressed (metric={metric_matched}, entity={entity_matched}, year={year_matched}, overlap={overlap:.2f})."
    else:
        explanation = f"Question not addressed: {'; '.join(reasons)}."

    return is_addressed, relevance_score, explanation


class RelevanceVerifier:
    """Orchestrator for deterministic question-answer relevance checks."""

    def verify_relevance(
        self,
        question: str,
        answer: str,
        claims: Optional[List[Claim]] = None,
    ) -> Tuple[bool, float, str]:
        """Verify that answer addresses question intent, metric, entity, and period."""
        return verify_question_relevance(question=question, answer=answer, claims=claims)
