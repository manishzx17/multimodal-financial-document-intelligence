"""Evaluation metrics for AuditRAG Phase 11 (Evaluation & Benchmarking).

This module defines deterministic metrics for:
1. Retrieval: Document Hit@K, Page Hit@K, Document MRR, Page MRR.
2. Answer Grounding & Quality: Exact Match (EM), Token F1, Numerical Match.
3. Factuality & Hallucination Protection: Claim Support Rate, Contradiction Rate,
   Unsupported Rate, Abstention Rate, and Reason Breakdown.
"""

from __future__ import annotations

import re
import string
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union
from pydantic import BaseModel, Field

from auditrag.verification.models import VerificationReport, VerificationStatus


# ---------------------------------------------------------------------------
# 1. Retrieval Metrics
# ---------------------------------------------------------------------------

class RetrievalItem(BaseModel):
    """Normalized representation of a retrieved candidate for evaluation."""
    document_id: str
    page_number: Optional[int] = None
    score: Optional[float] = None


def compute_hit_at_k(
    retrieved: Sequence[Any],
    gold_doc_id: str,
    gold_pages: Optional[Set[int]] = None,
    k: int = 5,
) -> Tuple[float, float]:
    """Calculate (doc_hit@k, page_hit@k).

    Args:
        retrieved: List of objects having `document_id` and optional `page_number`.
        gold_doc_id: Ground-truth target document UID.
        gold_pages: Ground-truth pages containing evidence.
        k: Cutoff rank.

    Returns:
        (doc_hit, page_hit) where each is 1.0 if found in top-k, else 0.0.
    """
    top_k = retrieved[:k]
    doc_hit = 0.0
    page_hit = 0.0

    for item in top_k:
        doc_id = getattr(item, "document_id", None) or (item.get("document_id") if isinstance(item, dict) else None)
        page_num = getattr(item, "page_number", None) or (item.get("page_number") if isinstance(item, dict) else None)

        if doc_id == gold_doc_id:
            doc_hit = 1.0
            if gold_pages and page_num in gold_pages:
                page_hit = 1.0

        if doc_hit == 1.0 and (not gold_pages or page_hit == 1.0):
            break

    # If no gold pages were annotated, page_hit defaults to doc_hit
    if not gold_pages:
        page_hit = doc_hit

    return doc_hit, page_hit


def compute_mrr(
    retrieved: Sequence[Any],
    gold_doc_id: str,
    gold_pages: Optional[Set[int]] = None,
) -> Tuple[float, float]:
    """Calculate (doc_mrr, page_mrr) over retrieved ranking.

    Returns:
        Reciprocal ranks (1 / rank) for the first matching document and page.
    """
    doc_mrr = 0.0
    page_mrr = 0.0

    for rank, item in enumerate(retrieved, start=1):
        doc_id = getattr(item, "document_id", None) or (item.get("document_id") if isinstance(item, dict) else None)
        page_num = getattr(item, "page_number", None) or (item.get("page_number") if isinstance(item, dict) else None)

        if doc_mrr == 0.0 and doc_id == gold_doc_id:
            doc_mrr = 1.0 / rank

        if page_mrr == 0.0 and doc_id == gold_doc_id and gold_pages and page_num in gold_pages:
            page_mrr = 1.0 / rank

        if doc_mrr > 0.0 and (not gold_pages or page_mrr > 0.0):
            break

    if not gold_pages:
        page_mrr = doc_mrr

    return doc_mrr, page_mrr


# ---------------------------------------------------------------------------
# 2. Answer Grounding & Quality Metrics
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Normalize text by lowercasing, removing punctuation, articles, currency, and extra whitespace."""
    if not text:
        return ""
    text = str(text).lower()
    # Remove currency signs and percent
    text = text.replace("$", " ").replace("%", " ").replace(",", "")
    # Remove punctuation
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    # Remove common stop articles
    words = [w for w in text.split() if w not in {"a", "an", "the"}]
    return " ".join(words)


def extract_numbers(text: str) -> List[float]:
    """Extract numeric values from text."""
    matches = re.findall(r"(?<![a-zA-Z])[-+]?\$?\d+(?:,\d+)*(?:\.\d+)?%?", str(text))
    numbers = []
    for m in matches:
        clean = m.replace("$", "").replace("%", "").replace(",", "").strip()
        if not clean or clean in ("-", "+"):
            continue
        try:
            numbers.append(float(clean))
        except ValueError:
            pass
    return numbers


def gold_to_strings(gold_answer: Any) -> List[str]:
    """Convert gold answer field (which may be str, int, float, or list) into candidate strings."""
    if gold_answer is None:
        return []
    if isinstance(gold_answer, list):
        out = []
        for g in gold_answer:
            out.extend(gold_to_strings(g))
        return out
    return [str(gold_answer).strip()]


def exact_match_score(prediction: str, gold_answer: Any) -> float:
    """Determine Exact Match (EM) between prediction and any gold answer candidate."""
    norm_pred = normalize_text(prediction)
    golds = gold_to_strings(gold_answer)
    if not golds:
        return 0.0
    for g in golds:
        norm_gold = normalize_text(g)
        if norm_gold and (norm_gold == norm_pred or norm_gold in norm_pred):
            return 1.0
    return 0.0


def token_f1_score(prediction: str, gold_answer: Any) -> float:
    """Compute token-level F1 overlap between prediction and gold answer candidates."""
    norm_pred = normalize_text(prediction)
    pred_tokens = norm_pred.split()
    if not pred_tokens:
        return 0.0

    golds = gold_to_strings(gold_answer)
    if not golds:
        return 0.0

    best_f1 = 0.0
    for g in golds:
        gold_tokens = normalize_text(g).split()
        if not gold_tokens:
            continue

        common = set(pred_tokens) & set(gold_tokens)
        num_same = sum(min(pred_tokens.count(w), gold_tokens.count(w)) for w in common)
        if num_same == 0:
            continue

        precision = num_same / len(pred_tokens)
        recall = num_same / len(gold_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        if f1 > best_f1:
            best_f1 = f1

    return round(best_f1, 4)


def numeric_match_score(prediction: str, gold_answer: Any, tolerance: float = 1e-3) -> float:
    """Check if the numeric values in gold_answer appear in prediction within tolerance."""
    pred_numbers = extract_numbers(prediction)
    if not pred_numbers:
        return 0.0

    gold_numbers = []
    if isinstance(gold_answer, (int, float)):
        gold_numbers.append(float(gold_answer))
    elif isinstance(gold_answer, list):
        for g in gold_answer:
            gold_numbers.extend(extract_numbers(g))
    else:
        gold_numbers.extend(extract_numbers(str(gold_answer)))

    if not gold_numbers:
        return 0.0

    # Check if all gold numbers are matched in pred_numbers
    matched = 0
    for gn in gold_numbers:
        if any(abs(gn - pn) <= max(tolerance, abs(gn) * 0.01) for pn in pred_numbers):
            matched += 1

    return 1.0 if matched == len(gold_numbers) else (matched / len(gold_numbers))


# ---------------------------------------------------------------------------
# 3. Verification & Hallucination Protection Metrics
# ---------------------------------------------------------------------------

class VerificationMetricsSummary(BaseModel):
    """Summary of atomic claim verification results."""
    total_claims: int = 0
    supported_claims: int = 0
    contradicted_claims: int = 0
    not_supported_claims: int = 0
    claim_support_rate: float = 0.0
    claim_contradiction_rate: float = 0.0
    claim_unsupported_rate: float = 0.0
    average_factual_consistency: float = 0.0


def compute_verification_metrics(reports: Sequence[VerificationReport]) -> VerificationMetricsSummary:
    """Aggregate claim-level verification statistics across multiple reports."""
    total_claims = sum(
        len(r.claims) if r.claims else (r.supported_count + r.contradicted_count + r.not_supported_count)
        for r in reports
    )
    supported = sum(r.supported_count for r in reports)
    contradicted = sum(r.contradicted_count for r in reports)
    not_supported = sum(r.not_supported_count for r in reports)

    support_rate = supported / total_claims if total_claims > 0 else 0.0
    contra_rate = contradicted / total_claims if total_claims > 0 else 0.0
    unsupp_rate = not_supported / total_claims if total_claims > 0 else 0.0

    avg_consistency = (
        sum(r.factual_consistency_score for r in reports) / len(reports)
        if reports
        else 0.0
    )

    return VerificationMetricsSummary(
        total_claims=total_claims,
        supported_claims=supported,
        contradicted_claims=contradicted,
        not_supported_claims=not_supported,
        claim_support_rate=round(support_rate, 4),
        claim_contradiction_rate=round(contra_rate, 4),
        claim_unsupported_rate=round(unsupp_rate, 4),
        average_factual_consistency=round(avg_consistency, 4),
    )


class AbstentionMetricsSummary(BaseModel):
    """Summary of hallucination protection and abstention gating."""
    total_evaluated: int = 0
    accepted_count: int = 0
    abstained_count: int = 0
    abstention_rate: float = 0.0
    reasons_breakdown: Dict[str, int] = Field(default_factory=dict)


def compute_abstention_metrics(responses: Sequence[Any]) -> AbstentionMetricsSummary:
    """Compute abstention rates and breakdown across pipeline responses."""
    total = len(responses)
    if total == 0:
        return AbstentionMetricsSummary()

    abstained = sum(1 for r in responses if getattr(r, "is_abstained", False))
    accepted = total - abstained
    reasons: Dict[str, int] = {}

    for r in responses:
        if getattr(r, "is_abstained", False):
            reason = getattr(r, "abstention_reason", "UNKNOWN") or "UNKNOWN"
            reasons[reason] = reasons.get(reason, 0) + 1

    return AbstentionMetricsSummary(
        total_evaluated=total,
        accepted_count=accepted,
        abstained_count=abstained,
        abstention_rate=round(abstained / total, 4),
        reasons_breakdown=reasons,
    )
