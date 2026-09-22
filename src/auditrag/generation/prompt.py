"""Grounded prompt orchestration for multimodal VLM answer generation in AuditRAG."""

from __future__ import annotations

from pathlib import Path
import re
from typing import List, Set, Tuple

from auditrag.config import PROJECT_ROOT
from auditrag.retrieval.hybrid import HybridRetrievalResult

SYSTEM_PROMPT = """You are AuditRAG, an auditor-grade multimodal financial question answering assistant.
Your task is to answer the user's question accurately, concisely, and strictly using ONLY the provided evidence (text snippets and page images).

STRICT RULES:
1. Directly answer the question asked. Do not substitute the requested financial metric with unrelated figures (e.g. if asked about 'total assets', report total assets, not depreciation or revenues).
2. Rely ONLY on explicit facts, tables, and disclosures in the provided evidence. Do NOT extrapolate, assume, or bring in external financial or domain knowledge not present in the evidence.
3. If the provided evidence does not contain the answer to the user's question, reply ONLY: "The provided evidence does not contain sufficient information to answer this question."
4. Every fact, statement, or numerical figure you assert MUST be followed by a concise source citation referencing the document ID and page number in the exact format: [Doc: <document_id>, Page: <page_number>].
5. Maintain numerical fidelity, preserving currency and units (e.g., in millions, thousands, percentage) exactly as disclosed in the source text.
"""


def _score_chunk_relevance(question: str, chunk_text: str) -> float:
    """Score candidate chunk relevance based on token overlap with question."""
    if not chunk_text:
        return 0.0
    stop_words = {"what", "was", "the", "in", "from", "of", "to", "for", "and", "did", "is", "were", "a", "an"}
    q_tokens = [w.lower() for w in re.findall(r"\b[a-zA-Z0-9$%,.-]{2,}\b", question) if w.lower() not in stop_words]
    if not q_tokens:
        return 1.0

    text_lower = chunk_text.lower()
    matches = sum(1 for t in q_tokens if t in text_lower)
    return matches / len(q_tokens)


def build_grounded_prompt(
    question: str,
    retrieval_results: List[HybridRetrievalResult],
) -> Tuple[str, List[str]]:
    """Construct a grounded prompt with deduplicated, prioritized evidence and page images.

    Returns:
        prompt_text: Full formatted text prompt including system rules, formatted evidence,
                     and the user question.
        image_paths: Deduplicated list of existing absolute or project-relative image paths.
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    # Deduplicate and prioritize retrieval candidates
    seen_texts: Set[str] = set()
    candidate_items = []

    for r in retrieval_results:
        text = (r.chunk_text or "").strip()
        # Normalization key for deduplication
        dedup_key = (r.document_id, r.page_number, text[:150])
        if dedup_key in seen_texts:
            continue
        seen_texts.add(dedup_key)

        relevance = _score_chunk_relevance(question, text) if text else 0.0
        candidate_items.append((relevance, r))

    # Sort candidates by relevance score descending while preserving original RRF rank as tie-breaker
    candidate_items.sort(key=lambda item: item[0], reverse=True)

    evidence_lines: List[str] = []
    collected_images: List[str] = []
    seen_images: set = set()

    for idx, (_, res) in enumerate(candidate_items, start=1):
        sources_str = ", ".join(res.sources) if res.sources else "retrieval"
        header = f"[Evidence {idx}] (Sources: {sources_str}) [Doc: {res.document_id}, Page: {res.page_number}]"
        evidence_lines.append(header)

        if res.chunk_text:
            cleaned_text = res.chunk_text.strip()
            evidence_lines.append(f"Text Content:\n{cleaned_text}")

        if res.image_path:
            img_path = Path(res.image_path)
            if not img_path.is_absolute():
                full_img_path = PROJECT_ROOT / img_path
            else:
                full_img_path = img_path

            if full_img_path.exists():
                evidence_lines.append(f"Associated Page Image: {res.image_path}")
                str_path = str(full_img_path.resolve())
                if str_path not in seen_images:
                    seen_images.add(str_path)
                    collected_images.append(str(res.image_path))

        evidence_lines.append("")  # Empty separator line

    evidence_block = "\n".join(evidence_lines).strip()
    if not evidence_block:
        evidence_block = "No retrieval evidence provided."

    prompt_text = f"""{SYSTEM_PROMPT}

=== RETRIEVED EVIDENCE ===
{evidence_block}

=== USER QUESTION ===
{question.strip()}

=== AUDIT-GRADE ANSWER ===
Provide your grounded response below, strictly citing [Doc: <document_id>, Page: <page_number>] for every asserted fact or number:"""

    return prompt_text, collected_images
