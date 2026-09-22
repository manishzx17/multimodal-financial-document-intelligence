"""Bounding-box coordinate mapping for AuditRAG Phase 8."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from auditrag.ingestion.models import NormalizedBlock, NormalizedPage


def merge_bboxes(bboxes: List[List[int]]) -> List[int]:
    """Merge a list of [x0, y0, x1, y1] coordinates into an enclosing bounding box."""
    if not bboxes:
        return [0, 0, 0, 0]
    x0 = min(b[0] for b in bboxes)
    y0 = min(b[1] for b in bboxes)
    x1 = max(b[2] for b in bboxes)
    y1 = max(b[3] for b in bboxes)
    return [x0, y0, x1, y1]


def normalize_token(token: str) -> str:
    """Normalize token for case-insensitive matching."""
    return re.sub(r"[^\w\d%]", "", token).lower()


def find_word_boxes_in_block(
    block: NormalizedBlock,
    search_tokens: List[str],
) -> Optional[Dict[str, Any]]:
    """Search for a sequence of tokens in block word_list and return matching coordinates."""
    if not block.words or not block.words.word_list or not block.words.bbox_list:
        return None

    words = block.words.word_list
    bboxes = block.words.bbox_list
    norm_words = [normalize_token(w) for w in words]
    norm_search = [normalize_token(t) for t in search_tokens if normalize_token(t)]

    if not norm_search:
        return None

    # Slide window across words to find matching token subsequence
    sub_len = len(norm_search)
    for i in range(len(norm_words) - sub_len + 1):
        if norm_words[i : i + sub_len] == norm_search:
            matched_bboxes = bboxes[i : i + sub_len]
            matched_words = words[i : i + sub_len]
            return {
                "bbox": merge_bboxes(matched_bboxes),
                "word_bboxes": matched_bboxes,
                "text": " ".join(matched_words),
                "block_uuid": block.uuid,
            }

    return None


def find_text_boxes_on_page(
    page: NormalizedPage,
    query_text: str,
) -> List[Dict[str, Any]]:
    """Find word-level or block-level bounding boxes for query_text on a page.

    Returns:
        List of dicts containing 'bbox', 'word_bboxes', 'text', and 'block_uuid'.
    """
    if not query_text or not query_text.strip():
        return []

    matches: List[Dict[str, Any]] = []
    tokens = [t for t in query_text.strip().split() if t]

    for block in page.blocks:
        # 1. Try word-level subsequence matching
        match = find_word_boxes_in_block(block, tokens)
        if match:
            matches.append(match)
            continue

        # 2. Try single token matching if multi-token search didn't hit
        if len(tokens) > 1:
            # Check individual numbers or key words
            for tok in tokens:
                if len(tok) >= 3 or any(char.isdigit() for char in tok):
                    sub_match = find_word_boxes_in_block(block, [tok])
                    if sub_match and sub_match not in matches:
                        matches.append(sub_match)

        # 3. Fallback: Check if query_text is a substring of block text
        if not matches and query_text.lower() in block.text.lower():
            matches.append({
                "bbox": block.bbox,
                "word_bboxes": block.words.bbox_list if block.words else [block.bbox],
                "text": block.text.strip(),
                "block_uuid": block.uuid,
            })

    return matches
