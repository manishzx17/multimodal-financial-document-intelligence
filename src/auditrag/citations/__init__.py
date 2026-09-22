"""Visual Bounding-Box Citations module for AuditRAG Phase 8."""

from auditrag.citations.bounding_box import find_text_boxes_on_page, merge_bboxes
from auditrag.citations.citation_engine import (
    AnswerWithCitations,
    VisualCitation,
    VisualCitationEngine,
)
from auditrag.citations.visualizer import draw_citation_overlay

__all__ = [
    "merge_bboxes",
    "find_text_boxes_on_page",
    "draw_citation_overlay",
    "VisualCitation",
    "AnswerWithCitations",
    "VisualCitationEngine",
]
