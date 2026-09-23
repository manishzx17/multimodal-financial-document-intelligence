"""Streamlit UI module for AuditRAG Phase 17 (Interactive Visual Evidence & Citation Viewer)."""

from auditrag.ui.citation_viewer import (
    get_citation_details,
    get_or_create_citation_overlay,
    resolve_image_path,
)
from auditrag.ui.dashboard import (
    load_all_benchmark_reports,
    load_json_artifact,
    render_evaluation_dashboard,
)
from auditrag.ui.numerical_reasoning import (
    NumericalTraceItem,
    extract_numerical_traces,
)

__all__ = [
    "resolve_image_path",
    "get_or_create_citation_overlay",
    "get_citation_details",
    "NumericalTraceItem",
    "extract_numerical_traces",
    "load_json_artifact",
    "load_all_benchmark_reports",
    "render_evaluation_dashboard",
]
