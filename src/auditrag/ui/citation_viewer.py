"""Citation viewer and visual overlay utilities for the AuditRAG Streamlit UI.

Provides robust image path resolution, dynamic on-demand overlay rendering,
and formatted citation metadata extraction for interactive inspection.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from auditrag.citations.citation_engine import VisualCitation
from auditrag.citations.visualizer import draw_citation_overlay
from auditrag.config import CITATIONS_DIR, PROJECT_ROOT


def resolve_image_path(img_path: Union[str, Path, None]) -> Optional[Path]:
    """Resolve an image path relative to PROJECT_ROOT or as an absolute path.

    Returns:
        Existing Path if file exists, else None.
    """
    if not img_path:
        return None

    p = Path(img_path)
    if p.is_absolute() and p.exists():
        return p

    # Try relative to PROJECT_ROOT
    candidate = PROJECT_ROOT / p
    if candidate.exists():
        return candidate.resolve()

    # Try as direct path relative to cwd
    if p.exists():
        return p.resolve()

    return None


def get_or_create_citation_overlay(
    citation: VisualCitation,
    all_citations_on_page: Optional[List[VisualCitation]] = None,
    focused_only: bool = False,
    output_dir: Optional[Path] = None,
) -> Tuple[Optional[Path], bool]:
    """Return an existing overlay image path or dynamically generate one on-the-fly.

    Args:
        citation: The target VisualCitation being inspected.
        all_citations_on_page: Optional list of all citations occurring on this page.
        focused_only: If True, highlight only this specific citation with high-contrast accent.
        output_dir: Directory where dynamically generated overlays are stored.

    Returns:
        Tuple of (resolved_path_or_None, was_newly_generated_bool)
    """
    source_p = resolve_image_path(citation.source_image_path)
    if not source_p or not source_p.exists():
        return None, False

    out_dir = output_dir or CITATIONS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Reuse existing overlay if focused_only is False and path exists
    if not focused_only and citation.overlay_image_path:
        existing = resolve_image_path(citation.overlay_image_path)
        if existing and existing.exists():
            return existing, False

    # 2. Determine target file path for generation
    safe_label = re.sub(r"[^\w]", "", citation.label or "cit")
    if focused_only:
        out_filename = f"{citation.document_id}_{citation.page_number}_cit_{safe_label}_focused.png"
        cits_to_draw = [citation]
        # High-contrast highlight for single focused citation
        highlight_color = (239, 68, 68, 85)   # Coral / Red highlight
        border_color = (220, 38, 38, 255)     # Solid red border
    else:
        out_filename = f"{citation.document_id}_{citation.page_number}_overlay.png"
        cits_to_draw = all_citations_on_page or [citation]
        highlight_color = (245, 158, 11, 75)  # Amber highlight
        border_color = (217, 119, 6, 255)     # Solid amber border

    out_path = out_dir / out_filename

    # If the target file already exists on disk, reuse it
    if out_path.exists():
        return out_path.resolve(), False

    # 3. Generate overlay on-the-fly
    try:
        # Validate that citation has valid bounding box
        if not citation.bbox or len(citation.bbox) != 4:
            return None, False

        saved_path = draw_citation_overlay(
            image_path=source_p,
            citations=cits_to_draw,
            output_path=out_path,
            highlight_color=highlight_color,
            border_color=border_color,
            border_width=3,
        )
        if saved_path and saved_path.exists():
            return saved_path.resolve(), True
    except Exception as e:
        # Gracefully handle any drawing failure
        print(f"Warning: Failed to render dynamic citation overlay: {e}")

    return None, False


def get_citation_details(citation: VisualCitation) -> Dict[str, Any]:
    """Extract structured, display-friendly attributes from a VisualCitation."""
    bbox = citation.bbox or [0, 0, 0, 0]
    x0, y0, x1, y1 = bbox
    width = max(0, x1 - x0)
    height = max(0, y1 - y0)

    source_path = resolve_image_path(citation.source_image_path)
    overlay_path = resolve_image_path(citation.overlay_image_path)

    return {
        "label": citation.label or "[?]",
        "document_id": citation.document_id,
        "page_number": citation.page_number,
        "text": citation.text.strip(),
        "bbox": bbox,
        "width": width,
        "height": height,
        "word_bboxes_count": len(citation.word_bboxes),
        "block_uuid": citation.block_uuid,
        "source_image_path": str(source_path) if source_path else None,
        "overlay_image_path": str(overlay_path) if overlay_path else None,
        "source_exists": bool(source_path and source_path.exists()),
        "overlay_exists": bool(overlay_path and overlay_path.exists()),
    }
