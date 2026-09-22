"""Visual citation overlay renderer using Pillow for AuditRAG Phase 8."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

from PIL import Image, ImageDraw, ImageFont

from auditrag.config import CITATIONS_DIR, PROJECT_ROOT


def get_default_font(size: int = 14) -> ImageFont.ImageFont:
    """Load default font safely across platforms."""
    try:
        return ImageFont.truetype("Arial.ttf", size)
    except IOError:
        try:
            return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
        except IOError:
            return ImageFont.load_default()


def draw_citation_overlay(
    image_path: Union[str, Path],
    citations: List[Any],
    output_path: Optional[Union[str, Path]] = None,
    highlight_color: Tuple[int, int, int, int] = (245, 158, 11, 70),  # Amber with 70/255 alpha
    border_color: Tuple[int, int, int, int] = (217, 119, 6, 255),      # Solid amber border
    border_width: int = 3,
) -> Path:
    """Draw highlighted bounding boxes and citation badges on the page image.

    Args:
        image_path: Path to original page image PNG.
        citations: List of VisualCitation objects containing .bbox and optional .label.
        output_path: Path to save the annotated image.
        highlight_color: RGBA tuple for the box fill.
        border_color: RGBA tuple for the box border.
        border_width: Stroke width for bounding box rectangles.

    Returns:
        Path to the saved overlay image.
    """
    img_p = Path(image_path)
    if not img_p.is_absolute():
        img_p = PROJECT_ROOT / img_p

    if not img_p.exists():
        raise FileNotFoundError(f"Page image not found at: {img_p}")

    base_image = Image.open(img_p).convert("RGBA")
    overlay = Image.new("RGBA", base_image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    font = get_default_font(size=14)

    for idx, citation in enumerate(citations, start=1):
        bbox = getattr(citation, "bbox", None)
        if not bbox or len(bbox) != 4:
            continue

        x0, y0, x1, y1 = bbox
        # Ensure valid coordinate bounds
        x0, x1 = min(x0, x1), max(x0, x1)
        y0, y1 = min(y0, y1), max(y0, y1)

        # 1. Draw semi-transparent highlight rectangle
        draw.rectangle([x0, y0, x1, y1], fill=highlight_color, outline=border_color, width=border_width)

        # 2. Draw citation badge
        label = getattr(citation, "label", None) or f"[{idx}]"
        # Compute badge dimensions
        try:
            bbox_text = font.getbbox(label)
            text_w = bbox_text[2] - bbox_text[0]
            text_h = bbox_text[3] - bbox_text[1]
        except AttributeError:
            text_w, text_h = 24, 14

        badge_x0 = x0
        badge_y0 = max(0, y0 - text_h - 6)
        badge_x1 = badge_x0 + text_w + 8
        badge_y2 = badge_y0 + text_h + 6

        draw.rectangle([badge_x0, badge_y0, badge_x1, badge_y2], fill=border_color)
        draw.text((badge_x0 + 4, badge_y0 + 2), label, fill=(255, 255, 255, 255), font=font)

    # Composite overlay over original base image
    final_image = Image.alpha_composite(base_image, overlay).convert("RGB")

    # Determine destination path
    if output_path is None:
        CITATIONS_DIR.mkdir(parents=True, exist_ok=True)
        filename = f"{img_p.stem}_citations.png"
        out_p = CITATIONS_DIR / filename
    else:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

    final_image.save(out_p, format="PNG")
    return out_p
