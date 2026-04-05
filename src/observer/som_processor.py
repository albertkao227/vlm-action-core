"""
som_processor.py — Set-of-Mark (SoM) annotation for screenshots.

Overlays numbered bounding boxes on visually distinct regions of a
screenshot to give the VLM "handles" it can reference by ID.

Strategy:
  1. Convert to greyscale → adaptive threshold → find contours
  2. Filter contours by area (remove noise / whole-screen boxes)
  3. Merge overlapping boxes (non-max suppression lite)
  4. Draw numbered rectangles + build element_map {id: (cx, cy, w, h)}
"""

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

# ── Tuning knobs ────────────────────────────────────────────────────
MIN_AREA = 600          # ignore regions smaller than this (px²)
MAX_AREA_RATIO = 0.60   # ignore regions larger than 60 % of screen
MERGE_IOU_THRESH = 0.40 # merge boxes whose IoU exceeds this
BORDER_COLOR = (255, 50, 50)   # red
LABEL_BG     = (255, 50, 50)
LABEL_FG     = (255, 255, 255)
BORDER_WIDTH = 2
FONT_SIZE    = 14


# ── Helpers ─────────────────────────────────────────────────────────

def _iou(a: tuple, b: tuple) -> float:
    """Intersection-over-union of two (x, y, w, h) boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _nms_merge(boxes: list[tuple], iou_thresh: float) -> list[tuple]:
    """Simple greedy non-max-suppression by area (largest kept)."""
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    kept: list[tuple] = []
    for box in boxes:
        if all(_iou(box, k) < iou_thresh for k in kept):
            kept.append(box)
    return kept


# ── Public API ──────────────────────────────────────────────────────

def detect_elements(img: Image.Image) -> list[tuple[int, int, int, int]]:
    """
    Detect interactive-looking regions in a screenshot.

    Returns:
        List of (x, y, w, h) bounding boxes.
    """
    arr = np.array(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    screen_area = gray.shape[0] * gray.shape[1]
    max_area = int(screen_area * MAX_AREA_RATIO)

    # Adaptive threshold highlights UI edges
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=15, C=8,
    )

    # Morphological close to merge nearby edges into solid regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if MIN_AREA <= area <= max_area:
            boxes.append((x, y, w, h))

    # Deduplicate overlapping detections
    boxes = _nms_merge(boxes, MERGE_IOU_THRESH)
    # Sort top-left to bottom-right for stable IDs
    boxes.sort(key=lambda b: (b[1] // 40, b[0]))
    return boxes


def annotate_screenshot(
    img: Image.Image,
    save: bool = False,
    output_dir: Path | str | None = None,
) -> tuple[Image.Image, dict[int, dict[str, Any]]]:
    """
    Overlay Set-of-Mark bounding boxes onto a screenshot.

    Args:
        img: Raw screenshot (PIL RGB).
        save: Persist annotated image + JSON label to disk.
        output_dir: Where to save (defaults to data/processed/).

    Returns:
        (annotated_image, element_map)
        element_map: {element_id: {"cx": int, "cy": int, "x": int,
                                    "y": int, "w": int, "h": int}}
    """
    boxes = detect_elements(img)
    annotated = img.copy()
    draw = ImageDraw.Draw(annotated)

    # Try to use a monospace font; fall back to default
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", FONT_SIZE)
    except (OSError, IOError):
        font = ImageFont.load_default()

    element_map: dict[int, dict[str, Any]] = {}

    for idx, (x, y, w, h) in enumerate(boxes):
        eid = idx + 1  # 1-based IDs for the VLM
        cx, cy = x + w // 2, y + h // 2
        element_map[eid] = {"cx": cx, "cy": cy, "x": x, "y": y, "w": w, "h": h}

        # Draw bounding box
        draw.rectangle([x, y, x + w, y + h], outline=BORDER_COLOR, width=BORDER_WIDTH)

        # Draw label background + text
        label = str(eid)
        bbox = font.getbbox(label)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        lx, ly = x, max(y - th - 4, 0)
        draw.rectangle([lx, ly, lx + tw + 6, ly + th + 4], fill=LABEL_BG)
        draw.text((lx + 3, ly + 1), label, fill=LABEL_FG, font=font)

    if save:
        out = Path(output_dir) if output_dir else PROCESSED_DIR
        out.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        annotated.save(str(out / f"som_{ts}.png"))
        # Save element map as JSON
        import json
        with open(out / f"som_{ts}.json", "w") as f:
            json.dump(element_map, f, indent=2)

    return annotated, element_map


if __name__ == "__main__":
    from capture import capture_screenshot
    img = capture_screenshot()
    annotated, emap = annotate_screenshot(img, save=True)
    print(f"Detected {len(emap)} UI elements")
    for eid, info in list(emap.items())[:5]:
        print(f"  [{eid}] center=({info['cx']},{info['cy']}) size={info['w']}x{info['h']}")
