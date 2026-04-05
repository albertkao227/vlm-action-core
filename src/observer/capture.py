"""
capture.py — Screen capture using mss.

Grabs the primary monitor and returns a PIL Image.
Optionally saves raw screenshots to data/raw/.
"""

import os
import time
from pathlib import Path

import mss
from PIL import Image


# Default output directory for raw screenshots
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def capture_screenshot(
    save: bool = False,
    output_dir: Path | str | None = None,
    monitor_index: int = 1,
) -> Image.Image:
    """
    Capture a screenshot of the specified monitor.

    Args:
        save: If True, persist the screenshot as a PNG in output_dir.
        output_dir: Directory to save to (defaults to data/raw/).
        monitor_index: 1 = primary monitor (mss convention).

    Returns:
        PIL.Image.Image in RGB mode.
    """
    with mss.mss() as sct:
        monitor = sct.monitors[monitor_index]
        raw = sct.grab(monitor)
        img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")

    if save:
        out = Path(output_dir) if output_dir else RAW_DIR
        out.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        filepath = out / f"screenshot_{ts}.png"
        img.save(str(filepath))

    return img


if __name__ == "__main__":
    img = capture_screenshot(save=True)
    print(f"Captured screenshot: {img.size}")
