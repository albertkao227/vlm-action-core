"""
actions.py — Execute UI actions via PyAutoGUI.

Translates validated action dicts + element_map coordinates into
real mouse/keyboard events on macOS.
"""

import time
from typing import Any

import pyautogui

# Safety: move mouse to corner = abort
pyautogui.FAILSAFE = True
# Small pause between pyautogui calls for stability
pyautogui.PAUSE = 0.15


def execute_action(
    action: dict,
    element_map: dict[int, dict[str, Any]],
    dry_run: bool = False,
) -> str:
    """
    Dispatch a validated action dict to PyAutoGUI.

    Args:
        action: Validated action dict with 'action', 'element_id', etc.
        element_map: {element_id: {"cx": int, "cy": int, ...}} from SoM.
        dry_run: If True, print what would happen but don't touch the UI.

    Returns:
        Human-readable description of what was (or would be) done.
    """
    act = action["action"]

    if act == "done":
        return "Task marked as done."

    if act == "scroll":
        direction = action.get("direction", "down")
        clicks = 5 if direction == "down" else -5
        desc = f"Scroll {direction}"
        if not dry_run:
            pyautogui.scroll(clicks)
        return desc

    if act == "hotkey":
        keys = action["keys"]
        key_list = [k.strip() for k in keys.split("+")]
        desc = f"Hotkey: {'+'.join(key_list)}"
        if not dry_run:
            pyautogui.hotkey(*key_list)
        return desc

    # Actions that need an element_id
    eid = action.get("element_id")
    if eid is None or eid not in element_map:
        return f"⚠ element_id {eid} not found in element_map (has {list(element_map.keys())[:10]})"

    elem = element_map[eid]
    cx, cy = elem["cx"], elem["cy"]

    if act == "click":
        desc = f"Click element [{eid}] at ({cx}, {cy})"
        if not dry_run:
            pyautogui.click(cx, cy)
        return desc

    if act == "type":
        text = action.get("text", "")
        desc = f"Click [{eid}] at ({cx}, {cy}) then type '{text}'"
        if not dry_run:
            pyautogui.click(cx, cy)
            time.sleep(0.1)
            pyautogui.write(text, interval=0.03)
        return desc

    if act == "drag":
        tid = action.get("target_id")
        if tid is None or tid not in element_map:
            return f"⚠ target_id {tid} not found in element_map"
        target = element_map[tid]
        tx, ty = target["cx"], target["cy"]
        desc = f"Drag [{eid}] ({cx},{cy}) → [{tid}] ({tx},{ty})"
        if not dry_run:
            pyautogui.moveTo(cx, cy, duration=0.2)
            pyautogui.dragTo(tx, ty, duration=0.5, button="left")
        return desc

    return f"⚠ Unknown action type: {act}"
