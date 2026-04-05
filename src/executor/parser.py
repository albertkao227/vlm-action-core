"""
parser.py — Parse and validate LLM action output.

Strips conversational filler, extracts JSON, and validates that the
action dict contains the required fields for its action type.
"""

import json
import re
from typing import Any


VALID_ACTIONS = {"click", "type", "drag", "scroll", "hotkey", "done"}

# Required keys per action type (beyond 'action' and 'reason')
REQUIRED_KEYS: dict[str, set[str]] = {
    "click":  {"element_id"},
    "type":   {"element_id", "text"},
    "drag":   {"element_id", "target_id"},
    "scroll": {"direction"},
    "hotkey": {"keys"},
    "done":   set(),
}


def parse_action(raw: str | dict) -> dict:
    """
    Extract an action dict from raw LLM output.

    Accepts either a pre-parsed dict or a raw string (which may
    contain markdown fences or conversational filler).

    Returns:
        Parsed action dict.

    Raises:
        ValueError: If no valid JSON could be extracted.
    """
    if isinstance(raw, dict):
        return raw

    text = raw.strip()

    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip code fences
    fenced = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1).strip())

    # Find first JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))

    raise ValueError(f"Cannot parse action from: {text[:200]}")


def validate_action(action: dict) -> tuple[bool, str]:
    """
    Validate that an action dict is well-formed.

    Returns:
        (is_valid, error_message)  — error_message is "" when valid.
    """
    if not isinstance(action, dict):
        return False, "Action must be a dict"

    act = action.get("action")
    if act not in VALID_ACTIONS:
        return False, f"Unknown action '{act}'. Valid: {VALID_ACTIONS}"

    required = REQUIRED_KEYS.get(act, set())
    missing = required - set(action.keys())
    if missing:
        return False, f"Action '{act}' missing required keys: {missing}"

    # Type checks
    if "element_id" in action and not isinstance(action["element_id"], int):
        return False, "element_id must be an int"
    if "target_id" in action and not isinstance(action["target_id"], int):
        return False, "target_id must be an int"
    if "text" in action and not isinstance(action["text"], str):
        return False, "text must be a string"
    if "direction" in action and action["direction"] not in ("up", "down"):
        return False, "direction must be 'up' or 'down'"

    return True, ""
