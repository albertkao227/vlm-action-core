"""
prompts.py — System prompts and message builders for the Gemma 4 vision agent.

The system prompt forces the model to output strictly-formatted JSON
describing a single UI action per turn.
"""

import json

# ── JSON action schema (for reference in the prompt) ────────────────
ACTION_SCHEMA = {
    "action": "click | type | drag | scroll | hotkey | done",
    "element_id": "int — the SoM label number to target",
    "text": "str — text to type (only for 'type' action)",
    "target_id": "int — destination element for 'drag' action",
    "direction": "str — 'up' or 'down' (only for 'scroll')",
    "keys": "str — key combo e.g. 'command+c' (only for 'hotkey')",
    "reason": "str — one-sentence explanation of why you chose this action",
}

SYSTEM_PROMPT = f"""\
You are a precise UI-navigation agent running on macOS.

## Your Capabilities
You see a screenshot of the user's screen with **numbered red bounding boxes** \
(Set-of-Mark annotations). Each number identifies a clickable or interactive UI element.

## Your Task
Given the user's goal and the annotated screenshot, decide the **single next action** \
that makes progress toward the goal.

## Output Format
Respond with **ONLY** a JSON object — no markdown fences, no prose, no explanation \
outside the JSON. The JSON must follow this schema:

```
{json.dumps(ACTION_SCHEMA, indent=2)}
```

### Action Types
- **click**: Click the center of element `element_id`.
- **type**: Click element `element_id`, then type `text`.
- **drag**: Drag from element `element_id` to element `target_id`.
- **scroll**: Scroll `direction` ("up" or "down") at the current mouse position.
- **hotkey**: Press a keyboard shortcut specified by `keys` (e.g. "command+c").
- **done**: The task is complete. Use this when the goal has been achieved.

### Rules
1. Always include `"action"` and `"reason"`.
2. Include only the fields relevant to the chosen action.
3. If no numbered element matches the needed target, use `"action": "scroll"` to \
reveal more UI, or `"action": "done"` if the task appears impossible.
4. Never fabricate element IDs that don't exist in the screenshot.
5. Prefer the simplest action that makes progress.
"""


def build_user_message(
    task: str,
    history: list[dict] | None = None,
    step: int = 1,
) -> str:
    """
    Build the user-turn text that accompanies the annotated screenshot.

    Args:
        task: The natural-language goal the user wants to achieve.
        history: List of previous action dicts (for multi-step context).
        step: Current step number.

    Returns:
        Formatted user message string.
    """
    parts = [f"**Goal:** {task}", f"**Step:** {step}"]

    if history:
        recent = history[-5:]  # only last 5 to save context
        summary = "\n".join(
            f"  Step {i+1}: {h.get('action','?')} → {h.get('reason','')}"
            for i, h in enumerate(recent)
        )
        parts.append(f"**Previous actions:**\n{summary}")

    parts.append(
        "Analyze the annotated screenshot and output the next action as JSON."
    )
    return "\n\n".join(parts)
