#!/usr/bin/env python3
"""
main.py — Continuous observe → think → act loop for the VLM UI agent.

Usage:
    python main.py --task "open Safari and go to google.com"
    python main.py --task "click the Finder icon" --dry-run --max-steps 3
    python main.py --task "open Terminal" --model google/gemma-4-e4b-it
"""

import argparse
import sys
import tempfile
import time
from pathlib import Path

from src.observer.capture import capture_screenshot
from src.observer.som_processor import annotate_screenshot
from src.engine.llm_client import GemmaVisionClient
from src.engine.prompts import build_user_message
from src.executor.parser import parse_action, validate_action
from src.executor.actions import execute_action


def run_agent(
    task: str,
    model: str = "google/gemma-4-e2b-it",
    max_steps: int = 10,
    dry_run: bool = False,
    delay: float = 1.5,
    verbose: bool = True,
):
    """
    Main agent loop: observe → think → act.

    Args:
        task: Natural-language description of the goal.
        model: Ollama model name.
        max_steps: Safety cap on iterations.
        dry_run: If True, print actions but don't execute them.
        delay: Seconds to wait between steps (lets the UI settle).
        verbose: Print step-by-step details.
    """
    client = GemmaVisionClient(model=model)

    # Pre-flight check
    if not client.check_model():
        print(f"✗ Model '{model}' could not be loaded.")
        print(f"  Check your internet connection or model name.")
        sys.exit(1)

    print(f"{'═' * 60}")
    print(f"  VLM UI Agent")
    print(f"  Model : {model}")
    print(f"  Task  : {task}")
    print(f"  Mode  : {'DRY RUN' if dry_run else 'LIVE'}")
    print(f"  Steps : max {max_steps}")
    print(f"{'═' * 60}\n")

    history: list[dict] = []
    tmp_dir = Path(tempfile.mkdtemp(prefix="vlm_agent_"))

    for step in range(1, max_steps + 1):
        if verbose:
            print(f"── Step {step}/{max_steps} {'─' * 40}")

        # 1. Observe
        if verbose:
            print("  📸 Capturing screenshot…")
        screenshot = capture_screenshot()

        # 2. Annotate with Set-of-Mark
        if verbose:
            print("  🏷️  Annotating with SoM…")
        annotated, element_map = annotate_screenshot(screenshot)

        if not element_map:
            print("  ⚠ No UI elements detected. Retrying…")
            time.sleep(delay)
            continue

        if verbose:
            print(f"  Found {len(element_map)} elements")

        # Save annotated image for the LLM
        img_path = tmp_dir / f"step_{step:03d}.png"
        annotated.save(str(img_path))

        # 3. Think
        if verbose:
            print("  🧠 Querying Gemma 4…")
        user_msg = build_user_message(task, history=history, step=step)

        try:
            action = client.infer(
                image_path=img_path,
                user_message=user_msg,
                history=None,  # keep context short for small models
            )
        except ValueError as exc:
            print(f"  ✗ LLM inference failed: {exc}")
            break

        # 4. Parse & validate
        action = parse_action(action)
        is_valid, err = validate_action(action)
        if not is_valid:
            print(f"  ⚠ Invalid action: {err}")
            print(f"    Raw: {action}")
            break

        # 5. Check for completion
        if action["action"] == "done":
            reason = action.get("reason", "no reason given")
            print(f"\n  ✓ Agent says DONE: {reason}\n")
            break

        # 6. Act
        prefix = "  🖱️  [DRY]" if dry_run else "  🖱️ "
        result = execute_action(action, element_map, dry_run=dry_run)
        if verbose:
            print(f"{prefix} {result}")
            print(f"     Reason: {action.get('reason', '—')}")

        history.append(action)

        # Brief pause to let the UI update
        time.sleep(delay)
    else:
        print(f"\n  ⚠ Reached max steps ({max_steps}) without completion.\n")

    print(f"{'═' * 60}")
    print(f"  Agent finished after {len(history)} action(s).")
    print(f"{'═' * 60}")
    return history


def main():
    parser = argparse.ArgumentParser(
        description="VLM UI Navigation Agent — observe, think, act",
    )
    parser.add_argument(
        "--task", required=True,
        help="Natural-language task, e.g. 'open Safari'",
    )
    parser.add_argument(
        "--model", default="google/gemma-4-e2b-it",
        help="HuggingFace model ID (default: google/gemma-4-e2b-it)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=10,
        help="Maximum actions before stopping (default: 10)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print actions without executing them",
    )
    parser.add_argument(
        "--delay", type=float, default=1.5,
        help="Seconds between steps (default: 1.5)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress step-by-step output",
    )
    args = parser.parse_args()

    run_agent(
        task=args.task,
        model=args.model,
        max_steps=args.max_steps,
        dry_run=args.dry_run,
        delay=args.delay,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
