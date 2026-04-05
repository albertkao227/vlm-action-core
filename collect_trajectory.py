#!/usr/bin/env python3
"""
collect_trajectory.py — Record agent trajectories for fine-tuning.

Wraps the main agent loop and saves each step as a training example:
  - SoM-annotated screenshot → data/processed/
  - Action JSON + prompt → data/trajectories.jsonl

Usage:
    python collect_trajectory.py --task "open Terminal" --max-steps 5
"""

import argparse
import json
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


DATA_DIR = Path(__file__).resolve().parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
TRAJECTORIES_FILE = DATA_DIR / "trajectories.jsonl"


def collect(
    task: str,
    model: str = "google/gemma-4-e2b-it",
    max_steps: int = 10,
    execute: bool = True,
    delay: float = 2.0,
):
    """
    Run the agent loop and record every (screenshot, action) pair
    to disk for later fine-tuning.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    client = GemmaVisionClient(model=model)
    if not client.check_model():
        print(f"✗ Model '{model}' could not be loaded. Check model name.")
        sys.exit(1)

    print(f"Recording trajectory for: {task}")
    history: list[dict] = []
    trajectory: list[dict] = []

    for step in range(1, max_steps + 1):
        print(f"\n── Step {step}/{max_steps} ──")

        # 1. Capture & annotate
        screenshot = capture_screenshot()
        annotated, element_map = annotate_screenshot(screenshot)

        if not element_map:
            print("  ⚠ No elements detected, skipping")
            time.sleep(delay)
            continue

        # Save annotated image
        ts = time.strftime("%Y%m%d_%H%M%S")
        img_name = f"traj_{ts}_step{step:03d}.png"
        img_path = PROCESSED_DIR / img_name
        annotated.save(str(img_path))

        # 2. Query model
        user_msg = build_user_message(task, history=history, step=step)
        try:
            action = client.infer(image_path=img_path, user_message=user_msg)
        except ValueError as exc:
            print(f"  ✗ Inference failed: {exc}")
            break

        action = parse_action(action)
        is_valid, err = validate_action(action)
        if not is_valid:
            print(f"  ⚠ Invalid action: {err}")
            break

        # 3. Record training example
        example = {
            "image": str(img_path),
            "task": task,
            "step": step,
            "prompt": user_msg,
            "response": action,
            "element_map_size": len(element_map),
        }
        trajectory.append(example)
        print(f"  Recorded: {action['action']} — {action.get('reason', '')}")

        # Save element map alongside the image
        with open(PROCESSED_DIR / f"traj_{ts}_step{step:03d}.json", "w") as f:
            json.dump({"action": action, "element_map": element_map}, f, indent=2)

        if action["action"] == "done":
            print("\n  ✓ Agent marked task as done.")
            break

        # 4. Execute (optional — use for interactive collection)
        if execute:
            result = execute_action(action, element_map, dry_run=False)
            print(f"  Executed: {result}")

        history.append(action)
        time.sleep(delay)

    # Append trajectory to JSONL file
    with open(TRAJECTORIES_FILE, "a") as f:
        for ex in trajectory:
            # Convert response dict to string for JSON serialization
            ex_out = {**ex, "response": json.dumps(ex["response"])}
            f.write(json.dumps(ex_out) + "\n")

    print(f"\n{'═' * 50}")
    print(f"  Saved {len(trajectory)} examples to {TRAJECTORIES_FILE}")
    print(f"  Images in {PROCESSED_DIR}")
    print(f"{'═' * 50}")


def main():
    parser = argparse.ArgumentParser(description="Record agent trajectories")
    parser.add_argument("--task", required=True, help="Task description")
    parser.add_argument("--model", default="google/gemma-4-e2b-it")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument(
        "--no-execute", action="store_true",
        help="Record model outputs without executing actions",
    )
    parser.add_argument("--delay", type=float, default=2.0)
    args = parser.parse_args()

    collect(
        task=args.task,
        model=args.model,
        max_steps=args.max_steps,
        execute=not args.no_execute,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
