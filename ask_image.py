#!/usr/bin/env python3
"""
ask_image.py — Ask questions about images using Gemma 4 VLM.

Loads images from data/images/ (or any path) and answers free-form
questions using the HuggingFace Transformers pipeline on MPS.

Usage:
    # Ask about a specific image
    python ask_image.py --image data/images/todo.png --question "What items are on the list?"

    # Interactive mode — loops asking questions about the same image
    python ask_image.py --image data/images/todo.png --interactive

    # Ask about all images in a folder
    python ask_image.py --folder data/images --question "Describe this image"

    # Use a different model
    python ask_image.py --image data/images/todo.png -q "What is this?" --model google/gemma-4-e4b-it
"""

import argparse
import glob
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM


DEFAULT_MODEL = "google/gemma-4-e2b-it"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif", ".tiff"}


def load_model(model_id: str, dtype=torch.float16):
    """Load model + processor, returning (model, processor, device)."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading {model_id} → CPU first, then moving to {device} …")
    processor = AutoProcessor.from_pretrained(model_id)
    # Load to CPU first to avoid MPS single-buffer allocation limit,
    # then move to MPS incrementally via .to()
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype,
    )
    if device == "mps":
        model = model.to(device)
    print("✓ Model loaded\n")
    return model, processor, device


def ask(model, processor, device, image: Image.Image, question: str,
        max_tokens: int = 500, temperature: float = 0.7) -> str:
    """Send an image + question to the model, return the text answer."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        }
    ]

    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(device)

    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            top_p=0.95,
        )
    elapsed = time.time() - t0

    generated = output_ids[0, inputs["input_ids"].shape[1]:]
    answer = processor.decode(generated, skip_special_tokens=True)
    n_tokens = len(generated)
    tok_per_sec = n_tokens / elapsed if elapsed > 0 else 0
    print(f"  ({n_tokens} tokens in {elapsed:.1f}s — {tok_per_sec:.1f} tok/s)")
    return answer


def find_images(folder: str) -> list[Path]:
    """Find all image files in a folder."""
    folder = Path(folder)
    images = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    return images


def main():
    parser = argparse.ArgumentParser(
        description="Ask questions about images using Gemma 4 VLM",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", "-i", help="Path to a single image")
    group.add_argument("--folder", "-f", help="Path to folder of images")

    parser.add_argument("--question", "-q", help="Question to ask about the image(s)")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive mode — keep asking questions")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"HuggingFace model ID (default: {DEFAULT_MODEL})")
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    if not args.question and not args.interactive:
        parser.error("Provide --question or use --interactive mode")

    model, processor, device = load_model(args.model)

    # Collect image paths
    if args.image:
        paths = [Path(args.image)]
    else:
        paths = find_images(args.folder)
        if not paths:
            print(f"No images found in {args.folder}")
            sys.exit(1)
        print(f"Found {len(paths)} images in {args.folder}\n")

    for img_path in paths:
        if not img_path.exists():
            print(f"⚠ Image not found: {img_path}")
            continue

        img = Image.open(img_path).convert("RGB")
        print(f"{'━' * 50}")
        print(f"📷 {img_path.name}  ({img.width}×{img.height})")
        print(f"{'━' * 50}")

        if args.interactive:
            # Interactive loop for this image
            while True:
                try:
                    q = input("\n❓ Question (or 'next'/'quit'): ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    return
                if q.lower() in ("quit", "exit", "q"):
                    return
                if q.lower() in ("next", "n", ""):
                    break
                answer = ask(model, processor, device, img, q,
                             args.max_tokens, args.temperature)
                print(f"\n💬 {answer}")
        else:
            answer = ask(model, processor, device, img, args.question,
                         args.max_tokens, args.temperature)
            print(f"\n💬 {answer}\n")


if __name__ == "__main__":
    main()
