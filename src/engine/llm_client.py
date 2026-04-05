"""
llm_client.py — HuggingFace Transformers client for Gemma 4 vision inference.

Uses AutoProcessor + AutoModelForCausalLM with MPS (Metal) backend
for local inference on Apple Silicon.
"""

import json
import re
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

from .prompts import SYSTEM_PROMPT


# Default model — Gemma 4 E2B instruction-tuned (smallest, fastest)
DEFAULT_MODEL = "google/gemma-4-e2b-it"


class GemmaVisionClient:
    """
    Wraps HuggingFace Transformers to run Gemma 4 vision inference
    on Apple Silicon via MPS.

    Usage:
        client = GemmaVisionClient()
        action = client.infer("screenshot.png", "Click the search bar")
    """

    def __init__(self, model: str = DEFAULT_MODEL, dtype: str = "float16"):
        self.model_id = model
        self.dtype = getattr(torch, dtype, torch.float16)
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self._model = None
        self._processor = None

    def _ensure_loaded(self):
        """Lazy-load the model on first inference call."""
        if self._model is None:
            print(f"  Loading model: {self.model_id} …")
            print(f"  Device: {self.device} | dtype: {self.dtype}")
            self._processor = AutoProcessor.from_pretrained(self.model_id)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
                device_map=self.device,
            )
            print(f"  ✓ Model loaded")

    # ── public ──────────────────────────────────────────────────────

    def infer(
        self,
        image_path: str | Path,
        user_message: str,
        max_tokens: int = 300,
        temperature: float = 0.2,
        max_retries: int = 2,
    ) -> dict:
        """
        Send an annotated screenshot + prompt to Gemma 4 and return
        the parsed action dict.

        Args:
            image_path: Path to the SoM-annotated screenshot PNG.
            user_message: The user-turn text (from prompts.build_user_message).
            max_tokens: Max new tokens to generate.
            temperature: Sampling temperature (low = deterministic).
            max_retries: Retries if JSON parsing fails.

        Returns:
            Parsed action dict, e.g. {"action": "click", "element_id": 3, ...}

        Raises:
            ValueError: If the model never returns valid JSON.
        """
        self._ensure_loaded()

        # Load the image
        img = Image.open(image_path).convert("RGB")

        # Build chat messages
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_message},
                ],
            },
        ]

        last_error = None
        for attempt in range(1, max_retries + 1):
            # Apply chat template
            prompt = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Process inputs
            inputs = self._processor(
                text=prompt,
                images=[img],
                return_tensors="pt",
            ).to(self.device)

            # Generate
            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature if temperature > 0 else None,
                    do_sample=temperature > 0,
                )

            # Decode only the new tokens
            generated = output_ids[0, inputs["input_ids"].shape[1]:]
            raw = self._processor.decode(generated, skip_special_tokens=True)

            try:
                action = self._extract_json(raw)
                return action
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                # Retry with stronger JSON instruction
                messages[-1]["content"][-1]["text"] = (
                    f"{user_message}\n\n"
                    "IMPORTANT: Respond with ONLY a JSON object, no other text."
                )

        raise ValueError(
            f"Failed to get valid JSON after {max_retries} attempts. "
            f"Last error: {last_error}"
        )

    def check_model(self) -> bool:
        """Check if the model can be loaded. Returns True if successful."""
        try:
            self._ensure_loaded()
            return True
        except Exception as exc:
            print(f"  ✗ Could not load model: {exc}")
            return False

    # ── private ─────────────────────────────────────────────────────

    @staticmethod
    def _extract_json(text: str) -> dict:
        """
        Pull a JSON object out of the model's response, tolerating
        markdown fences, preamble text, etc.
        """
        text = text.strip()

        # Direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Strip markdown code fences
        fenced = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        if fenced:
            return json.loads(fenced.group(1).strip())

        # Find first { ... } block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))

        raise ValueError(f"No JSON object found in response: {text[:200]}")


if __name__ == "__main__":
    client = GemmaVisionClient()
    print(f"Device: {client.device}")
    if client.check_model():
        print(f"✓ Model '{client.model_id}' loaded successfully")
    else:
        print(f"✗ Failed to load model '{client.model_id}'")
