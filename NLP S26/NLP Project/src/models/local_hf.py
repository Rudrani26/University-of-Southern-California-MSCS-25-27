import time

from transformers import pipeline, AutoTokenizer

from .base import LLMBackend
from prompts.parse_response import extract_choice


class LocalHFBackend(LLMBackend):
    """
    LLMBackend implementation for Llama-3.1-8B-Instruct running locally
    via a Hugging Face text-generation pipeline.

    Deterministic decoding (temperature=0, do_sample=False) is enforced
    so results are reproducible across runs.
    """

    def __init__(self, model_id: str, device: str = "cuda"):
        """
        Load model and tokenizer.

        Args:
            model_id: HuggingFace model identifier,
                      e.g. "meta-llama/Llama-3.1-8B-Instruct"
            device:   "cuda" (default) or "cpu"
        """
        self.model_id = model_id

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        self.pipe = pipeline(
            "text-generation",
            model=model_id,
            tokenizer=self.tokenizer,
            device=0 if device == "cuda" else -1,
            # Deterministic decoding — required by the project spec
            temperature=0,
            do_sample=False,
        )

    def predict(self, prompt: str) -> dict:
        """
        Run inference and return a standardised result dict.

        Token counts:
            input_tokens  — tokens in the prompt
            output_tokens — tokens in the *generated* portion only
                            (total_tokens - input_tokens)

        Latency is measured with time.perf_counter() for sub-millisecond
        resolution and converted to milliseconds.

        Returns:
            {
                "prediction":    str | None,   # A/B/C/D or None
                "input_tokens":  int,
                "output_tokens": int,
                "latency_ms":    float,
            }
        Never raises; returns prediction=None if output cannot be parsed.
        """
        # Count input tokens directly from the tokenizer (no model call yet)
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        input_tokens = len(input_ids[0]) if isinstance(input_ids, list) else input_ids.shape[-1]

        # --- Inference (timed) ---
        start = time.perf_counter()
        try:
            outputs = self.pipe(
                prompt,
                max_new_tokens=16,          # answer is a single letter
                return_full_text=False,     # generated text only, no prompt echo
            )
            raw_output = outputs[0]["generated_text"]
        except Exception:
            # Safety net — never let a model error propagate to callers
            raw_output = ""
        end = time.perf_counter()

        latency_ms = (end - start) * 1000.0

        # Count generated tokens
        output_ids = self.tokenizer(raw_output, add_special_tokens=False)["input_ids"]
        output_tokens = len(output_ids)

        # Parse the single-letter answer
        prediction = extract_choice(raw_output)

        return {
            "prediction":    prediction,
            "input_tokens":  int(input_tokens),
            "output_tokens": int(output_tokens),
            "latency_ms":    latency_ms,
        }
