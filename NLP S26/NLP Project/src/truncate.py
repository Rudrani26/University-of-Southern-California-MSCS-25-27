import copy
import json
from pathlib import Path
from typing import Any


BUDGET_LABELS = {
    8192: "8k",
    32768: "32k",
    65536: "64k",
    131072: "128k",
}


def truncate(instance: dict, budget: int, tokenizer) -> dict:
    """
    Tokenize instance["input"], keep the first `budget` tokens, decode back
    to text, and return a shallow copy with only "input" replaced.

    The original instance is never modified.

    Args:
        instance: A validated LongBench v2 instance dict.
        budget: Maximum number of tokens to keep in the context.
        tokenizer: A Hugging Face tokenizer or compatible equivalent.

    Returns:
        A new dict identical to `instance` except for the truncated "input".
    """
    if budget <= 0:
        raise ValueError(f"Budget must be a positive integer, got {budget}")

    if "input" not in instance:
        raise KeyError("instance is missing required key 'input'")

    token_ids = tokenizer(
        instance["input"],
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]

    truncated_ids = token_ids[:budget]

    truncated_text = tokenizer.decode(
        truncated_ids,
        skip_special_tokens=True,
    )

    new_instance = copy.copy(instance)
    new_instance["input"] = truncated_text
    return new_instance


def generate_splits(
    instances: list[dict],
    budgets: list[int],
    tokenizer,
    out_dir: str,
) -> None:
    """
    For each budget:
      1. Call truncate() on every instance.
      2. Write results to out_dir/{label}.jsonl as newline-delimited JSON.
      3. Skip budgets where the output file already exists.

    Args:
        instances: Flat list of validated LongBench v2 instances.
        budgets: List of token budgets, e.g. [8192, 32768, 65536, 131072].
        tokenizer: A Hugging Face tokenizer.
        out_dir: Directory to write JSONL splits into.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for budget in budgets:
        label = BUDGET_LABELS.get(budget, f"{budget // 1000}k")
        out_file = out_path / f"{label}.jsonl"

        if out_file.exists():
            print(f"[truncate] Skipping {label} — {out_file} already exists.")
            continue

        print(f"[truncate] Generating {label} split ({len(instances)} instances)...")
        with out_file.open("w", encoding="utf-8") as f:
            for inst in instances:
                truncated = truncate(inst, budget, tokenizer)
                f.write(json.dumps(truncated, ensure_ascii=False) + "\n")

        print(f"[truncate] Wrote {label} split → {out_file}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    print("Running truncate tests...\n")

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        print("  Loaded gpt2 tokenizer for tests.\n")
    except Exception as e:
        raise RuntimeError(
            "Could not load a test tokenizer. Install transformers first.\n"
            f"Original error: {e}"
        ) from e

    def _make_instance(ctx: str | None = None) -> dict[str, Any]:
        return {
            "id": "test-001",
            "input": ctx or " ".join([f"word{i}" for i in range(20_000)]),
            "question": "What is the main topic?",
            "choices": ["Alpha", "Beta", "Gamma", "Delta"],
            "answer": "C",
            "category": "single-document QA",
        }

    # Test 1: truncated token count <= budget
    print("Test 1: truncated token length <= budget")
    budget = 8192
    inst = _make_instance()
    result = truncate(inst, budget, tokenizer)
    actual_len = len(tokenizer(result["input"], add_special_tokens=False)["input_ids"])
    assert actual_len <= budget, f"Expected <= {budget} tokens, got {actual_len}"
    print(f"  ✓ Token count after truncation: {actual_len} (budget: {budget})")

    # Test 2: question and choices are unchanged
    print("Test 2: question and choices are unchanged")
    assert result["question"] == inst["question"], "question was modified"
    assert result["choices"] == inst["choices"], "choices was modified"
    assert result["answer"] == inst["answer"], "answer was modified"
    assert result["category"] == inst["category"], "category was modified"
    assert result["id"] == inst["id"], "id was modified"
    print("  ✓ All non-input fields are unchanged")

    # Test 3: original instance is not mutated
    print("Test 3: original instance is not mutated")
    original_input = inst["input"]
    _ = truncate(inst, 100, tokenizer)
    assert inst["input"] == original_input, "Original instance was mutated"
    print("  ✓ Original instance untouched")

    # Test 4: context shorter than budget passes through intact
    print("Test 4: short context passes through intact")
    short_ctx = "This is a very short context."
    short_inst = _make_instance(ctx=short_ctx)
    result_short = truncate(short_inst, budget, tokenizer)
    short_ids = tokenizer(short_ctx, add_special_tokens=False)["input_ids"]
    result_short_ids = tokenizer(result_short["input"], add_special_tokens=False)["input_ids"]
    assert len(result_short_ids) == len(short_ids), "Short context token count changed"
    print("  ✓ Short context token count unchanged")

    # Test 5: invalid budget raises ValueError
    print("Test 5: invalid budget raises ValueError")
    try:
        truncate(inst, 0, tokenizer)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        print(f"  ✓ Raised ValueError as expected: {e}")

    # Test 6: generate_splits writes correct files and skips existing
    print("Test 6: generate_splits writes correct JSONL files")
    instances = [_make_instance(ctx=" ".join([f"word{i}" for i in range(300)])) for _ in range(5)]
    with tempfile.TemporaryDirectory() as tmpdir:
        budgets = [128, 256]
        generate_splits(instances, budgets, tokenizer, tmpdir)

        for b in budgets:
            label = BUDGET_LABELS.get(b, f"{b // 1000}k")
            out_file = Path(tmpdir) / f"{label}.jsonl"
            assert out_file.exists(), f"Missing split file: {out_file}"
            lines = out_file.read_text(encoding="utf-8").strip().split("\n")
            assert len(lines) == 5, f"Expected 5 lines, got {len(lines)}"
            for line in lines:
                record = json.loads(line)
                toks = tokenizer(record["input"], add_special_tokens=False)["input_ids"]
                assert len(toks) <= b, f"Token count {len(toks)} exceeds budget {b}"

        generate_splits(instances, budgets, tokenizer, tmpdir)
        print("  ✓ generate_splits skips already-existing files")

    print("\nAll tests passed ✓")