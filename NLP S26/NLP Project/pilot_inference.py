"""
Usage:
    python pilot_inference.py \
        --data_dir data/longbench_v2 \
        --splits_dir data/splits \
        --model_id meta-llama/Llama-3.1-8B-Instruct \
        --out_file results/pilot_8k.jsonl \
        [--device cuda] \
        [--dev]
"""

import argparse
import json
import random
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))

from dataset_loader import get_by_category, load
from src.logging_utils import InferenceLogger
from src.models.local_hf import LocalHFBackend
from src.prompts.direct_answer import build_da_prompt
from src.truncate import generate_splits

BUDGET = 8_192
BUDGET_LABEL = "8k"
TOTAL_SAMPLES = 50
SEED = 42
PROMPT_STRATEGY = "DA"

REQUIRED_RECORD_KEYS = {
    "instance_id",
    "category",
    "budget",
    "prompt_strategy",
    "model_id",
    "prediction",
    "ground_truth",
    "correct",
    "input_tokens",
    "output_tokens",
    "latency_ms",
}


def stratified_sample(
    instances: list[dict],
    n: int,
    seed: int = SEED,
) -> list[dict]:
    """
    Draws n instances stratified across categories.

    Each category gets floor(n / num_categories) samples; any remainder
    is distributed across categories in sorted order.

    Raises ValueError if any category does not have enough examples.
    """
    rng = random.Random(seed)
    grouped = get_by_category(instances)
    categories = sorted(grouped.keys())

    if not categories:
        raise ValueError("No categories found in the dataset")

    num_cats = len(categories)
    base_per_cat = n // num_cats
    remainder = n % num_cats

    sampled: list[dict] = []
    for i, cat in enumerate(categories):
        pool = grouped[cat]
        k = base_per_cat + (1 if i < remainder else 0)

        if len(pool) < k:
            raise ValueError(
                f"Category '{cat}' has only {len(pool)} instances, but {k} were requested."
            )

        sampled.extend(rng.sample(pool, k))

    if len(sampled) != n:
        raise RuntimeError(f"Expected {n} samples, got {len(sampled)}")

    return sampled


def main(args: argparse.Namespace) -> None:
    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Avoid appending duplicate rows if rerun
    if out_file.exists():
        out_file.unlink()

    # 1. Load full dataset
    print(f"[pilot] Loading dataset from {args.data_dir} ...")
    instances = load(args.data_dir, strict=not args.dev)
    print(f"[pilot] Loaded {len(instances)} instances.")

    # 2. Generate 8K split if needed
    split_file = Path(args.splits_dir) / f"{BUDGET_LABEL}.jsonl"
    if not split_file.exists():
        print(f"[pilot] {BUDGET_LABEL} split not found — generating via truncate ...")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        generate_splits(instances, [BUDGET], tokenizer, args.splits_dir)
    else:
        print(f"[pilot] {BUDGET_LABEL} split already exists at {split_file}, loading ...")

    # Load truncated instances from split
    truncated_instances: list[dict] = []
    with open(split_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                truncated_instances.append(json.loads(line))

    print(f"[pilot] Loaded {len(truncated_instances)} truncated instances from split.")

    # 3. Stratified sample of 50
    sampled = stratified_sample(truncated_instances, TOTAL_SAMPLES, seed=SEED)
    print(f"[pilot] Sampled {len(sampled)} instances (stratified, seed={SEED}).")

    # Category breakdown for transparency
    cat_counts: dict[str, int] = {}
    for inst in sampled:
        cat_counts[inst["category"]] = cat_counts.get(inst["category"], 0) + 1

    print("[pilot] Sample breakdown by category:")
    for cat in sorted(cat_counts):
        print(f"         {cat}: {cat_counts[cat]}")

    # 4. Load model
    print(f"[pilot] Loading model '{args.model_id}' on {args.device} ...")
    model = LocalHFBackend(model_id=args.model_id, device=args.device)
    print("[pilot] Model loaded.")

    # 5. Run inference and log results
    print(f"[pilot] Running inference → {out_file}")
    correct_count = 0

    with InferenceLogger(str(out_file)) as logger:
        for idx, inst in enumerate(sampled, start=1):
            prompt = build_da_prompt(inst)
            result = model.predict(prompt)

            prediction = result["prediction"]
            ground_truth = inst["answer"]
            is_correct = prediction == ground_truth

            if is_correct:
                correct_count += 1

            record = {
                "instance_id": inst["id"],
                "category": inst["category"],
                "budget": BUDGET,
                "prompt_strategy": PROMPT_STRATEGY,
                "model_id": args.model_id,
                "prediction": prediction,
                "ground_truth": ground_truth,
                "correct": is_correct,
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
                "latency_ms": result["latency_ms"],
            }

            # Safety check for schema drift
            missing = REQUIRED_RECORD_KEYS - set(record.keys())
            if missing:
                raise KeyError(f"Missing required record keys: {sorted(missing)}")

            logger.log(record)

            status = "✓" if is_correct else "✗"
            print(
                f"  [{idx:02d}/{TOTAL_SAMPLES}] {status}  "
                f"pred={prediction}  gt={ground_truth}  "
                f"{result['latency_ms']:.0f}ms  "
                f"cat={inst['category']}"
            )

    accuracy = (correct_count / len(sampled)) * 100
    print(f"\n[pilot] Done. Accuracy: {correct_count}/{len(sampled)} = {accuracy:.1f}%")
    print(f"[pilot] Results written to {out_file}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pilot inference: 50 stratified samples at 8K budget (Direct Answer)."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/longbench_v2",
        help="Directory containing raw LongBench v2 JSON files.",
    )
    parser.add_argument(
        "--splits_dir",
        type=str,
        default="data/splits",
        help="Directory to read/write truncated JSONL splits.",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model ID to load at runtime.",
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default="results/pilot_8k.jsonl",
        help="Path to write pilot result JSONL.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for model inference: 'cuda' or 'cpu'.",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        default=False,
        help=(
            "Development mode: disables strict dataset validation so the script "
            "can run on partial subsets of LongBench v2."
        ),
    )
    return parser.parse_args()



def _run_tests() -> None:
    print("Running pilot_inference tests...\n")

    def _fake_instance(cat: str, uid: str) -> dict:
        return {
            "id": uid,
            "input": "ctx",
            "question": "q?",
            "choices": ["A", "B", "C", "D"],
            "answer": "A",
            "category": cat,
        }

    cats = [
        "single-document QA",
        "multi-document QA",
        "long in-context learning",
        "dialogue history understanding",
        "code repository reasoning",
        "structured data understanding",
    ]

    fake_instances = []
    for cat in cats:
        for j in range(100):
            fake_instances.append(_fake_instance(cat, f"{cat}-{j}"))

    # Test 1: stratified_sample returns correct count
    print("Test 1: stratified_sample returns exactly 50 instances")
    sampled = stratified_sample(fake_instances, 50, seed=42)
    assert len(sampled) == 50, f"Expected 50, got {len(sampled)}"
    print(f"  ✓ Got {len(sampled)} instances")

    # Test 2: all 6 categories represented
    print("Test 2: all 6 categories represented in sample")
    sampled_cats = {inst["category"] for inst in sampled}
    for cat in cats:
        assert cat in sampled_cats, f"Missing category: {cat}"
    print("  ✓ All 6 categories present")

    # Test 3: deterministic with same seed
    print("Test 3: sampling is deterministic with same seed")
    sampled2 = stratified_sample(fake_instances, 50, seed=42)
    assert [i["id"] for i in sampled] == [i["id"] for i in sampled2]
    print("  ✓ Same seed produces identical sample")

    # Test 4: different seed produces different sample
    print("Test 4: different seed produces different sample")
    sampled3 = stratified_sample(fake_instances, 50, seed=99)
    assert [i["id"] for i in sampled] != [i["id"] for i in sampled3]
    print("  ✓ Different seed produces different sample")

    # Test 5: result record schema matches shared contract
    print("Test 5: result record contains all required schema fields")
    mock_record = {
        "instance_id": "x",
        "category": "single-document QA",
        "budget": 8192,
        "prompt_strategy": "DA",
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "prediction": "B",
        "ground_truth": "A",
        "correct": False,
        "input_tokens": 100,
        "output_tokens": 3,
        "latency_ms": 1200.0,
    }
    assert REQUIRED_RECORD_KEYS == set(mock_record.keys()), "Schema mismatch"
    print("  ✓ Record schema matches shared contract")

    # Test 6: ValueError raised when category has too few instances
    print("Test 6: ValueError raised when category has too few instances")
    thin_instances = []
    for cat in cats:
        for j in range(5):  # only 5 per category, not enough for 50 total
            thin_instances.append(_fake_instance(cat, f"{cat}-{j}"))
    try:
        stratified_sample(thin_instances, 50, seed=42)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        print(f"  ✓ Raised ValueError as expected: {e}")

    print("\nAll tests passed ✓")


if __name__ == "__main__":
    if "--test" in sys.argv:
        _run_tests()
    else:
        args = _parse_args()
        main(args)