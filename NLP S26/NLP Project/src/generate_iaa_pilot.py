"""
generate_iaa_pilot.py
---------------------
Generates one iaa_pilot_<budget>.csv per input JSONL file so that
annotation results can be compared across context lengths independently.

Output files
------------
    annotation/iaa_pilot_8k.csv
    annotation/iaa_pilot_32k.csv
    annotation/iaa_pilot_64k.csv    (Phase 3)
    annotation/iaa_pilot_128k.csv   (Phase 3)

Usage
-----
    # Phase 2 — 8K only
    python generate_iaa_pilot.py \
        --inputs results/pilot_8k.jsonl

    # Phase 2 — 8K and 32K (run once 32K results are ready)
    python generate_iaa_pilot.py \
        --inputs results/pilot_8k.jsonl results/pilot_32k.jsonl

    # Phase 3 — all four budgets
    python generate_iaa_pilot.py \
        --inputs results/pilot_8k.jsonl results/pilot_32k.jsonl \
                 results/pilot_64k.jsonl results/pilot_128k.jsonl \
        --n 20

Each budget gets its own independent CSV with n instances sampled and
stratified by category from that budget's incorrect predictions only.
Annotators fill each CSV independently so error rates are comparable
across budgets without cross-contamination.
"""

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Output columns
# ---------------------------------------------------------------------------

COLUMNS = [
    "instance_id",
    "category",
    "budget",
    "model_prediction",
    "ground_truth",
    "annotator_a_label",
    "annotator_a_hall",
    "annotator_b_label",
    "annotator_b_hall",
    "resolved_label",
    "resolution_note",
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [WARN] Line {lineno}: malformed JSON — {e}")
    return records


def validate_record(record: dict, lineno: int) -> bool:
    required = {"instance_id", "category", "budget", "prediction", "ground_truth", "correct"}
    missing = required - record.keys()
    if missing:
        print(f"  [WARN] Line {lineno}: missing fields {missing} — skipping.")
        return False
    return True


# ---------------------------------------------------------------------------
# Stratified sampling by category (within a single budget file)
# ---------------------------------------------------------------------------

def stratified_sample(records: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)

    by_category: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_category[r["category"]].append(r)
    for cat in by_category:
        rng.shuffle(by_category[cat])

    categories = list(by_category.keys())
    slots = {cat: n // len(categories) for cat in categories}
    for i, cat in enumerate(categories):
        if i < (n % len(categories)):
            slots[cat] += 1

    sampled: list[dict] = []
    overflow = 0
    for cat in categories:
        take = min(slots[cat], len(by_category[cat]))
        sampled.extend(by_category[cat][:take])
        overflow += slots[cat] - take

    if overflow > 0:
        pool: list[dict] = []
        for cat in categories:
            already = min(slots[cat], len(by_category[cat]))
            pool.extend(by_category[cat][already:])
        rng.shuffle(pool)
        sampled.extend(pool[:overflow])

    rng.shuffle(sampled)
    return sampled[:n]


# ---------------------------------------------------------------------------
# CSV row builder
# ---------------------------------------------------------------------------

def build_row(record: dict) -> dict:
    return {
        "instance_id":       record["instance_id"],
        "category":          record["category"],
        "budget":            record["budget"],
        "model_prediction":  record["prediction"],
        "ground_truth":      record["ground_truth"],
        "annotator_a_label": "",
        "annotator_a_hall":  "",
        "annotator_b_label": "",
        "annotator_b_hall":  "",
        "resolved_label":    "",
        "resolution_note":   "",
    }


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def process_file(path: Path, n: int, seed: int, out_dir: Path) -> Path:
    """
    Load one JSONL file, sample n incorrect predictions stratified by
    category, and write annotation/iaa_pilot_<budget>.csv.
    Returns the output path.
    """
    # Derive budget label from filename (e.g. pilot_8k.jsonl -> 8k)
    stem = path.stem                          # e.g. pilot_8k
    budget_label = stem.split("_")[-1]        # e.g. 8k
    out_path = out_dir / f"iaa_pilot_{budget_label}.csv"

    print(f"\n{'='*55}")
    print(f"  File   : {path.name}")
    print(f"  Output : {out_path.name}")
    print(f"{'='*55}")

    # Load
    raw = load_jsonl(path)
    valid = [r for i, r in enumerate(raw, 1) if validate_record(r, i)]
    print(f"  Total records : {len(raw)}   Valid : {len(valid)}")

    # Filter incorrect
    incorrect = [r for r in valid if not r.get("correct", True)]
    print(f"  Incorrect     : {len(incorrect)}")

    if not incorrect:
        print("  [SKIP] No incorrect predictions — nothing to annotate.")
        return None

    actual_n = min(n, len(incorrect))
    if actual_n < n:
        print(f"  [WARN] Only {actual_n} incorrect records available; sampling all {actual_n}.")

    # Sample
    sampled = stratified_sample(incorrect, actual_n, seed)

    # Print category breakdown
    breakdown: dict[str, int] = defaultdict(int)
    for r in sampled:
        breakdown[r["category"]] += 1
    print(f"  Sampled {len(sampled)} instances (stratified by category, seed={seed}):")
    for cat, count in sorted(breakdown.items()):
        print(f"    {cat:<35} {count}")

    # Write
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for record in sampled:
            writer.writerow(build_row(record))

    print(f"  Wrote {len(sampled)} rows → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one iaa_pilot_<budget>.csv per input JSONL file "
            "for independent per-budget annotation and comparison."
        )
    )
    parser.add_argument(
        "--inputs", "-i",
        nargs="+",
        type=Path,
        default=[Path("results/pilot_8k.jsonl")],
        help="One or more pilot JSONL files, one per budget level.",
    )
    parser.add_argument(
        "--out_dir", "-o",
        type=Path,
        default=Path("annotation"),
        help="Directory to write CSVs (default: annotation/)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=20,
        help="Instances to sample per budget (default: 20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    missing = [p for p in args.inputs if not p.exists()]
    if missing:
        for p in missing:
            print(f"[ERROR] File not found: {p}")
        sys.exit(1)

    outputs = []
    for path in args.inputs:
        out = process_file(path, args.n, args.seed, args.out_dir)
        if out:
            outputs.append(out)

    print(f"\n{'='*55}")
    print(f"Done. {len(outputs)} CSV(s) written:")
    for o in outputs:
        print(f"  {o}")
    print("\nNext steps:")
    print("  1. Annotators fill in each CSV independently.")
    print("  2. Run generate_adjudication_log.py once per CSV to compute κ per budget.")
    print("  3. Compare error distributions and κ scores across budgets.")


if __name__ == "__main__":
    main()
