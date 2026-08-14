"""
generate_adjudication_log.py
-----------------------------
Reads the filled annotation/iaa_pilot.csv and generates
annotation/adjudication_log.md with one entry per disagreement.

Run AFTER both annotators have filled in their columns:
    python generate_adjudication_log.py \
        --input  annotation/iaa_pilot.csv \
        --output annotation/adjudication_log.md

What it does
------------
1. Reads iaa_pilot.csv.
2. Finds every row where annotator_a_label != annotator_b_label.
3. Writes a pre-formatted entry in adjudication_log.md for each disagreement,
   with a blank resolved_label and resolution fields for you to fill in.
4. If resolved_label is already filled in the CSV, it copies it into the log.
5. Prints a Cohen's kappa summary at the top of the log using the filled labels.
"""

import argparse
import csv
import sys
from pathlib import Path
from iaa import cohen_kappa, summarize_disagreements

VALID_LABELS = {"RF", "RsF", "INC"}


def load_csv(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def validate_rows(rows: list[dict]) -> list[dict]:
    required = {
        "instance_id", "category", "budget",
        "annotator_a_label", "annotator_b_label",
    }
    valid = []
    for i, row in enumerate(rows, start=2):  # start=2 because row 1 is header
        missing = required - row.keys()
        if missing:
            print(f"[WARN] Row {i} missing columns {missing} — skipping.")
            continue
        valid.append(row)
    return valid


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate adjudication_log.md from filled iaa_pilot.csv"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("annotation/iaa_pilot.csv"),
        help="Path to filled iaa_pilot.csv",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("annotation/adjudication_log.md"),
        help="Path to write adjudication_log.md",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"[ERROR] Input file not found: {args.input}")
        sys.exit(1)

    rows = load_csv(args.input)
    rows = validate_rows(rows)

    if not rows:
        print("[ERROR] No valid rows found in CSV.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Separate rows with complete labels vs incomplete
    # ------------------------------------------------------------------
    complete = [
        r for r in rows
        if r["annotator_a_label"] in VALID_LABELS
        and r["annotator_b_label"] in VALID_LABELS
    ]
    incomplete = [r for r in rows if r not in complete]

    if incomplete:
        print(f"[WARN] {len(incomplete)} row(s) have missing/invalid annotator labels — skipped from kappa.")

    if not complete:
        print("[ERROR] No rows with valid labels found. Fill in annotator columns first.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Cohen's kappa
    # ------------------------------------------------------------------
    labels_a = [r["annotator_a_label"] for r in complete]
    labels_b = [r["annotator_b_label"] for r in complete]
    kappa = cohen_kappa(labels_a, labels_b)

    # ------------------------------------------------------------------
    # Disagreements
    # ------------------------------------------------------------------
    ids = [r["instance_id"] for r in complete]
    disagreements = summarize_disagreements(labels_a, labels_b, ids)

    # Map instance_id -> full row for easy lookup
    row_by_id = {r["instance_id"]: r for r in complete}

    print(f"  Total annotated : {len(complete)}")
    print(f"  Agreements      : {len(complete) - len(disagreements)}")
    print(f"  Disagreements   : {len(disagreements)}")
    print(f"  Cohen's κ       : {kappa:.4f}")

    # ------------------------------------------------------------------
    # Write adjudication_log.md
    # ------------------------------------------------------------------
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:

        f.write("# Adjudication Log\n\n")
        f.write("**Task:** LongBench Diagnostic Error Annotation  \n")
        f.write(f"**Total instances annotated:** {len(complete)}  \n")
        f.write(f"**Agreements:** {len(complete) - len(disagreements)}  \n")
        f.write(f"**Disagreements:** {len(disagreements)}  \n")
        f.write(f"**Cohen's κ (before adjudication):** {kappa:.4f}  \n\n")
        f.write("---\n\n")

        if not disagreements:
            f.write("No disagreements found. All instances were in full agreement.\n")
        else:
            f.write("## Disagreement Entries\n\n")
            f.write(
                "For each entry below: review the instance, agree on a resolved label, "
                "and write a 2-sentence resolution note.\n\n"
            )
            f.write("---\n\n")

            for i, d in enumerate(disagreements, start=1):
                row = row_by_id[d["instance_id"]]
                resolved = row.get("resolved_label", "").strip()
                resolution_note = row.get("resolution_note", "").strip()

                f.write(f"### Disagreement {i}\n\n")
                f.write(f"**Instance ID:** `{d['instance_id']}`  \n")
                f.write(f"**Category:** {row.get('category', 'N/A')}  \n")
                f.write(f"**Budget:** {row.get('budget', 'N/A')} tokens  \n")
                f.write(f"**Model prediction:** {row.get('model_prediction', 'N/A')}  \n")
                f.write(f"**Ground truth:** {row.get('ground_truth', 'N/A')}  \n\n")
                f.write(f"| Annotator | Label | HALL |\n")
                f.write(f"|-----------|-------|------|\n")
                f.write(
                    f"| A | {row.get('annotator_a_label', '')} "
                    f"| {row.get('annotator_a_hall', '')} |\n"
                )
                f.write(
                    f"| B | {row.get('annotator_b_label', '')} "
                    f"| {row.get('annotator_b_hall', '')} |\n\n"
                )

                # If already resolved in the CSV, show it; otherwise leave blank
                if resolved in VALID_LABELS:
                    f.write(f"**Resolved label:** {resolved}  \n")
                else:
                    f.write(f"**Resolved label:** _(fill in: RF / RsF / INC)_  \n")

                if resolution_note:
                    f.write(f"**Resolution:** {resolution_note}  \n\n")
                else:
                    f.write(
                        "**Resolution:** _(write 2 sentences explaining why one label "
                        "was chosen and what made this instance ambiguous)_  \n\n"
                    )

                f.write("---\n\n")

    print(f"\nWrote adjudication log → {args.output}")
    if disagreements:
        print(f"  {len(disagreements)} entr{'y' if len(disagreements)==1 else 'ies'} to resolve.")
        print("  Fill in 'Resolved label' and 'Resolution' for each, then update iaa_pilot.csv.")


if __name__ == "__main__":
    main()
