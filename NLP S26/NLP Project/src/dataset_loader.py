import json
import re
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = {"id", "input", "question", "choices", "answer", "category"}
VALID_ANSWERS = {"A", "B", "C", "D"}
EXPECTED_CATEGORIES = {
    "single-document QA",
    "multi-document QA",
    "long in-context learning",
    "dialogue history understanding",
    "code repository reasoning",
    "structured data understanding",
}

CATEGORY_ALIASES = {
    "single document qa": "single-document QA",
    "single document understanding": "single-document QA",
    "single document reasoning": "single-document QA",

    "multi document qa": "multi-document QA",
    "multi document understanding": "multi-document QA",
    "multi document reasoning": "multi-document QA",

    "in context learning": "long in-context learning",
    "long in context learning": "long in-context learning",
    "long in context understanding": "long in-context learning",
    "long in context reasoning": "long in-context learning",

    "dialogue history understanding": "dialogue history understanding",
    "long dialogue history understanding": "dialogue history understanding",

    "code repository reasoning": "code repository reasoning",
    "code repository understanding": "code repository reasoning",

    "structured data understanding": "structured data understanding",
    "structured data reasoning": "structured data understanding",
}


def _normalize_category(category: str) -> str:
    """
    Normalize dataset category names into canonical EXPECTED_CATEGORIES.
    Handles casing, punctuation, hyphens, and common 'Long ...' variants.
    """
    normalized = re.sub(r"[^a-z0-9]+", " ", category.lower()).strip()
    normalized = re.sub(r"\s+", " ", normalized)

    candidates = [normalized]
    if normalized.startswith("long "):
        candidates.append(normalized[5:].strip())

    for cand in candidates:
        if cand in CATEGORY_ALIASES:
            return CATEGORY_ALIASES[cand]

    raise ValueError(
        f"Invalid category '{category}'. "
        f"Expected one of: {sorted(EXPECTED_CATEGORIES)}"
    )


def load(data_dir: str, strict: bool = True) -> list[dict]:
    """
    Reads all .json files under data_dir (non-recursive).

    Each instance must have:
      - id
      - input
      - question
      - choices (list of 4 strings)
      - answer (A/B/C/D)
      - category (one of the expected categories)

    Returns a flat list of validated dictionaries.
    Raises ValueError on any missing or malformed field.

    Args:
        data_dir: Path to directory containing LongBench v2 JSON files.
        strict:   If True (default), raises ValueError if any expected category
                  is missing from the loaded dataset. Set to False when testing
                  on partial subsets of the data.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")
    if not data_path.is_dir():
        raise NotADirectoryError(f"Expected a directory, got: {data_path}")

    json_files = sorted(data_path.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No .json files found in {data_path}")

    instances: list[dict] = []

    for json_file in json_files:
        with json_file.open("r", encoding="utf-8") as f:
            try:
                raw = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in {json_file}: {e}") from e

        # Support either a single object or a list of objects
        if isinstance(raw, dict):
            raw = [raw]
        if not isinstance(raw, list):
            raise ValueError(f"Unexpected top-level type in {json_file}: {type(raw)}")

        for idx, item in enumerate(raw):
            validated = _validate(item, source=f"{json_file.name}[{idx}]")
            instances.append(validated)

    if strict:
        _validate_dataset(instances)

    return instances


def _validate(item: Any, source: str) -> dict:
    """
    Validate one instance and return a cleaned copy.
    Raises ValueError if any required field is missing or malformed.
    """
    if not isinstance(item, dict):
        raise ValueError(f"{source}: instance is not a dict — got {type(item)}")

    for field in REQUIRED_FIELDS:
        if field not in item:
            raise ValueError(f"{source}: missing required field '{field}'")
        if item[field] is None:
            raise ValueError(f"{source}: field '{field}' is None")

    # id, input, question, category must be non-empty strings
    for field in ("id", "input", "question", "category"):
        if not isinstance(item[field], str) or not item[field].strip():
            raise ValueError(f"{source}: '{field}' must be a non-empty string")

    # Normalize category
    category = _normalize_category(item["category"])

    # choices must be a list of exactly 4 non-empty strings
    choices = item["choices"]
    if not isinstance(choices, list) or len(choices) != 4:
        raise ValueError(
            f"{source}: 'choices' must be a list of exactly 4 strings, "
            f"got {type(choices)} with length "
            f"{len(choices) if isinstance(choices, list) else 'N/A'}"
        )
    for i, choice in enumerate(choices):
        if not isinstance(choice, str) or not choice.strip():
            raise ValueError(f"{source}: choices[{i}] is not a non-empty string")

    # Normalize answer
    answer = item["answer"]
    if not isinstance(answer, str):
        raise ValueError(f"{source}: 'answer' must be a string, got {type(answer)}")
    answer = answer.strip().upper()
    if answer not in VALID_ANSWERS:
        raise ValueError(
            f"{source}: 'answer' must be one of {sorted(VALID_ANSWERS)}, got {answer!r}"
        )

    # Return a cleaned shallow copy with only expected keys
    return {
        "id": item["id"].strip(),
        "input": item["input"],
        "question": item["question"],
        "choices": choices[:],
        "answer": answer,
        "category": category,
    }


def _validate_dataset(instances: list[dict]) -> None:
    """
    Dataset-level sanity checks.
    Raises ValueError if the benchmark looks incomplete or malformed.
    Only called when strict=True.
    """
    if not instances:
        raise ValueError("Dataset is empty")

    missing_categories = EXPECTED_CATEGORIES - {inst["category"] for inst in instances}
    if missing_categories:
        raise ValueError(f"Missing expected categories: {sorted(missing_categories)}")


def get_by_category(instances: list[dict]) -> dict[str, list[dict]]:
    """
    Groups a flat list of instances by their 'category' key.
    Returns a dict mapping category name -> list of instances.
    Always includes all expected categories as keys.
    """
    grouped: dict[str, list[dict]] = {cat: [] for cat in EXPECTED_CATEGORIES}

    for inst in instances:
        if "category" not in inst:
            raise ValueError("Instance missing 'category' key")

        cat = inst["category"]
        if cat not in EXPECTED_CATEGORIES:
            raise ValueError(f"Unexpected category: {cat}")

        grouped[cat].append(inst)

    return grouped

if __name__ == "__main__":
    import sys

    data_dir = "data/longbench_v2"
    if "--data_dir" in sys.argv:
        idx = sys.argv.index("--data_dir")
        if idx + 1 >= len(sys.argv):
            raise ValueError("--data_dir was provided but no path followed it")
        data_dir = sys.argv[idx + 1]

    print("=" * 60)
    print("dataset_loader.py — sanity check")
    print("=" * 60)
    print(f"Loading from {data_dir} ...")

    instances = load(data_dir, strict=True)
    print(f"Loaded {len(instances)} instances")

    grouped = get_by_category(instances)
    print("Category counts:")
    for cat in sorted(grouped.keys()):
        print(f"  {cat}: {len(grouped[cat])}")

    print("\nFirst instance preview:")
    first = instances[0]
    print(f"  id: {first['id']}")
    print(f"  category: {first['category']}")
    print(f"  answer: {first['answer']}")
    print(f"  question: {first['question'][:100]}...")