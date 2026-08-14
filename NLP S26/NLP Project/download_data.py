from datasets import load_dataset
import json
from pathlib import Path

dataset = load_dataset("THUDM/LongBench-v2", split="train")

out_dir = Path("data/longbench_v2")
out_dir.mkdir(parents=True, exist_ok=True)

instances = []
for item in dataset:
    instances.append({
        "id":       item["_id"],
        "input":    item["context"],
        "question": item["question"],
        "choices":  [item["choice_A"], item["choice_B"], item["choice_C"], item["choice_D"]],
        "answer":   item["answer"],
        "category": item["domain"],
    })

with open(out_dir / "longbench_v2.json", "w", encoding="utf-8") as f:
    json.dump(instances, f, ensure_ascii=False, indent=2)

print(f"Saved {len(instances)} instances")