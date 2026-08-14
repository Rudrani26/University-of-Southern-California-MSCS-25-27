import json
from typing import Any


class InferenceLogger:
    REQUIRED_KEYS = {
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

    def __init__(self, out_path: str):
        self.out_path = out_path
        self.file = open(out_path, "a", encoding="utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def log(self, record: dict[str, Any]):
        missing = self.REQUIRED_KEYS - set(record.keys())
        if missing:
            raise KeyError(f"Missing key(s): {sorted(missing)}")

        self.file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.file.flush()

    def close(self):
        if not self.file.closed:
            self.file.flush()
            self.file.close()