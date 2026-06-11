"""Score LLaMA-Factory generated predictions for the emotion test split.

Reads the ``generated_predictions.jsonl`` written by
``llamafactory-cli train config/emotion_llama3_predict.yaml`` (one JSON
object per line with ``label`` and ``predict`` fields) and reports overall
accuracy plus a per-class breakdown.

Usage:
    python scripts/evaluate.py saves/llama3-8b-emotion-lora/predict/generated_predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", help="Path to generated_predictions.jsonl")
    args = parser.parse_args()

    total = 0
    correct = 0
    per_class = defaultdict(lambda: [0, 0])  # label -> [correct, total]

    with open(args.predictions, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            label = record["label"].strip().lower()
            predict = record["predict"].strip().lower()
            hit = int(label == predict)
            total += 1
            correct += hit
            per_class[label][0] += hit
            per_class[label][1] += 1

    if total == 0:
        raise SystemExit(f"No records found in {args.predictions}")

    print(f"samples: {total}")
    print(f"accuracy: {correct / total:.4f}\n")
    print(f"{'label':<10} {'accuracy':>8} {'n':>6}")
    for label in sorted(per_class):
        hits, n = per_class[label]
        print(f"{label:<10} {hits / n:>8.4f} {n:>6}")


if __name__ == "__main__":
    main()
