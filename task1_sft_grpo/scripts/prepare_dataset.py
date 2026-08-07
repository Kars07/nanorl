"""Prepare dataset script: Downloads GSM8K and formats it into standard conversation format."""

import json
import os

from datasets import load_dataset


def format_gsm8k_example(idx: int, example: dict, split: str) -> dict:
    """Format GSM8K example into standard conversation schema."""
    question = example["question"].strip()
    answer = example["answer"].strip()
    return {
        "id": f"gsm8k_{split}_{idx}",
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "source": "gsm8k",
        "category": "math_reasoning",
    }


def main():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    print("Loading GSM8K dataset...")
    ds = load_dataset("gsm8k", "main")

    train_raw = [ex for ex in ds["train"]]
    test_raw = [ex for ex in ds["test"]]

    with open("data/raw/gsm8k_train_raw.json", "w", encoding="utf-8") as f:
        json.dump(train_raw, f, indent=2)

    with open("data/raw/gsm8k_test_raw.json", "w", encoding="utf-8") as f:
        json.dump(test_raw, f, indent=2)

    train_data = [format_gsm8k_example(i, ex, "train") for i, ex in enumerate(train_raw)]
    test_data = [format_gsm8k_example(i, ex, "test") for i, ex in enumerate(test_raw)]

    # Write processed jsonl files
    with open("data/processed/sft_data.jsonl", "w", encoding="utf-8") as f:
        for item in train_data[:1000]:  # Save 1000 items for standard SFT training
            f.write(json.dumps(item) + "\n")

    with open("data/processed/sft_eval.jsonl", "w", encoding="utf-8") as f:
        for item in test_data[:200]:  # Save 200 items for evaluation
            f.write(json.dumps(item) + "\n")

    # Save tiny overfit subset (16 examples)
    overfit_data = train_data[:16]
    with open("data/processed/overfit_subset.jsonl", "w", encoding="utf-8") as f:
        for item in overfit_data:
            f.write(json.dumps(item) + "\n")

    # Save GRPO prompts subset
    grpo_prompts = []
    for ex in test_data[:100]:
        grpo_prompts.append(
            {
                "id": ex["id"],
                "prompt": ex["messages"][0]["content"],
                "target_answer": ex["messages"][1]["content"],
                "source": ex["source"],
                "category": ex["category"],
            }
        )
    with open("data/processed/grpo_prompts.jsonl", "w", encoding="utf-8") as f:
        for item in grpo_prompts:
            f.write(json.dumps(item) + "\n")

    print("Dataset preparation complete:")
    print(f"  Processed SFT train: {len(train_data[:1000])} samples -> data/processed/sft_data.jsonl")
    print(f"  Processed SFT eval: {len(test_data[:200])} samples -> data/processed/sft_eval.jsonl")
    print(f"  Processed overfit subset: {len(overfit_data)} samples -> data/processed/overfit_subset.jsonl")
    print(f"  Processed GRPO prompts: {len(grpo_prompts)} samples -> data/processed/grpo_prompts.jsonl")


if __name__ == "__main__":
    main()
