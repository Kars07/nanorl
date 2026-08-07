"""Inspect tokens script for detailed per-token breakdown."""

import argparse
import json

from rich.console import Console
from rich.table import Table
from transformers import AutoTokenizer

from sft_lab.masking import build_sft_labels_and_metadata


def main():
    parser = argparse.ArgumentParser(description="Inspect token details for selected example.")
    parser.add_argument("--data_path", type=str, default="data/fixtures/valid.jsonl", help="Dataset path")
    parser.add_argument("--example_index", type=int, default=0, help="Index of example")
    parser.add_argument("--tokenizer_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Tokenizer ID")
    parser.add_argument("--max_seq_length", type=int, default=256, help="Max sequence length")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True)

    records = []
    with open(args.data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    ex = records[args.example_index]
    processed = build_sft_labels_and_metadata(
        messages=ex["messages"],
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        assistant_only_loss=True,
    )

    console = Console()
    table = Table(title=f"Token Details Breakdown (ID: {ex.get('id', args.example_index)})")
    table.add_column("Pos", justify="right")
    table.add_column("Token", style="yellow")
    table.add_column("ID", justify="right")
    table.add_column("Role", style="blue")
    table.add_column("Special?", justify="center")
    table.add_column("AttnMask", justify="center")
    table.add_column("Label", justify="right")
    table.add_column("Train?", justify="center")

    for i in range(len(processed["input_ids"])):
        tid = processed["input_ids"][i]
        lbl = processed["labels"][i]
        role = processed["roles"][i]
        attn = processed["attention_mask"][i]
        tok_str = tokenizer.decode([tid])
        is_special = "YES" if tid in tokenizer.all_special_ids else "NO"
        is_train = "[green]TRAIN[/green]" if lbl != -100 else "[red]IGNORE[/red]"

        table.add_row(
            str(i),
            repr(tok_str),
            str(tid),
            role,
            is_special,
            str(attn),
            str(lbl),
            is_train,
        )

    console.print(table)


if __name__ == "__main__":
    main()
