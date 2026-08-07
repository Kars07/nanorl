"""Chat-template microscope script."""

import argparse
import json

from rich.console import Console
from rich.table import Table
from transformers import AutoTokenizer

from sft_lab.masking import build_sft_labels_and_metadata, create_token_inspection_table


def main():
    parser = argparse.ArgumentParser(description="Inspect chat template rendering, token IDs, and loss mask.")
    parser.add_argument("--data_path", type=str, default="data/fixtures/valid.jsonl", help="Dataset path")
    parser.add_argument("--example_index", type=int, default=0, help="Index of example to inspect")
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
    messages = ex["messages"]

    processed = build_sft_labels_and_metadata(
        messages=messages,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        assistant_only_loss=True,
    )

    console = Console()
    console.print(f"\n[bold blue]=== RAW CONVERSATION (ID: {ex.get('id', args.example_index)}) ===[/bold blue]")
    console.print(json.dumps(messages, indent=2))

    console.print("\n[bold blue]=== RENDERED CHAT TEMPLATE STRING ===[/bold blue]")
    console.print(repr(processed["full_text"]))

    console.print(f"\n[bold blue]=== TOKEN IDS ({len(processed['input_ids'])} tokens) ===[/bold blue]")
    console.print(processed["input_ids"])

    table_data = create_token_inspection_table(
        input_ids=processed["input_ids"],
        labels=processed["labels"],
        roles=processed["roles"],
        tokenizer=tokenizer,
    )

    table = Table(title="Token-by-Token Inspection & Loss Mask")
    table.add_column("idx", justify="right", style="cyan")
    table.add_column("token", style="yellow")
    table.add_column("token_id", justify="right", style="magenta")
    table.add_column("label", justify="right", style="green")
    table.add_column("trained?", justify="center", style="bold")
    table.add_column("role", style="blue")

    trained_count = 0
    for row in table_data:
        trained_str = "[green]YES[/green]" if row["trained"] else "[red]NO[/red]"
        if row["trained"]:
            trained_count += 1
        table.add_row(
            str(row["idx"]),
            repr(row["token"]),
            str(row["token_id"]),
            str(row["label"]),
            trained_str,
            row["role"],
        )

    console.print(table)
    console.print(
        f"\n[bold green]Summary: Total tokens: {len(processed['input_ids'])}, Trained assistant tokens: {trained_count}[/bold green]"
    )

    # Round-trip diagnostic
    decoded_text = tokenizer.decode(processed["input_ids"])
    console.print("\n[bold blue]=== ROUND-TRIP DIAGNOSTICS ===[/bold blue]")
    console.print(f"Decoded text matches rendered template: {decoded_text == processed['full_text']}")


if __name__ == "__main__":
    main()
