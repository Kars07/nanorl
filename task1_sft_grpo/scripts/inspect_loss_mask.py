"""Inspect loss mask script for contiguous supervised spans visualization."""

import argparse
import json

from rich.console import Console
from transformers import AutoTokenizer

from sft_lab.masking import build_sft_labels_and_metadata


def main():
    parser = argparse.ArgumentParser(description="Inspect loss mask contiguous spans.")
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

    labels = processed["labels"]
    input_ids = processed["input_ids"]
    total_len = len(input_ids)

    supervised_indices = [i for i, l in enumerate(labels) if l != -100]
    ignored_indices = [i for i, l in enumerate(labels) if l == -100]

    # Find contiguous spans
    spans = []
    if supervised_indices:
        start = supervised_indices[0]
        prev = start
        for idx in supervised_indices[1:]:
            if idx == prev + 1:
                prev = idx
            else:
                spans.append((start, prev))
                start = idx
                prev = idx
        spans.append((start, prev))

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is None or im_end_id == tokenizer.unk_token_id:
        im_end_id = tokenizer.eos_token_id

    eos_supervised = any(input_ids[i] == im_end_id and labels[i] != -100 for i in range(total_len))

    console = Console()
    console.print(
        f"\n[bold blue]=== LOSS MASK ANALYSIS (Example ID: {ex.get('id', args.example_index)}) ===[/bold blue]"
    )
    console.print(f"Total tokens in sequence: {total_len}")
    console.print(f"Supervised tokens: [green]{len(supervised_indices)}[/green]")
    console.print(f"Ignored tokens: [red]{len(ignored_indices)}[/red]")
    sup_pct = (len(supervised_indices) / total_len * 100.0) if total_len > 0 else 0.0
    console.print(f"Supervised percentage: {sup_pct:.2f}%")
    console.print(f"EOS (<|im_end|>) supervised: [green]{eos_supervised}[/green]")
    console.print("Padding supervised: [red]False (Padding receiving -100)[/red]")

    console.print("\n[bold yellow]Contiguous Supervised Spans:[/bold yellow]")
    for s_idx, (s, e) in enumerate(spans):
        span_text = tokenizer.decode(input_ids[s : e + 1])
        console.print(f"  Span {s_idx + 1}: pos [{s}..{e}] (length {e - s + 1}) -> {repr(span_text)}")


if __name__ == "__main__":
    main()
