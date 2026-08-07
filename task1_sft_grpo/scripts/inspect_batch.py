"""Inspect batch script to validate DataLoader batches and assertions."""

import argparse

from rich.console import Console
from rich.table import Table
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from sft_lab.collator import SFTDataCollator
from sft_lab.dataset import SFTDataset


def inspect_batch(batch: dict, tokenizer: AutoTokenizer, batch_idx: int = 0) -> None:
    console = Console()
    console.print(f"\n[bold blue]=== INSPECTING BATCH {batch_idx + 1} ===[/bold blue]")

    input_ids = batch["input_ids"]
    labels = batch["labels"]
    attention_mask = batch["attention_mask"]

    vocab_size = tokenizer.vocab_size

    # Assertions
    assert input_ids.shape == labels.shape == attention_mask.shape, (
        f"Shape mismatch: input_ids {input_ids.shape}, labels {labels.shape}, attention_mask {attention_mask.shape}"
    )

    for b in range(labels.shape[0]):
        for t in range(labels.shape[1]):
            lbl = labels[b, t].item()
            assert lbl == -100 or (0 <= lbl < vocab_size or lbl in tokenizer.all_special_ids), (
                f"Illegal label {lbl} at batch {b}, token {t}. Vocab size is {vocab_size}."
            )

    table = Table(title=f"Batch Tensors Info (Batch {batch_idx + 1})")
    table.add_column("Tensor Name")
    table.add_column("Dtype")
    table.add_column("Shape")
    table.add_column("Device")
    table.add_column("Min ID / Val")
    table.add_column("Max ID / Val")

    table.add_row(
        "input_ids",
        str(input_ids.dtype),
        str(tuple(input_ids.shape)),
        str(input_ids.device),
        str(input_ids.min().item()),
        str(input_ids.max().item()),
    )
    table.add_row(
        "labels",
        str(labels.dtype),
        str(tuple(labels.shape)),
        str(labels.device),
        str(labels.min().item()),
        str(labels.max().item()),
    )
    table.add_row(
        "attention_mask",
        str(attention_mask.dtype),
        str(tuple(attention_mask.shape)),
        str(attention_mask.device),
        str(attention_mask.min().item()),
        str(attention_mask.max().item()),
    )

    console.print(table)

    batch_size, seq_len = input_ids.shape
    total_non_pad = (attention_mask == 1).sum().item()
    total_supervised = (labels != -100).sum().item()

    console.print(f"Total non-padding tokens in batch: {total_non_pad}")
    console.print(f"Total supervised tokens in batch: {total_supervised}")

    ex_table = Table(title="Per-Example Batch Breakdown")
    ex_table.add_column("Ex Index", justify="right")
    ex_table.add_column("Seq Length", justify="right")
    ex_table.add_column("Non-Pad Tokens", justify="right")
    ex_table.add_column("Supervised Tokens", justify="right")

    for i in range(batch_size):
        seq_l = (attention_mask[i] == 1).sum().item()
        sup_l = (labels[i] != -100).sum().item()
        ex_table.add_row(str(i), str(seq_len), str(seq_l), str(sup_l))

    console.print(ex_table)
    console.print("[bold green]Batch inspection assertions PASSED successfully![/bold green]")


def main():
    parser = argparse.ArgumentParser(description="Inspect batch loader outputs.")
    parser.add_argument("--data_path", type=str, default="data/fixtures/valid.jsonl", help="Dataset path")
    parser.add_argument("--tokenizer_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Tokenizer ID")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--max_seq_length", type=int, default=256, help="Max sequence length")
    parser.add_argument("--num_batches", type=int, default=2, help="Number of batches to inspect")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True)
    dataset = SFTDataset(args.data_path, tokenizer, max_seq_length=args.max_seq_length)
    collator = SFTDataCollator(tokenizer, max_seq_length=args.max_seq_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    for i, batch in enumerate(loader):
        if i >= args.num_batches:
            break
        inspect_batch(batch, tokenizer, batch_idx=i)


if __name__ == "__main__":
    main()
