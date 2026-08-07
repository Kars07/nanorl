"""Inspect activations script using native PyTorch hooks."""

import argparse

import torch
from rich.console import Console
from rich.table import Table

from sft_lab.collator import SFTDataCollator
from sft_lab.dataset import SFTDataset
from sft_lab.hooks import ActivationTrackerHook
from sft_lab.model import load_model_and_tokenizer


def main():
    parser = argparse.ArgumentParser(description="Inspect forward pass activation statistics.")
    parser.add_argument("--data_path", type=str, default="data/fixtures/valid.jsonl", help="Dataset path")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Model ID")
    parser.add_argument("--dtype", type=str, default="float32", help="Dtype")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model_and_tokenizer(model_id=args.model_id, dtype=args.dtype, device=device)
    model.eval()

    tracker = ActivationTrackerHook(model)

    dataset = SFTDataset(args.data_path, tokenizer, max_seq_length=256)
    collator = SFTDataCollator(tokenizer, max_seq_length=256)
    batch = collator([dataset[0]])

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    with torch.no_grad():
        _ = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

    console = Console()
    console.print("\n[bold blue]=== ACTIVATION INSPECTION ===[/bold blue]")

    table = Table(title="Module Activation Statistics")
    table.add_column("Module Name")
    table.add_column("Shape")
    table.add_column("RMS", justify="right", style="cyan")
    table.add_column("Max Abs", justify="right", style="magenta")
    table.add_column("NaN Count", justify="center", style="bold red")
    table.add_column("Inf Count", justify="center", style="bold red")

    for mod_name, stats in tracker.stats.items():
        nan_style = "[bold red]" + str(stats["nan_count"]) + "[/bold red]" if stats["nan_count"] > 0 else "0"
        inf_style = "[bold red]" + str(stats["inf_count"]) + "[/bold red]" if stats["inf_count"] > 0 else "0"
        table.add_row(
            mod_name,
            str(stats["shape"]),
            f"{stats['rms']:.4f}",
            f"{stats['max_abs']:.4f}",
            nan_style,
            inf_style,
        )

    console.print(table)
    tracker.remove()


if __name__ == "__main__":
    main()
