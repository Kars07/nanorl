"""Inspect gradients script to compute backward pass and show gradient norms."""

import argparse

import torch
from rich.console import Console
from rich.table import Table

from sft_lab.collator import SFTDataCollator
from sft_lab.dataset import SFTDataset
from sft_lab.hooks import compute_gradient_stats
from sft_lab.metrics import compute_manual_causal_lm_loss
from sft_lab.model import load_model_and_tokenizer


def main():
    parser = argparse.ArgumentParser(description="Inspect gradients after backward pass.")
    parser.add_argument("--data_path", type=str, default="data/fixtures/valid.jsonl", help="Dataset path")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Model ID")
    parser.add_argument("--dtype", type=str, default="float32", help="Dtype")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model_and_tokenizer(model_id=args.model_id, dtype=args.dtype, device=device)
    model.train()

    dataset = SFTDataset(args.data_path, tokenizer, max_seq_length=256)
    collator = SFTDataCollator(tokenizer, max_seq_length=256)
    batch = collator([dataset[0]])

    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    model.zero_grad()
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    loss = compute_manual_causal_lm_loss(out.logits, labels)
    loss.backward()

    grad_stats = compute_gradient_stats(model)

    console = Console()
    console.print("\n[bold blue]=== GRADIENT INSPECTION ===[/bold blue]")
    console.print(f"Global Gradient Norm (L2): [green]{grad_stats['global_grad_norm']:.6f}[/green]")

    table = Table(title="Top Parameter Gradient Norms & Max Values")
    table.add_column("Parameter Name")
    table.add_column("Grad Norm (L2)", justify="right", style="cyan")
    table.add_column("Grad Max Abs", justify="right", style="magenta")

    param_grads = grad_stats["per_parameter_grads"]
    sorted_params = sorted(param_grads.items(), key=lambda x: x[1]["norm"], reverse=True)

    for name, stats in sorted_params[:15]:
        table.add_row(name, f"{stats['norm']:.6f}", f"{stats['max_abs']:.6f}")

    console.print(table)


if __name__ == "__main__":
    main()
