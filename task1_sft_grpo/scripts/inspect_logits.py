"""Inspect logits script to show per-token loss decomposition table."""

import argparse

import torch
from rich.console import Console
from rich.table import Table

from sft_lab.collator import SFTDataCollator
from sft_lab.dataset import SFTDataset
from sft_lab.metrics import compute_manual_causal_lm_loss, decompose_per_token_logits
from sft_lab.model import load_model_and_tokenizer


def main():
    parser = argparse.ArgumentParser(description="Inspect logits and per-token loss decomposition.")
    parser.add_argument("--data_path", type=str, default="data/fixtures/valid.jsonl", help="Dataset path")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Model ID")
    parser.add_argument("--example_index", type=int, default=0, help="Example index")
    parser.add_argument("--dtype", type=str, default="float32", help="Dtype")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model_and_tokenizer(model_id=args.model_id, dtype=args.dtype, device=device)
    model.eval()

    dataset = SFTDataset(args.data_path, tokenizer, max_seq_length=256)
    collator = SFTDataCollator(tokenizer, max_seq_length=256)
    batch = collator([dataset[args.example_index]])

    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = out.logits
        model_loss = out.loss.item() if out.loss is not None else None

    manual_loss = compute_manual_causal_lm_loss(logits, labels).item()

    rows = decompose_per_token_logits(logits, labels, input_ids, tokenizer, batch_idx=0)

    console = Console()
    console.print("\n[bold blue]=== CE LOSS COMPARISON ===[/bold blue]")
    console.print(f"Model Loss:  {model_loss}")
    console.print(f"Manual Loss: {manual_loss:.6f}")
    if model_loss is not None:
        console.print(f"Difference:  {abs(model_loss - manual_loss):.8f}")

    table = Table(title="Per-Token Logits & Loss Breakdown (Supervised Tokens)")
    table.add_column("Pos", justify="right")
    table.add_column("Input", style="yellow")
    table.add_column("Target", style="cyan")
    table.add_column("Tgt Logit", justify="right")
    table.add_column("Tgt Prob", justify="right")
    table.add_column("Tgt Logprob", justify="right")
    table.add_column("Top-1 Token", style="bold green")
    table.add_column("Top-5 Tokens")
    table.add_column("Entropy", justify="right")
    table.add_column("Token CE", justify="right", style="red")

    sup_rows = [r for r in rows if r["is_supervised"]]
    ign_rows = [r for r in rows if not r["is_supervised"]]

    for r in sup_rows:
        table.add_row(
            str(r["position"]),
            repr(r["input_token"]),
            repr(r["target_token"]),
            f"{r['target_logit']:.3f}",
            f"{r['target_prob']:.4f}",
            f"{r['target_logprob']:.4f}",
            repr(r["top1_token"]),
            r["top5_tokens"],
            f"{r['entropy']:.3f}",
            f"{r['token_ce']:.4f}",
        )

    console.print(table)

    console.print(f"\n[bold yellow]Sample Ignored Tokens (Total ignored: {len(ign_rows)}):[/bold yellow]")
    ign_table = Table(title="Ignored Tokens Sample (Do NOT contribute to CE loss)")
    ign_table.add_column("Pos", justify="right")
    ign_table.add_column("Input", style="yellow")
    ign_table.add_column("Top-1 Token", style="bold green")
    ign_table.add_column("Entropy", justify="right")

    for r in ign_rows[:5]:
        ign_table.add_row(
            str(r["position"]),
            repr(r["input_token"]),
            repr(r["top1_token"]),
            f"{r['entropy']:.3f}",
        )
    console.print(ign_table)


if __name__ == "__main__":
    main()
