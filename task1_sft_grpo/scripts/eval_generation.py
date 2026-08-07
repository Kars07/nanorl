"""Evaluate generation quality script."""

import argparse
import json
import os

import torch
from rich.console import Console
from transformers import AutoModelForCausalLM, AutoTokenizer

from sft_lab.dataset import SFTDataset
from sft_lab.generation import generate_completions


def evaluate_generation(model_dir: str, data_path: str = "data/processed/sft_eval.jsonl", num_samples: int = 20):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True).to(device)
    model.eval()

    dataset = SFTDataset(data_path, tokenizer, max_seq_length=256)
    records = dataset.records[:num_samples]

    prompts = [r["messages"][0]["content"] for r in records]
    targets = [r["messages"][1]["content"] for r in records]

    # Deterministic generation (greedy)
    greedy_comps = generate_completions(model, tokenizer, prompts, max_new_tokens=64, do_sample=False, device=device)

    # Stochastic generation (sampling)
    sample_comps = generate_completions(
        model,
        tokenizer,
        prompts,
        max_new_tokens=64,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        device=device,
    )

    im_end = "<|im_end|>"
    eos_id = tokenizer.eos_token_id

    def compute_stats(comps: list[str]):
        token_lens = []
        eos_count = 0
        exact_matches = 0
        for comp, tgt in zip(comps, targets):
            toks = tokenizer.encode(comp, add_special_tokens=False)
            token_lens.append(len(toks))
            # EOS termination check
            if len(toks) > 0 and toks[-1] == eos_id:
                eos_count += 1
            if comp.strip() == tgt.strip():
                exact_matches += 1

        avg_len = sum(token_lens) / len(token_lens) if token_lens else 0.0
        eos_rate = (eos_count / len(comps) * 100.0) if comps else 0.0
        exact_rate = (exact_matches / len(comps) * 100.0) if comps else 0.0

        return {
            "avg_generated_tokens": avg_len,
            "eos_termination_rate": eos_rate,
            "exact_match_rate": exact_rate,
        }

    greedy_stats = compute_stats(greedy_comps)
    sample_stats = compute_stats(sample_comps)

    console = Console()
    console.print("\n[bold blue]=== GENERATION EVALUATION REPORT ===[/bold blue]")
    console.print(f"Model Path: `{model_dir}`")
    console.print(f"Evaluated Samples: {len(records)}")

    console.print("\n[bold green]Deterministic (Greedy Decoding) Stats:[/bold green]")
    console.print(f"  Avg Generated Tokens: {greedy_stats['avg_generated_tokens']:.2f}")
    console.print(f"  EOS Termination Rate: {greedy_stats['eos_termination_rate']:.2f}%")
    console.print(f"  Exact Match Rate:     {greedy_stats['exact_match_rate']:.2f}%")

    console.print("\n[bold yellow]Stochastic (Sample Temp=0.7) Stats:[/bold yellow]")
    console.print(f"  Avg Generated Tokens: {sample_stats['avg_generated_tokens']:.2f}")
    console.print(f"  EOS Termination Rate: {sample_stats['eos_termination_rate']:.2f}%")
    console.print(f"  Exact Match Rate:     {sample_stats['exact_match_rate']:.2f}%")

    samples_log = []
    for i in range(min(5, len(records))):
        samples_log.append(
            {
                "prompt": prompts[i],
                "target": targets[i],
                "greedy_completion": greedy_comps[i],
                "sample_completion": sample_comps[i],
            }
        )

    report = {
        "model_dir": model_dir,
        "num_samples": len(records),
        "greedy_stats": greedy_stats,
        "sample_stats": sample_stats,
        "sample_outputs": samples_log,
    }

    os.makedirs("artifacts/reports", exist_ok=True)
    out_file = "artifacts/reports/eval_generation_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    console.print(f"\nReport saved to `{out_file}`")


def main():
    parser = argparse.ArgumentParser(description="Evaluate generation performance.")
    parser.add_argument("--model_dir", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Model path")
    parser.add_argument("--data_path", type=str, default="data/processed/sft_eval.jsonl", help="Eval dataset path")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of samples to evaluate")
    args = parser.parse_args()

    evaluate_generation(args.model_dir, args.data_path, args.num_samples)


if __name__ == "__main__":
    main()
