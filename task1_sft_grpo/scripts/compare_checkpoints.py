"""Compare checkpoints script to verify model weight updates and performance deltas."""

import argparse
import math

import torch
from rich.console import Console
from rich.table import Table
from transformers import AutoModelForCausalLM, AutoTokenizer

from sft_lab.collator import SFTDataCollator
from sft_lab.dataset import SFTDataset
from sft_lab.generation import generate_completions
from sft_lab.metrics import compute_manual_causal_lm_loss


def compare_checkpoints(base_dir: str, trained_dir: str, probe_data_path: str = "data/fixtures/valid.jsonl"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    console = Console()

    console.print("\n[bold blue]=== CHECKPOINT COMPARISON ===[/bold blue]")
    console.print(f"Base Model:    `{base_dir}`")
    console.print(f"Trained Model: `{trained_dir}`")

    tokenizer = AutoTokenizer.from_pretrained(base_dir, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(base_dir, trust_remote_code=True).to(device)
    trained_model = AutoModelForCausalLM.from_pretrained(trained_dir, trust_remote_code=True).to(device)

    base_model.eval()
    trained_model.eval()

    total_params = 0
    changed_params = 0
    total_l2_diff = 0.0

    module_diffs = {}

    for (name_b, p_b), (name_t, p_t) in zip(base_model.named_parameters(), trained_model.named_parameters()):
        assert name_b == name_t, f"Parameter mismatch: {name_b} vs {name_t}"
        diff = p_t.detach().float() - p_b.detach().float()
        l2 = torch.norm(diff, 2).item()
        numel = p_b.numel()

        total_params += numel
        total_l2_diff += l2**2

        param_changed = l2 > 1e-6
        if param_changed:
            changed_params += numel

        prefix = name_b.split(".")[0]
        if prefix not in module_diffs:
            module_diffs[prefix] = {"l2": 0.0, "count": 0}
        module_diffs[prefix]["l2"] += l2**2
        module_diffs[prefix]["count"] += numel

    global_l2 = total_l2_diff**0.5
    pct_changed = (changed_params / total_params * 100.0) if total_params > 0 else 0.0

    console.print(f"Total Parameters: {total_params:,}")
    console.print(f"Changed Parameters: {changed_params:,} ({pct_changed:.2f}%)")
    console.print(f"Global Parameter L2 Difference: [green]{global_l2:.6f}[/green]")

    table = Table(title="Per-Module L2 Differences")
    table.add_column("Module Prefix")
    table.add_column("Params Count", justify="right")
    table.add_column("L2 Difference", justify="right", style="cyan")

    for mod, data in module_diffs.items():
        table.add_row(mod, f"{data['count']:,}", f"{math.sqrt(data['l2']):.6f}")

    console.print(table)

    # Sanity check assertion: ensure parameters actually changed!
    assert pct_changed > 0.0 and global_l2 > 1e-5, (
        f"FAILURE: Trained checkpoint parameters are IDENTICAL to base model! (L2 diff: {global_l2})"
    )

    # Evaluate Probe Set CE Loss & Target Logprobs
    dataset = SFTDataset(probe_data_path, tokenizer, max_seq_length=256)
    collator = SFTDataCollator(tokenizer, max_seq_length=256)
    batch = collator([dataset[0]])

    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    with torch.no_grad():
        out_b = base_model(input_ids=input_ids, attention_mask=attention_mask)
        loss_b = compute_manual_causal_lm_loss(out_b.logits, labels).item()

        out_t = trained_model(input_ids=input_ids, attention_mask=attention_mask)
        loss_t = compute_manual_causal_lm_loss(out_t.logits, labels).item()

    console.print("\nProbe Set CE Loss:")
    console.print(f"  Base Model Loss:    {loss_b:.4f}")
    console.print(f"  Trained Model Loss: {loss_t:.4f}")
    console.print(f"  Loss Improvement:  {loss_b - loss_t:.4f}")

    # Prompt generation comparison
    probe_prompt = dataset[0]["messages"][0]["content"]
    gen_b = generate_completions(base_model, tokenizer, [probe_prompt], max_new_tokens=48, do_sample=False)[0]
    gen_t = generate_completions(trained_model, tokenizer, [probe_prompt], max_new_tokens=48, do_sample=False)[0]

    console.print(f"\nPrompt: {repr(probe_prompt)}")
    console.print(f"Base Generation:    {repr(gen_b)}")
    console.print(f"Trained Generation: {repr(gen_t)}")


def main():
    parser = argparse.ArgumentParser(description="Compare base vs trained model checkpoints.")
    parser.add_argument("--base_dir", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Base model ID or path")
    parser.add_argument(
        "--trained_dir", type=str, default="artifacts/checkpoints/sft_overfit", help="Trained model path"
    )
    parser.add_argument("--probe_data", type=str, default="data/fixtures/valid.jsonl", help="Probe dataset path")
    args = parser.parse_args()

    compare_checkpoints(args.base_dir, args.trained_dir, args.probe_data)


if __name__ == "__main__":
    main()
