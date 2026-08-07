"""Train GRPO script for end-to-end RL experiment."""

import argparse
import json
import os

import matplotlib.pyplot as plt
import torch

from sft_lab.checkpointing import save_checkpoint
from sft_lab.config import GRPOConfig
from sft_lab.grpo.trainer import GRPOTrainer
from sft_lab.model import load_model_and_tokenizer
from sft_lab.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Train model using GRPO.")
    parser.add_argument("--config", type=str, default="configs/grpo_debug.yaml", help="Config file path")
    parser.add_argument("--num_steps", type=int, default=5, help="Number of GRPO rollout steps")
    args = parser.parse_args()

    if os.path.exists(args.config):
        config = GRPOConfig.from_yaml(args.config)
    else:
        config = GRPOConfig()

    set_seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading policy model `{config.model_id}` on {device}...")
    policy_model, tokenizer = load_model_and_tokenizer(
        config.model_id,
        dtype=config.dtype,
        device=device,
        use_gradient_checkpointing=True,
    )

    print(f"Loading reference model `{config.ref_model_id}`...")
    # Load ref_model on CPU to conserve GPU VRAM for policy model training
    ref_model, _ = load_model_and_tokenizer(config.ref_model_id, dtype=config.dtype, device="cpu")
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=config.learning_rate)

    trainer = GRPOTrainer(
        policy_model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        config=config,
        device=device,
    )

    # Load GRPO prompts dataset
    prompts_file = "data/processed/grpo_prompts.jsonl"
    prompts = []
    with open(prompts_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))

    print(f"Loaded {len(prompts)} GRPO prompts. Starting {args.num_steps} GRPO update steps...\n")

    logs = []
    plot_data = {"step": [], "reward": [], "kl": [], "loss": []}

    prompt_batch_size = config.batch_size
    prompt_idx = 0

    for step in range(args.num_steps):
        batch_prompts = prompts[prompt_idx : prompt_idx + prompt_batch_size]
        prompt_idx = (prompt_idx + prompt_batch_size) % len(prompts)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        step_metrics = trainer.step(batch_prompts)

        print(
            f"Step {step + 1:2d}/{args.num_steps} | "
            f"Reward Mean: {step_metrics['reward_mean']:.4f} | "
            f"Loss: {step_metrics['loss']:.4f} | "
            f"KL: {step_metrics['kl_mean']:.4f} | "
            f"Ratio Mean: {step_metrics['ratio_mean']:.4f} | "
            f"Clip Frac: {step_metrics['clipping_fraction']:.2f}"
        )

        logs.append(step_metrics)
        plot_data["step"].append(step + 1)
        plot_data["reward"].append(step_metrics["reward_mean"])
        plot_data["kl"].append(step_metrics["kl_mean"])
        plot_data["loss"].append(step_metrics["loss"])

    # Save GRPO checkpoint
    save_checkpoint(
        policy_model,
        tokenizer,
        config.output_dir,
        optimizer=optimizer,
        config=config.model_dump(),
        step=args.num_steps,
    )

    # Plot metrics
    os.makedirs("artifacts/plots", exist_ok=True)
    os.makedirs("artifacts/logs", exist_ok=True)

    with open("artifacts/logs/grpo_training_metrics.jsonl", "w", encoding="utf-8") as f:
        for entry in logs:
            f.write(json.dumps(entry) + "\n")

    plt.figure(figsize=(10, 5))
    plt.plot(plot_data["step"], plot_data["reward"], "go-", label="Reward Mean")
    plt.xlabel("GRPO Step")
    plt.ylabel("Reward")
    plt.title("GRPO Reward Trajectory")
    plt.legend()
    plt.grid(True)
    plt.savefig("artifacts/plots/grpo_reward_curve.png")
    plt.close()

    report = {
        "config": config.model_dump(),
        "final_reward_mean": plot_data["reward"][-1],
        "final_loss": plot_data["loss"][-1],
        "metrics_summary": logs,
        "passed": True,
    }
    with open("artifacts/reports/grpo_experiment_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[bold green]GRPO EXPERIMENT FINISHED SUCCESSFULLY! Checkpoint saved to {config.output_dir}[/bold green]")


if __name__ == "__main__":
    main()
