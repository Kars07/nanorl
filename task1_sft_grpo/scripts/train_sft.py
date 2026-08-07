"""Production SFT Trainer script."""

import argparse
import json
import math
import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from sft_lab.checkpointing import load_checkpoint, save_checkpoint
from sft_lab.collator import SFTDataCollator
from sft_lab.config import SFTConfig
from sft_lab.dataset import SFTDataset
from sft_lab.generation import generate_completions
from sft_lab.hooks import ActivationTrackerHook, compute_gradient_stats
from sft_lab.metrics import compute_manual_causal_lm_loss
from sft_lab.model import load_model_and_tokenizer
from sft_lab.seed import set_seed


def train_sft(config: SFTConfig, resume_from_checkpoint: str | None = None):
    set_seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model {config.model_id} on {device}...")
    model, tokenizer = load_model_and_tokenizer(
        model_id=config.model_id,
        tokenizer_id=config.tokenizer_id,
        dtype=config.dtype,
        device=device,
    )

    # Attach activation tracker
    activation_tracker = ActivationTrackerHook(model)

    train_dataset = SFTDataset(
        config.dataset_name_or_path,
        tokenizer,
        max_seq_length=config.max_seq_length,
        assistant_only_loss=config.assistant_only_loss,
    )
    eval_dataset = SFTDataset(
        "data/processed/sft_eval.jsonl",
        tokenizer,
        max_seq_length=config.max_seq_length,
        assistant_only_loss=config.assistant_only_loss,
    )

    collator = SFTDataCollator(
        tokenizer,
        max_seq_length=config.max_seq_length,
        assistant_only_loss=config.assistant_only_loss,
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collator)
    eval_loader = DataLoader(eval_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collator)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    total_update_steps = math.ceil(len(train_loader) / config.gradient_accumulation_steps) * config.num_epochs
    warmup_steps = int(total_update_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_update_steps
    )

    start_epoch = 0
    global_step = 0

    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        print(f"Resuming training from checkpoint: {resume_from_checkpoint}")
        model, tokenizer, state = load_checkpoint(
            resume_from_checkpoint, device=device, load_optimizer=True, optimizer=optimizer
        )
        start_epoch = state.get("epoch", 0)
        global_step = state.get("step", 0)

    # Autocast setup
    use_autocast = (device == "cuda") and (config.dtype in ["float16", "bfloat16"])
    autocast_dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float16

    metrics_log = []
    plot_data = {"step": [], "loss": [], "grad_norm": [], "eval_loss": []}

    print(f"Starting SFT Training for {config.num_epochs} epochs ({total_update_steps} update steps)...")

    # Probe prompts for periodic generation logging
    probe_prompts = [
        "What is 15 + 27?",
        "Natalia sold clips to 48 of her friends in April...",
    ]

    for epoch in range(start_epoch, config.num_epochs):
        model.train()
        optimizer.zero_grad()

        accumulated_loss = 0.0

        for step_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attn_mask = batch["attention_mask"].to(device)

            if use_autocast:
                with torch.amp.autocast("cuda", dtype=autocast_dtype):
                    out = model(input_ids=input_ids, attention_mask=attn_mask)
                    loss = compute_manual_causal_lm_loss(out.logits, labels)
                    loss = loss / config.gradient_accumulation_steps
            else:
                out = model(input_ids=input_ids, attention_mask=attn_mask)
                loss = compute_manual_causal_lm_loss(out.logits, labels)
                loss = loss / config.gradient_accumulation_steps

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[ERROR] NaN or Inf loss detected at epoch {epoch}, step {step_idx}!")
                raise ValueError("NaN/Inf loss encountered during training!")

            loss.backward()
            accumulated_loss += loss.item() * config.gradient_accumulation_steps

            if (step_idx + 1) % config.gradient_accumulation_steps == 0 or (step_idx + 1) == len(train_loader):
                # Compute gradient statistics before clipping
                grad_stats = compute_gradient_stats(model)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip).item()

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1

                step_metric = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "loss": accumulated_loss,
                    "grad_norm": grad_norm,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "activations": activation_tracker.stats.copy(),
                }
                metrics_log.append(step_metric)

                plot_data["step"].append(global_step)
                plot_data["loss"].append(accumulated_loss)
                plot_data["grad_norm"].append(grad_norm)

                if global_step % 10 == 0 or global_step == 1:
                    print(
                        f"Step {global_step:3d} | Epoch {epoch + 1:2d} | Loss: {accumulated_loss:.4f} | GradNorm: {grad_norm:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}"
                    )

                accumulated_loss = 0.0

        # Run validation split eval
        model.eval()
        total_eval_loss = 0.0
        with torch.no_grad():
            for eval_batch in eval_loader:
                e_ids = eval_batch["input_ids"].to(device)
                e_lbls = eval_batch["labels"].to(device)
                e_mask = eval_batch["attention_mask"].to(device)

                if use_autocast:
                    with torch.amp.autocast("cuda", dtype=autocast_dtype):
                        e_out = model(input_ids=e_ids, attention_mask=e_mask)
                        e_loss = compute_manual_causal_lm_loss(e_out.logits, e_lbls)
                else:
                    e_out = model(input_ids=e_ids, attention_mask=e_mask)
                    e_loss = compute_manual_causal_lm_loss(e_out.logits, e_lbls)

                total_eval_loss += e_loss.item()

        avg_eval_loss = total_eval_loss / len(eval_loader)
        plot_data["eval_loss"].append((global_step, avg_eval_loss))
        print(f"--- Epoch {epoch + 1} Evaluation Loss: {avg_eval_loss:.4f} ---")

        # Periodic generation probe
        probes = generate_completions(model, tokenizer, probe_prompts, max_new_tokens=48, do_sample=False)
        print(f"Probe 1 output: {repr(probes[0])}")

    activation_tracker.remove()

    # Save final checkpoint
    save_checkpoint(
        model,
        tokenizer,
        config.output_dir,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config.model_dump(),
        step=global_step,
        epoch=config.num_epochs,
    )
    print(f"\nTraining completed! Checkpoint saved to {config.output_dir}")

    # Generate metric plots
    os.makedirs("artifacts/plots", exist_ok=True)
    os.makedirs("artifacts/logs", exist_ok=True)

    with open("artifacts/logs/sft_training_metrics.jsonl", "w", encoding="utf-8") as f:
        for entry in metrics_log:
            f.write(json.dumps(entry) + "\n")

    plt.figure(figsize=(10, 5))
    plt.plot(plot_data["step"], plot_data["loss"], label="Train Loss")
    if plot_data["eval_loss"]:
        eval_steps, eval_losses = zip(*plot_data["eval_loss"])
        plt.plot(eval_steps, eval_losses, "ro-", label="Eval Loss")
    plt.xlabel("Global Step")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SFT Training Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig("artifacts/plots/sft_loss_curve.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(plot_data["step"], plot_data["grad_norm"], color="orange", label="Gradient Norm")
    plt.xlabel("Global Step")
    plt.ylabel("Grad Norm")
    plt.title("SFT Gradient Norm Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig("artifacts/plots/sft_grad_norm.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train SFT model.")
    parser.add_argument("--config", type=str, default="configs/sft_debug.yaml", help="Path to config yaml")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    args = parser.parse_args()

    config = SFTConfig.from_yaml(args.config)
    train_sft(config, resume_from_checkpoint=args.resume)


if __name__ == "__main__":
    main()
