"""Checkpoint saving, reloading, and metadata management."""

import json
import os
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from sft_lab.seed import generate_environment_report


def save_checkpoint(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    output_dir: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    config: Optional[Dict[str, Any]] = None,
    step: int = 0,
    epoch: int = 0,
) -> str:
    """Save full training checkpoint including weights, tokenizer, optimizer, scheduler, config, and env report."""
    os.makedirs(output_dir, exist_ok=True)

    # Save model & tokenizer HF format
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save training state dicts
    trainer_state = {
        "step": step,
        "epoch": epoch,
    }
    torch.save(trainer_state, os.path.join(output_dir, "trainer_state.pt"))

    if optimizer is not None:
        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))

    if scheduler is not None:
        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

    if config is not None:
        with open(os.path.join(output_dir, "sft_config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    env_report = generate_environment_report()
    with open(os.path.join(output_dir, "environment_report.json"), "w", encoding="utf-8") as f:
        json.dump(env_report, f, indent=2)

    return output_dir


def load_checkpoint(
    checkpoint_dir: str,
    device: str | None = None,
    load_optimizer: bool = False,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizer, Dict[str, Any]]:
    """Load checkpoint model, tokenizer, and trainer state."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(checkpoint_dir, trust_remote_code=True)
    model.to(device)

    trainer_state_path = os.path.join(checkpoint_dir, "trainer_state.pt")
    trainer_state = {}
    if os.path.exists(trainer_state_path):
        trainer_state = torch.load(trainer_state_path, map_location=device)

    if load_optimizer and optimizer is not None:
        opt_path = os.path.join(checkpoint_dir, "optimizer.pt")
        if os.path.exists(opt_path):
            optimizer.load_state_dict(torch.load(opt_path, map_location=device))

    return model, tokenizer, trainer_state
