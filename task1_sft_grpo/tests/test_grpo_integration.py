"""Integration tests for GRPO training loop and gradient flow."""

import torch

from sft_lab.config import GRPOConfig
from sft_lab.grpo.trainer import GRPOTrainer
from sft_lab.model import load_model_and_tokenizer


def test_grpo_end_to_end_single_step():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy_model, tokenizer = load_model_and_tokenizer("Qwen/Qwen2.5-0.5B-Instruct", dtype="bfloat16", device=device)
    ref_model, _ = load_model_and_tokenizer("Qwen/Qwen2.5-0.5B-Instruct", dtype="bfloat16", device=device)

    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=1e-6)
    config = GRPOConfig(num_generations=2, num_epochs=1, batch_size=1)

    trainer = GRPOTrainer(policy_model, ref_model, tokenizer, optimizer, config, device=device)

    prompts = [
        {"id": "p1", "prompt": "What is 2 + 3?", "target_answer": "5"},
    ]

    metrics = trainer.step(prompts)

    assert "loss" in metrics
    assert "reward_mean" in metrics
    assert not torch.isnan(torch.tensor(metrics["loss"]))
