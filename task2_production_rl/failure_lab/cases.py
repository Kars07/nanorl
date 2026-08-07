"""Production RL failure injection suite implementing 8 controlled failure modes."""

from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn


def inject_stale_inference_weights(trainer_weights: np.ndarray, lag_steps: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Failure Mode 1: Sampler retains old weights while trainer advances K optimizer steps."""
    sampler_weights = trainer_weights.copy()
    updated_trainer_weights = trainer_weights + (0.05 * lag_steps)
    return updated_trainer_weights, sampler_weights


def inject_wrong_tokenizer(text: str) -> Dict[str, Any]:
    """Failure Mode 2: Mismatched tokenizer encoding/decoding."""
    # Simulate tokenizer mismatch by altering special tokens
    tokens_raw = [ord(c) for c in text[:10]]
    tokens_mismatched = [ord(c) + 1 for c in text[:10]]
    return {
        "raw_tokens": tokens_raw,
        "mismatched_tokens": tokens_mismatched,
    }


def inject_different_chat_formatting(prompt: str) -> Dict[str, str]:
    """Failure Mode 3: Discrepancy between trainer and sampler chat template."""
    template_a = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    template_b = f"User: {prompt}\nAssistant: "
    return {
        "format_a": template_a,
        "format_b": template_b,
    }


def inject_different_token_sequence(tokens: torch.Tensor) -> torch.Tensor:
    """Failure Mode 4: Token sequence length or ID mismatch."""
    altered = tokens.clone()
    altered[0, -1] = (altered[0, -1] + 1) % 1000
    return altered


def inject_policy_lag_off_policy(current_version: int = 5, rollout_version: int = 1) -> Dict[str, int]:
    """Failure Mode 5: Rollout from policy N scored against trainer policy N+K."""
    lag = current_version - rollout_version
    return {
        "current_version": current_version,
        "rollout_version": rollout_version,
        "lag": lag,
    }


def inject_different_precision(logits_fp32: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Failure Mode 6: Precision mismatch (FP32 vs BF16 casting)."""
    logits_bf16 = logits_fp32.to(torch.bfloat16).to(torch.float32)
    return logits_fp32, logits_bf16


def inject_wrong_completion_mask(seq_len: int, prompt_len: int, pad_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Failure Mode 7: Completion mask incorrectly includes prompt or trailing pad tokens."""
    correct_mask = torch.zeros(seq_len, dtype=torch.long)
    correct_mask[prompt_len : seq_len - pad_len] = 1

    buggy_mask = torch.ones(seq_len, dtype=torch.long)  # includes prompt + pad!
    return correct_mask, buggy_mask


def inject_unsynced_sampler_weights(trainer_model: nn.Module, sampler_model: nn.Module) -> Tuple[float, float]:
    """Failure Mode 8: Weight update applied to trainer but not to sampler."""
    with torch.no_grad():
        for p in trainer_model.parameters():
            p.add_(0.1)

    fp_trainer = torch.norm(torch.cat([p.flatten() for p in trainer_model.parameters()])).item()
    fp_sampler = torch.norm(torch.cat([p.flatten() for p in sampler_model.parameters()])).item()
    return fp_trainer, fp_sampler
