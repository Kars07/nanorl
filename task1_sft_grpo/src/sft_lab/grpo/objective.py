"""GRPO objective function and loss calculation with PPO clipping."""

from typing import Dict, Tuple

import torch

from sft_lab.grpo.kl import compute_kl_divergence


def compute_grpo_loss(
    current_logprobs: torch.Tensor,  # (B, S) - trainable
    old_logprobs: torch.Tensor,  # (B, S) - detached
    reference_logprobs: torch.Tensor,  # (B, S) - detached
    advantages: torch.Tensor,  # (B, 1) or (B, S) - detached
    completion_mask: torch.Tensor,  # (B, S) - 1 for completion, 0 for prompt/padding
    clip_eps: float = 0.2,
    kl_coeff: float = 0.04,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute GRPO clipped loss and diagnostics.

    Args:
        current_logprobs: (B, S) policy logprobs
        old_logprobs: (B, S) rollout policy logprobs
        reference_logprobs: (B, S) reference model logprobs
        advantages: (B, 1) or (B, S) advantages
        completion_mask: (B, S) mask of active completion tokens
        clip_eps: PPO clipping epsilon
        kl_coeff: KL penalty coefficient

    Returns:
        (total_loss, metrics_dict)
    """
    assert current_logprobs.shape == old_logprobs.shape == reference_logprobs.shape == completion_mask.shape

    # Policy ratio: exp(current - old)
    log_ratio = current_logprobs - old_logprobs
    ratio = torch.exp(log_ratio)

    if advantages.dim() == 2 and advantages.shape[1] == 1:
        adv = advantages.expand_as(current_logprobs)
    else:
        adv = advantages

    # Clipped surrogate objective
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    policy_loss_per_token = -torch.min(surr1, surr2)

    # KL penalty
    kl_per_token = compute_kl_divergence(current_logprobs, reference_logprobs)
    kl_loss_per_token = kl_coeff * kl_per_token

    # Total per-token loss
    total_loss_per_token = policy_loss_per_token + kl_loss_per_token

    # Masked loss over trainable completion tokens only
    active_tokens = completion_mask.sum().float()
    if active_tokens > 0:
        policy_loss = (policy_loss_per_token * completion_mask).sum() / active_tokens
        kl_loss = (kl_loss_per_token * completion_mask).sum() / active_tokens
        total_loss = (total_loss_per_token * completion_mask).sum() / active_tokens
    else:
        policy_loss = torch.tensor(0.0, device=current_logprobs.device)
        kl_loss = torch.tensor(0.0, device=current_logprobs.device)
        total_loss = torch.tensor(0.0, device=current_logprobs.device)

    # Metrics and diagnostics
    with torch.no_grad():
        clipped_tokens = (torch.abs(ratio - 1.0) > clip_eps) & (completion_mask == 1)
        clipping_fraction = (clipped_tokens.sum().float() / active_tokens).item() if active_tokens > 0 else 0.0

        mask_bool = completion_mask == 1
        if mask_bool.any():
            ratio_mean = ratio[mask_bool].mean().item()
            ratio_min = ratio[mask_bool].min().item()
            ratio_max = ratio[mask_bool].max().item()
            kl_mean = kl_per_token[mask_bool].mean().item()
        else:
            ratio_mean, ratio_min, ratio_max, kl_mean = 1.0, 1.0, 1.0, 0.0

    metrics = {
        "loss": total_loss.item(),
        "policy_loss": policy_loss.item(),
        "kl_loss": kl_loss.item(),
        "kl_mean": kl_mean,
        "ratio_mean": ratio_mean,
        "ratio_min": ratio_min,
        "ratio_max": ratio_max,
        "clipping_fraction": clipping_fraction,
    }

    return total_loss, metrics
