"""Loss function for Prime-RL trainer."""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class LossInputs:
    trainer_logprobs: torch.Tensor
    inference_logprobs: torch.Tensor
    ref_logprobs: Optional[torch.Tensor]
    advantages: torch.Tensor
    loss_mask: torch.Tensor


@dataclass
class LossOutputs:
    loss: torch.Tensor
    policy_loss: torch.Tensor
    kl_penalty: torch.Tensor
    clip_fraction: float


def compute_grpo_loss(
    inputs: LossInputs,
    clip_eps: float = 0.2,
    kl_coeff: float = 0.04,
) -> LossOutputs:
    """Compute PPO clipped surrogate loss and Schulman k3 KL penalty on packed 1D token streams."""
    trainer_lp = inputs.trainer_logprobs
    inf_lp = inputs.inference_logprobs
    advs = inputs.advantages
    mask = inputs.loss_mask.bool()

    if not mask.any():
        zero = torch.tensor(0.0, device=trainer_lp.device, requires_grad=True)
        return LossOutputs(zero, zero, zero, 0.0)

    t_lp = trainer_lp[mask]
    i_lp = inf_lp[mask]
    a_vals = advs[mask]

    ratio = torch.exp(t_lp - i_lp)
    surr1 = ratio * a_vals
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * a_vals

    policy_loss = -torch.min(surr1, surr2).mean()

    # Schulman k3 KL estimate
    if inputs.ref_logprobs is not None:
        ref_lp = inputs.ref_logprobs[mask]
        log_r = ref_lp - t_lp
        kl = torch.exp(log_r) - log_r - 1.0
        kl_penalty = kl.mean()
    else:
        kl_penalty = torch.tensor(0.0, device=trainer_lp.device)

    total_loss = policy_loss + kl_coeff * kl_penalty

    clipped = (ratio < (1.0 - clip_eps)) | (ratio > (1.0 + clip_eps))
    clip_frac = float(clipped.float().mean().item())

    return LossOutputs(
        loss=total_loss,
        policy_loss=policy_loss,
        kl_penalty=kl_penalty,
        clip_fraction=clip_frac,
    )
