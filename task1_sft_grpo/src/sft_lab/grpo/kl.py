"""KL divergence penalty estimators for GRPO."""

import torch


def compute_kl_divergence(
    current_logprob: torch.Tensor,
    reference_logprob: torch.Tensor,
    estimator: str = "schulman",
) -> torch.Tensor:
    """Compute per-token KL divergence penalty.

    Args:
        current_logprob: (B, S) policy model token log-probabilities (trainable)
        reference_logprob: (B, S) reference model token log-probabilities (detached)
        estimator: 'schulman' for k3 unbiased estimator (exp(ref - cur) - (ref - cur) - 1)

    Returns:
        (B, S) Tensor of non-negative per-token KL divergence estimates
    """
    delta = reference_logprob - current_logprob
    if estimator == "schulman":
        # k3 estimator: exp(r_ref - r_cur) - (r_ref - r_cur) - 1
        kl = torch.exp(delta) - delta - 1.0
    else:
        # Standard reverse KL difference: r_cur - r_ref
        kl = current_logprob - reference_logprob

    return kl
