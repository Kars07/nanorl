"""Per-token log-probability computation utilities for GRPO."""

import torch


def get_per_token_logprobs(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Extract log probabilities for selected sequence tokens memory-efficiently.

    Args:
        logits: (B, S, V) model output logits
        input_ids: (B, S) token IDs

    Returns:
        (B, S - 1) Tensor of log probabilities for input_ids[:, 1:]
    """
    assert logits.dim() == 3, f"Logits shape must be 3D (B, S, V), got {logits.shape}"
    assert input_ids.dim() == 2, f"Input IDs shape must be 2D (B, S), got {input_ids.shape}"
    assert logits.shape[:2] == input_ids.shape, (
        f"Mismatch between logits {logits.shape[:2]} and input_ids {input_ids.shape}"
    )

    shift_logits = logits[:, :-1, :].contiguous()
    shift_tokens = input_ids[:, 1:].contiguous()

    # Memory-efficient logprob: logprob(target) = target_logit - logsumexp(logits)
    # Avoids allocating (B, S, V) log_softmax tensor.
    target_logits = shift_logits.gather(dim=-1, index=shift_tokens.unsqueeze(-1)).squeeze(-1).float()
    log_sum_exp = torch.logsumexp(shift_logits.float(), dim=-1)

    per_token_logprobs = target_logits - log_sum_exp
    return per_token_logprobs


def compute_reference_logprobs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute reference or old-policy logprobs with zero gradient graph."""
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logprobs = get_per_token_logprobs(out.logits, input_ids)

    assert not logprobs.requires_grad, "Reference/old logprobs must not require gradients!"
    return logprobs.detach()
