"""Unit tests for GRPO logprob computation and gradient properties."""

import torch

from sft_lab.grpo.logprobs import compute_reference_logprobs, get_per_token_logprobs
from sft_lab.model import load_model_and_tokenizer


def test_logprobs_gather_parity():
    """Compare get_per_token_logprobs with manual log_softmax gather."""
    logits = torch.randn(2, 5, 100, requires_grad=True)
    input_ids = torch.randint(0, 100, (2, 5))

    lp = get_per_token_logprobs(logits, input_ids)

    # Manual gather
    shift_logits = logits[:, :-1, :].float()
    shift_ids = input_ids[:, 1:]
    manual_lp = torch.log_softmax(shift_logits, dim=-1).gather(-1, shift_ids.unsqueeze(-1)).squeeze(-1)

    assert torch.allclose(lp, manual_lp, atol=1e-6)


def test_logprobs_gradient_detach_properties():
    """Verify old/ref logprobs have no gradient while current logprobs do."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model_and_tokenizer("Qwen/Qwen2.5-0.5B-Instruct", dtype="float32", device=device)

    input_ids = torch.tensor([[151644, 872, 198, 9707, 151645]], device=device)
    attn_mask = torch.ones_like(input_ids)

    ref_lp = compute_reference_logprobs(model, input_ids, attn_mask)
    assert not ref_lp.requires_grad, "Reference logprobs must have requires_grad=False"

    model.train()
    out = model(input_ids=input_ids, attention_mask=attn_mask)
    cur_lp = get_per_token_logprobs(out.logits, input_ids)

    assert cur_lp.requires_grad, "Current policy logprobs must require gradients"
