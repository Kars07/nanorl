"""Unit tests for GRPO policy ratio behavior."""

import torch


def test_ratio_identity():
    logprob = torch.tensor([-1.2, -0.5, -2.1])
    ratio = torch.exp(logprob - logprob)
    assert torch.allclose(ratio, torch.ones_like(ratio)), "Ratio when current == old must be 1.0"


def test_ratio_increase_decrease():
    old_lp = torch.tensor([-1.0, -1.0])
    cur_lp = torch.tensor([-0.5, -1.5])  # position 0 increased logprob, position 1 decreased

    ratio = torch.exp(cur_lp - old_lp)
    assert ratio[0].item() > 1.0, "Increased current logprob must yield ratio > 1"
    assert ratio[1].item() < 1.0, "Decreased current logprob must yield ratio < 1"
