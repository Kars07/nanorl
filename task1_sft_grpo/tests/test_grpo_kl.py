"""Unit tests for KL divergence estimator."""

import torch

from sft_lab.grpo.kl import compute_kl_divergence


def test_kl_zero_when_identical():
    cur_lp = torch.tensor([-1.5, -0.2, -3.0])
    ref_lp = cur_lp.clone()

    kl = compute_kl_divergence(cur_lp, ref_lp, estimator="schulman")
    assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-6), "KL when policy == reference must be ~0"


def test_kl_non_negative():
    cur_lp = torch.randn(10)
    ref_lp = torch.randn(10)

    kl = compute_kl_divergence(cur_lp, ref_lp, estimator="schulman")
    assert (kl >= -1e-6).all(), "Schulman KL estimator must be non-negative"
