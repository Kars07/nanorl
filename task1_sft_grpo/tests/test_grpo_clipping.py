"""Unit tests for GRPO PPO clipping bounds and advantages."""

import torch

from sft_lab.grpo.objective import compute_grpo_loss


def test_clipping_positive_advantage():
    # Large ratio increase with positive advantage should be clipped at (1 + eps) * adv
    cur_lp = torch.tensor([[0.0]])
    old_lp = torch.tensor([[-2.0]])  # ratio = exp(2.0) = 7.38
    ref_lp = cur_lp.clone()
    adv = torch.tensor([[1.0]])
    comp_mask = torch.tensor([[1]])

    loss, metrics = compute_grpo_loss(cur_lp, old_lp, ref_lp, adv, comp_mask, clip_eps=0.2, kl_coeff=0.0)

    # Policy loss should be - (1 + 0.2) * 1.0 = -1.2
    assert abs(metrics["policy_loss"] - (-1.2)) < 1e-5
    assert metrics["clipping_fraction"] == 1.0


def test_clipping_negative_advantage():
    # Large ratio decrease with negative advantage should be clipped at (1 - eps) * adv
    cur_lp = torch.tensor([[-2.0]])
    old_lp = torch.tensor([[0.0]])  # ratio = exp(-2.0) = 0.135
    ref_lp = cur_lp.clone()
    adv = torch.tensor([[-1.0]])
    comp_mask = torch.tensor([[1]])

    loss, metrics = compute_grpo_loss(cur_lp, old_lp, ref_lp, adv, comp_mask, clip_eps=0.2, kl_coeff=0.0)

    # surr1 = 0.135 * (-1) = -0.135; surr2 = 0.8 * (-1) = -0.8
    # min(-0.135, -0.8) = -0.8 -> policy_loss = - (-0.8) = +0.8
    assert abs(metrics["policy_loss"] - 0.8) < 1e-4
