"""Unit tests for GRPO group-relative advantage computation."""

import pytest
import torch

from sft_lab.grpo.advantages import compute_group_relative_advantages


def test_group_advantage_normalization():
    rewards = [1.0, 0.0, 1.0, 0.0]
    group_ids = ["p1", "p1", "p1", "p1"]

    advs = compute_group_relative_advantages(rewards, group_ids, use_std_normalization=True)
    assert advs.shape == (4, 1)
    assert abs(advs.mean().item()) < 1e-5, "Mean advantage must be approximately zero"


def test_constant_reward_group():
    rewards = [1.0, 1.0, 1.0, 1.0]
    group_ids = ["p1", "p1", "p1", "p1"]

    advs = compute_group_relative_advantages(rewards, group_ids, use_std_normalization=True)
    assert not torch.isnan(advs).any(), "Constant reward group must not produce NaN"
    assert (advs == 0.0).all(), "Constant reward group must yield 0 advantages"


def test_mismatched_group_ids_length():
    with pytest.raises(AssertionError):
        compute_group_relative_advantages([1.0, 2.0], ["p1"])
