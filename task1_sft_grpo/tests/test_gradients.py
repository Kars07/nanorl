"""Unit tests for gradient statistics and hooks."""

import torch
import torch.nn as nn

from sft_lab.hooks import ActivationTrackerHook, compute_gradient_stats


def test_gradient_stats():
    model = nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 2))
    x = torch.randn(4, 10)
    y = torch.tensor([0, 1, 0, 1])

    out = model(x)
    loss = torch.nn.functional.cross_entropy(out, y)
    loss.backward()

    stats = compute_gradient_stats(model)

    assert "global_grad_norm" in stats
    assert stats["global_grad_norm"] > 0.0
    assert len(stats["per_parameter_grads"]) > 0


def test_activation_tracker():
    model = nn.Sequential(nn.Linear(10, 10), nn.ReLU())
    tracker = ActivationTrackerHook(model)

    x = torch.randn(4, 10)
    _ = model(x)

    assert len(tracker.stats) > 0
    tracker.remove()
