"""Unit tests for mismatch probe components."""

import torch
import torch.nn as nn

from mismatch_probe.logprob_probe import compute_trainer_selected_logprobs
from mismatch_probe.policy_version import PolicyVersionTracker
from mismatch_probe.weight_fingerprint import assert_weight_identity, compute_weight_fingerprint


def test_policy_version_tracker():
    tracker = PolicyVersionTracker(initial_version=0)
    assert tracker.version == 0
    v1 = tracker.increment({"step": 1})
    assert v1 == 1
    assert len(tracker.get_history()) == 2


def test_weight_fingerprint():
    model_a = nn.Linear(10, 10)
    model_b = nn.Linear(10, 10)
    model_b.load_state_dict(model_a.state_dict())

    fp_a = compute_weight_fingerprint(model_a)
    fp_b = compute_weight_fingerprint(model_b)

    assert fp_a["digest"] == fp_b["digest"]
    assert_weight_identity(fp_a, fp_b)

    # Mutate model_b
    with torch.no_grad():
        model_b.weight.add_(0.5)

    fp_b_mut = compute_weight_fingerprint(model_b)
    assert fp_a["digest"] != fp_b_mut["digest"]


def test_selected_logprobs():
    model = nn.Sequential(
        nn.Embedding(100, 16),
        nn.Linear(16, 100),
    )
    tokens = torch.tensor([[1, 5, 9, 12, 45]])
    mask = torch.ones_like(tokens)

    logprobs = compute_trainer_selected_logprobs(model, tokens, mask)
    assert logprobs.shape == (1, 4)
    assert not torch.isnan(logprobs).any()
