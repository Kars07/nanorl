"""Unit tests verifying failure injection detectors."""

import numpy as np
import torch
import torch.nn as nn

from failure_lab.cases import (
    inject_different_chat_formatting,
    inject_different_precision,
    inject_different_token_sequence,
    inject_policy_lag_off_policy,
    inject_stale_inference_weights,
    inject_unsynced_sampler_weights,
    inject_wrong_completion_mask,
    inject_wrong_tokenizer,
)


def test_stale_inference_weights():
    w = np.ones((5,), dtype=np.float32)
    w_trainer, w_sampler = inject_stale_inference_weights(w, lag_steps=3)
    assert not np.allclose(w_trainer, w_sampler)


def test_wrong_tokenizer():
    res = inject_wrong_tokenizer("Hello world")
    assert res["raw_tokens"] != res["mismatched_tokens"]


def test_different_chat_formatting():
    res = inject_different_chat_formatting("Test prompt")
    assert res["format_a"] != res["format_b"]


def test_different_token_sequence():
    toks = torch.tensor([[10, 20, 30]])
    alt = inject_different_token_sequence(toks)
    assert not torch.equal(toks, alt)


def test_policy_lag_off_policy():
    res = inject_policy_lag_off_policy(current_version=10, rollout_version=2)
    assert res["lag"] == 8


def test_different_precision():
    fp32 = torch.tensor([1.2345678, 2.3456789], dtype=torch.float32)
    f32, bf16_cast = inject_different_precision(fp32)
    diff = torch.abs(f32 - bf16_cast).max().item()
    assert diff > 0.0, "Casting FP32 to BF16 should produce non-zero quantization precision delta"


def test_wrong_completion_mask():
    correct, buggy = inject_wrong_completion_mask(seq_len=10, prompt_len=3, pad_len=2)
    assert not torch.equal(correct, buggy)
    assert buggy.sum().item() > correct.sum().item()


def test_unsynced_sampler_weights():
    m_trainer = nn.Linear(4, 4)
    m_sampler = nn.Linear(4, 4)
    m_sampler.load_state_dict(m_trainer.state_dict())

    fp1, fp2 = inject_unsynced_sampler_weights(m_trainer, m_sampler)
    assert fp1 != fp2
