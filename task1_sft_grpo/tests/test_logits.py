"""Unit tests for per-token logit decomposition."""

import torch
from transformers import AutoTokenizer

from sft_lab.metrics import decompose_per_token_logits


def test_decompose_per_token_logits():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
    seq_len = 10
    vocab = tokenizer.vocab_size

    logits = torch.randn(1, seq_len, vocab)
    labels = torch.tensor([[-100, -100, 10, 20, 30, -100, -100, -100, -100, -100]])
    input_ids = torch.randint(0, vocab, (1, seq_len))

    rows = decompose_per_token_logits(logits, labels, input_ids, tokenizer, batch_idx=0)

    assert len(rows) == seq_len - 1

    sup_rows = [r for r in rows if r["is_supervised"]]
    assert len(sup_rows) == 3

    for r in sup_rows:
        assert r["token_ce"] >= 0.0
        assert 0.0 <= r["target_prob"] <= 1.0
        assert r["target_logprob"] <= 0.0
