"""Unit tests for SFTDataCollator batch padding and shapes."""

import pytest
from transformers import AutoTokenizer

from sft_lab.collator import SFTDataCollator


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)


def test_collator_batch_padding(tokenizer):
    collator = SFTDataCollator(tokenizer, max_seq_length=256)
    batch_raw = [
        {
            "messages": [
                {"role": "user", "content": "Short prompt"},
                {"role": "assistant", "content": "Short ans"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Much longer prompt string asking a question"},
                {
                    "role": "assistant",
                    "content": "Much longer assistant response answering the question in detail.",
                },
            ]
        },
    ]

    batch = collator(batch_raw)

    input_ids = batch["input_ids"]
    labels = batch["labels"]
    attn_mask = batch["attention_mask"]

    assert input_ids.shape == labels.shape == attn_mask.shape
    assert input_ids.shape[0] == 2

    # Check padding in labels is -100
    for b in range(input_ids.shape[0]):
        for t in range(input_ids.shape[1]):
            if attn_mask[b, t] == 0:
                assert labels[b, t] == -100
