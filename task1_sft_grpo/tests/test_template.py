"""Unit tests for chat template rendering and masking."""

import pytest
from transformers import AutoTokenizer

from sft_lab.masking import build_sft_labels_and_metadata


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)


def test_chat_template_rendering(tokenizer):
    messages = [
        {"role": "user", "content": "What is 10 + 20?"},
        {"role": "assistant", "content": "10 + 20 = 30."},
    ]
    processed = build_sft_labels_and_metadata(messages, tokenizer, max_seq_length=256)

    # Check BOS/EOS and template tokens
    assert "<|im_start|>user" in processed["full_text"]
    assert "<|im_start|>assistant" in processed["full_text"]
    assert "<|im_end|>" in processed["full_text"]

    labels = processed["labels"]
    input_ids = processed["input_ids"]

    # Check that at least one token is trained
    trained_tokens = [l for l in labels if l != -100]
    assert len(trained_tokens) > 0, "Assistant tokens must have trained labels != -100"

    # Check that user prompt tokens have label -100
    roles = processed["roles"]
    for l, r in zip(labels, roles):
        if r == "user":
            assert l == -100, f"User token role {r} has non--100 label {l}"
