"""Unit tests for masking logic and assertions."""

import pytest
from transformers import AutoTokenizer

from sft_lab.masking import build_sft_labels_and_metadata


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)


def test_assistant_only_loss_masking(tokenizer):
    messages = [
        {"role": "system", "content": "You are a math tutor."},
        {"role": "user", "content": "Calculate 5 * 6."},
        {"role": "assistant", "content": "5 * 6 = 30."},
    ]
    processed = build_sft_labels_and_metadata(messages, tokenizer, assistant_only_loss=True)

    labels = processed["labels"]
    roles = processed["roles"]

    for lbl, role in zip(labels, roles):
        if role in ["system", "user"]:
            assert lbl == -100, f"Role '{role}' received trained label {lbl}"

    # Assistant content tokens must be trained
    asst_trained = sum(1 for lbl, role in zip(labels, roles) if role == "assistant" and lbl != -100)
    assert asst_trained > 0, "Assistant tokens must be trained"
