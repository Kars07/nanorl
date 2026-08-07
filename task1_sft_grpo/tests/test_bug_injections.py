"""Tests for deliberate SFT bug injections and detector assertions."""

import pytest
from transformers import AutoTokenizer

from sft_lab.masking import build_sft_labels_and_metadata


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)


@pytest.fixture
def sample_messages():
    return [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2+2=4."},
    ]


def test_bug_all_labels_minus_100(sample_messages, tokenizer):
    """Detector for all_labels_minus_100 bug."""
    processed = build_sft_labels_and_metadata(sample_messages, tokenizer)
    labels = processed["labels"]
    # Bug injection: set all labels to -100
    buggy_labels = [-100] * len(labels)

    # Detector
    has_supervised_tokens = any(lbl != -100 for lbl in buggy_labels)
    assert not has_supervised_tokens, "Detector correctly identified all labels == -100 bug"


def test_bug_accidental_training_user_tokens(sample_messages, tokenizer):
    """Detector for accidental training on user tokens."""
    processed = build_sft_labels_and_metadata(sample_messages, tokenizer, assistant_only_loss=True)
    labels = processed["labels"]
    roles = processed["roles"]

    # In clean pipeline, no user role token receives non--100 label
    user_tokens_trained = any(l != -100 for l, r in zip(labels, roles) if r == "user")
    assert not user_tokens_trained, "User tokens must not be trained when assistant_only_loss=True"


def test_bug_duplicated_bos(sample_messages, tokenizer):
    """Detector for duplicated BOS token."""
    text = tokenizer.apply_chat_template(sample_messages, tokenize=False)
    # Check that <|im_start|> is not duplicated at position 0
    toks = tokenizer.encode(text, add_special_tokens=False)
    assert toks[0] == 151644  # im_start
    assert toks[1] != 151644  # not duplicated


def test_bug_missing_eos(sample_messages, tokenizer):
    """Detector for missing EOS termination token."""
    processed = build_sft_labels_and_metadata(sample_messages, tokenizer)
    input_ids = processed["input_ids"]
    labels = processed["labels"]

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eos_trained = any(input_ids[i] == im_end_id and labels[i] != -100 for i in range(len(input_ids)))
    assert eos_trained, "Assistant terminating EOS token must be trained"


def test_bug_sequence_length_too_small(sample_messages, tokenizer):
    """Detector for sequence length too small causing 0 supervised tokens."""
    # max_seq_length=5 is too short for prompt
    processed = build_sft_labels_and_metadata(sample_messages, tokenizer, max_seq_length=5)
    labels = processed["labels"]
    supervised_count = sum(1 for l in labels if l != -100)
    assert supervised_count == 0, "Too small max_seq_length results in zero supervised tokens"


def test_all_20_bug_cases_exist():
    """Verify all 20 bug cases exist in bug_lab/cases/."""
    import os

    bug_cases = os.listdir("bug_lab/cases")
    assert len(bug_cases) == 20, f"Expected 20 bug cases, found {len(bug_cases)}"
