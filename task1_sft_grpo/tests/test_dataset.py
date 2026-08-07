"""Unit tests for dataset loading and inspection."""

from sft_lab.dataset import inspect_dataset_file, validate_conversation_structure


def test_dataset_fixtures_inspection():
    fixture_path = "data/fixtures/test_fixtures.jsonl"
    report = inspect_dataset_file(fixture_path, max_seq_length=256)

    assert report["total_examples"] == 6
    assert report["validation_issues_count"] > 0
    issues = report["issues_by_type"]

    assert "missing_field" in issues
    assert "empty_message" in issues
    assert "role_transition_error" in issues

    assert report["duplicates"]["duplicate_ids_count"] == 1


def test_valid_conversation_validation():
    valid_ex = {
        "id": "valid_1",
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ],
    }
    issues = validate_conversation_structure(valid_ex, 0)
    assert len(issues) == 0
