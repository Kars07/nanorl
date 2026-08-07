"""Dataset validation, inspection, and PyTorch Dataset classes."""

import json
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

VALID_ROLES = {"system", "user", "assistant", "tool"}


@dataclass
class ValidationIssue:
    example_id: str
    issue_type: str
    message: str


def validate_conversation_structure(example: Dict[str, Any], index: int) -> List[ValidationIssue]:
    """Validate structure of a single conversation record."""
    issues = []
    ex_id = str(example.get("id", f"record_{index}"))

    if "messages" not in example:
        issues.append(ValidationIssue(ex_id, "missing_field", "Missing 'messages' field"))
        return issues

    messages = example["messages"]
    if not isinstance(messages, list) or len(messages) == 0:
        issues.append(ValidationIssue(ex_id, "empty_conversation", "'messages' is empty or not a list"))
        return issues

    has_assistant = False
    prev_role = None

    for msg_idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            issues.append(ValidationIssue(ex_id, "malformed_record", f"Message at index {msg_idx} is not a dict"))
            continue

        role = msg.get("role")
        content = msg.get("content")

        if role not in VALID_ROLES:
            issues.append(ValidationIssue(ex_id, "unknown_role", f"Unknown role '{role}' at index {msg_idx}"))

        if content is None or (isinstance(content, str) and len(content.strip()) == 0):
            issues.append(
                ValidationIssue(ex_id, "empty_message", f"Empty content for role '{role}' at index {msg_idx}")
            )

        if role == "system" and msg_idx != 0:
            issues.append(
                ValidationIssue(
                    ex_id,
                    "role_transition_error",
                    f"System role at index {msg_idx} (must be first)",
                )
            )

        if prev_role is not None and prev_role == role and role in {"user", "assistant"}:
            issues.append(
                ValidationIssue(
                    ex_id,
                    "role_transition_error",
                    f"Consecutive messages with role '{role}' at index {msg_idx}",
                )
            )

        if role == "assistant":
            has_assistant = True

        prev_role = role

    if not has_assistant:
        issues.append(ValidationIssue(ex_id, "no_assistant_response", "Conversation contains no assistant response"))

    return issues


def compute_stats(values: List[float | int]) -> Dict[str, float]:
    """Compute summary statistics for numeric list."""
    if not values:
        return {
            "min": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }
    arr = np.array(values, dtype=float)
    return {
        "min": float(np.min(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def inspect_dataset_file(
    file_path: str,
    tokenizer_name_or_path: str = "Qwen/Qwen2.5-0.5B-Instruct",
    max_seq_length: int = 512,
) -> Dict[str, Any]:
    """Perform comprehensive inspection of a dataset JSON/JSONL file."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)

    records = []
    if file_path.endswith(".jsonl"):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            records = json.load(f)

    total_examples = len(records)
    issues_list: List[ValidationIssue] = []

    # Track statistics
    turn_counts = []
    user_counts = []
    assistant_counts = []
    system_counts = []
    tool_counts = []
    char_counts = []
    token_counts = []
    prompt_token_counts = []
    assistant_token_counts = []
    assistant_ratios = []

    truncated_count = 0
    assistant_truncated_count = 0
    supervised_tokens_lost = 0
    zero_supervised_count = 0

    seen_ids = set()
    duplicate_ids = set()
    exact_duplicate_convs = 0
    conv_hashes = set()

    exact_duplicate_rendered = 0
    rendered_hashes = set()

    source_stats: Dict[str, Dict[str, int]] = {}
    category_stats: Dict[str, Dict[str, int]] = {}

    for i, ex in enumerate(records):
        ex_id = str(ex.get("id", f"record_{i}"))
        if ex_id in seen_ids:
            duplicate_ids.add(ex_id)
        seen_ids.add(ex_id)

        ex_issues = validate_conversation_structure(ex, i)
        issues_list.extend(ex_issues)

        messages = ex.get("messages", [])
        if not messages:
            continue

        # Check duplicate conversations
        conv_str = json.dumps(messages, sort_keys=True)
        if conv_str in conv_hashes:
            exact_duplicate_convs += 1
        conv_hashes.add(conv_str)

        # Message counts
        user_c = sum(1 for m in messages if m.get("role") == "user")
        asst_c = sum(1 for m in messages if m.get("role") == "assistant")
        sys_c = sum(1 for m in messages if m.get("role") == "system")
        tool_c = sum(1 for m in messages if m.get("role") == "tool")

        turn_counts.append(len(messages))
        user_counts.append(user_c)
        assistant_counts.append(asst_c)
        system_counts.append(sys_c)
        tool_counts.append(tool_c)

        total_chars = sum(len(str(m.get("content", ""))) for m in messages)
        char_counts.append(total_chars)

        # Tokenizer analysis using chat template
        try:
            full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            if full_text in rendered_hashes:
                exact_duplicate_rendered += 1
            rendered_hashes.add(full_text)

            full_tokens = tokenizer.encode(full_text, add_special_tokens=False)
            total_toks = len(full_tokens)
            token_counts.append(total_toks)

            # Prompt tokens vs assistant tokens
            prompt_msgs = [m for m in messages if m.get("role") != "assistant"]
            if prompt_msgs:
                prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
                prompt_toks = len(tokenizer.encode(prompt_text, add_special_tokens=False))
            else:
                prompt_toks = 0

            asst_toks = max(0, total_toks - prompt_toks)
            prompt_token_counts.append(prompt_toks)
            assistant_token_counts.append(asst_toks)

            ratio = asst_toks / total_toks if total_toks > 0 else 0.0
            assistant_ratios.append(ratio)

            # Truncation analysis
            if total_toks > max_seq_length:
                truncated_count += 1
                if prompt_toks >= max_seq_length:
                    assistant_truncated_count += 1
                    supervised_tokens_lost += asst_toks
                    zero_supervised_count += 1
                else:
                    supervised_in_window = max_seq_length - prompt_toks
                    lost = asst_toks - supervised_in_window
                    if lost > 0:
                        assistant_truncated_count += 1
                        supervised_tokens_lost += lost
            else:
                if asst_toks == 0:
                    zero_supervised_count += 1

        except Exception as err:
            issues_list.append(ValidationIssue(ex_id, "template_rendering_error", str(err)))

        source = ex.get("source", "unknown")
        category = ex.get("category", "unknown")

        if source not in source_stats:
            source_stats[source] = {"count": 0, "tokens": 0}
        source_stats[source]["count"] += 1
        source_stats[source]["tokens"] += token_counts[-1] if token_counts else 0

        if category not in category_stats:
            category_stats[category] = {"count": 0, "tokens": 0}
        category_stats[category]["count"] += 1
        category_stats[category]["tokens"] += token_counts[-1] if token_counts else 0

    # Group issues by type
    issues_by_type: Dict[str, List[str]] = {}
    for issue in issues_list:
        if issue.issue_type not in issues_by_type:
            issues_by_type[issue.issue_type] = []
        issues_by_type[issue.issue_type].append(f"{issue.example_id}: {issue.message}")

    report = {
        "dataset_file": file_path,
        "tokenizer_name": tokenizer_name_or_path,
        "max_seq_length": max_seq_length,
        "total_examples": total_examples,
        "validation_issues_count": len(issues_list),
        "issues_by_type": issues_by_type,
        "duplicates": {
            "duplicate_ids_count": len(duplicate_ids),
            "duplicate_ids": list(duplicate_ids),
            "exact_duplicate_conversations": exact_duplicate_convs,
            "exact_duplicate_rendered": exact_duplicate_rendered,
        },
        "stats": {
            "turns": compute_stats(turn_counts),
            "user_messages": compute_stats(user_counts),
            "assistant_messages": compute_stats(assistant_counts),
            "system_messages": compute_stats(system_counts),
            "tool_messages": compute_stats(tool_counts),
            "characters": compute_stats(char_counts),
            "total_tokens": compute_stats(token_counts),
            "prompt_tokens": compute_stats(prompt_token_counts),
            "assistant_tokens": compute_stats(assistant_token_counts),
            "assistant_ratio": compute_stats(assistant_ratios),
        },
        "truncation": {
            "max_seq_length": max_seq_length,
            "truncated_examples": truncated_count,
            "truncated_percentage": (truncated_count / total_examples * 100.0) if total_examples > 0 else 0.0,
            "assistant_truncated_examples": assistant_truncated_count,
            "assistant_truncated_percentage": (assistant_truncated_count / total_examples * 100.0)
            if total_examples > 0
            else 0.0,
            "supervised_tokens_lost": supervised_tokens_lost,
            "zero_supervised_examples": zero_supervised_count,
        },
        "sources": source_stats,
        "categories": category_stats,
    }
    return report


class SFTDataset(Dataset):
    """PyTorch Dataset for SFT conversations."""

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 512,
        assistant_only_loss: bool = True,
        allow_zero_supervised_tokens: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.assistant_only_loss = assistant_only_loss

        self.records = []
        if data_path.endswith(".jsonl"):
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.records.append(json.loads(line))
        else:
            with open(data_path, "r", encoding="utf-8") as f:
                self.records = json.load(f)

        # Pre-filter or check zero supervised tokens
        valid_records = []
        for i, ex in enumerate(self.records):
            issues = validate_conversation_structure(ex, i)
            # Filter out severely malformed records (e.g. missing messages or no assistant)
            fatal = any(
                iss.issue_type in {"missing_field", "empty_conversation", "no_assistant_response"} for iss in issues
            )
            if not fatal:
                valid_records.append(ex)

        self.records = valid_records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.records[idx]
        messages = ex["messages"]

        return {
            "id": ex.get("id", str(idx)),
            "messages": messages,
            "source": ex.get("source", "unknown"),
            "category": ex.get("category", "unknown"),
        }
