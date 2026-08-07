"""Assistant loss masking and token role tagging utilities."""

from typing import Any, Dict, List

from transformers import PreTrainedTokenizer


def build_sft_labels_and_metadata(
    messages: List[Dict[str, str]],
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int = 512,
    assistant_only_loss: bool = True,
) -> Dict[str, Any]:
    """Render conversation using chat template, tokenize, assign loss labels and role tags.

    Rules for assistant_only_loss:
    - Non-assistant tokens (system, user, headers) receive -100 label.
    - Assistant content tokens receive token_id label (trained).
    - Assistant terminating EOS token (<|im_end|>) receives token_id label (trained, so model learns when to stop).
    - Trailing newlines or padding tokens receive -100.
    """
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    input_ids = tokenizer.encode(full_text, add_special_tokens=False)

    labels = [-100] * len(input_ids)
    roles = ["system"] * len(input_ids)

    # Re-tokenize prefixes step by step to find exact span boundaries
    # We iterate through messages and identify turn token ranges
    cur_text = ""
    prev_tok_len = 0

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is None or im_end_id == tokenizer.unk_token_id:
        im_end_id = tokenizer.eos_token_id

    for idx, msg in enumerate(messages):
        role = msg["role"]
        # Apply chat template up to current message
        sub_messages = messages[: idx + 1]
        sub_text = tokenizer.apply_chat_template(sub_messages, tokenize=False, add_generation_prompt=False)
        sub_ids = tokenizer.encode(sub_text, add_special_tokens=False)
        cur_tok_len = len(sub_ids)

        # Token slice for this message turn
        turn_range = range(prev_tok_len, min(cur_tok_len, len(input_ids)))

        for t_idx in turn_range:
            roles[t_idx] = role

        if role == "assistant" or not assistant_only_loss:
            # We need to distinguish assistant header (<|im_start|>assistant\n) from content + <|im_end|>
            # Find prompt text right before assistant content
            prompt_msgs = messages[:idx]
            prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            content_start = len(prompt_ids)

            # Trailing token for turn might be newline after <|im_end|>
            for t_idx in turn_range:
                if t_idx >= content_start:
                    tid = input_ids[t_idx]
                    # Check if token is trailing newline after im_end
                    if t_idx > 0 and input_ids[t_idx - 1] == im_end_id and tid in (198, tokenizer.eos_token_id):
                        labels[t_idx] = -100
                    else:
                        labels[t_idx] = tid

        prev_tok_len = cur_tok_len

    # Handle max sequence length truncation
    if len(input_ids) > max_seq_length:
        input_ids = input_ids[:max_seq_length]
        labels = labels[:max_seq_length]
        roles = roles[:max_seq_length]

    attention_mask = [1] * len(input_ids)

    return {
        "full_text": full_text,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "roles": roles,
    }


def create_token_inspection_table(
    input_ids: List[int],
    labels: List[int],
    roles: List[str],
    tokenizer: PreTrainedTokenizer,
) -> List[Dict[str, Any]]:
    """Generate detailed token inspection rows."""
    table = []
    for idx, (tid, label, role) in enumerate(zip(input_ids, labels, roles)):
        tok_str = tokenizer.decode([tid])
        is_special = tid in tokenizer.all_special_ids
        trained = label != -100
        table.append(
            {
                "idx": idx,
                "token": tok_str,
                "token_id": tid,
                "label": label,
                "trained": trained,
                "role": role,
                "is_special": is_special,
            }
        )
    return table
