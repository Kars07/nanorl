"""Metrics and manual loss verification module for SFT microscope."""

from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizer


def compute_manual_causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Compute manual Causal LM cross-entropy loss from logits and labels.

    Args:
        logits: Tensor of shape (batch, sequence, vocab_size)
        labels: Tensor of shape (batch, sequence)
        ignore_index: Label value to ignore in loss computation

    Returns:
        Scalar scalar tensor representing cross-entropy loss.
    """
    assert logits.dim() == 3, f"Logits rank must be 3, got shape {logits.shape}"
    assert labels.dim() == 2, f"Labels rank must be 2, got shape {labels.shape}"
    assert logits.shape[:2] == labels.shape, (
        f"Mismatch between logits shape {logits.shape[:2]} and labels shape {labels.shape}"
    )

    # Causal shift: prediction at position t is for target at position t+1
    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = labels[:, 1:].contiguous()

    # Reshape for cross_entropy
    vocab_size = shift_logits.size(-1)
    flat_logits = shift_logits.view(-1, vocab_size)
    flat_labels = shift_labels.view(-1)

    loss = F.cross_entropy(flat_logits, flat_labels, ignore_index=ignore_index)
    return loss


def decompose_per_token_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    input_ids: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
    batch_idx: int = 0,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Decompose logits per token position into detailed metrics."""
    b_logits = logits[batch_idx, :-1].float()  # (seq_len - 1, vocab)
    b_targets = labels[batch_idx, 1:]  # (seq_len - 1,)
    b_inputs = input_ids[batch_idx, 1:]  # (seq_len - 1,)

    probs = F.softmax(b_logits, dim=-1)
    log_probs = F.log_softmax(b_logits, dim=-1)

    rows = []
    seq_len = b_logits.shape[0]

    for t in range(seq_len):
        target_id = b_targets[t].item()
        input_id = b_inputs[t].item()

        input_tok_str = tokenizer.decode([input_id])
        target_tok_str = tokenizer.decode([target_id]) if target_id != -100 else "<IGNORED>"

        t_logits = b_logits[t]
        t_probs = probs[t]
        t_log_probs = log_probs[t]

        # Top 1 & Top K
        top_k_probs, top_k_ids = torch.topk(t_probs, k=top_k)
        top1_id = top_k_ids[0].item()
        top1_tok = tokenizer.decode([top1_id])

        top5_str_list = [f"{tokenizer.decode([idx.item()])} ({p.item():.4f})" for p, idx in zip(top_k_probs, top_k_ids)]

        # Entropy = - sum(p * log(p))
        entropy = -torch.sum(t_probs * t_log_probs).item()

        if target_id != -100:
            target_logit = t_logits[target_id].item()
            target_prob = t_probs[target_id].item()
            target_logprob = t_log_probs[target_id].item()
            token_ce = -target_logprob
            is_top1 = top1_id == target_id
            is_supervised = True
        else:
            target_logit = 0.0
            target_prob = 0.0
            target_logprob = 0.0
            token_ce = 0.0
            is_top1 = False
            is_supervised = False

        rows.append(
            {
                "position": t + 1,  # actual token position in input_ids
                "input_token": input_tok_str,
                "target_token": target_tok_str,
                "target_token_id": target_id,
                "target_logit": target_logit,
                "target_prob": target_prob,
                "target_logprob": target_logprob,
                "top1_token": top1_tok,
                "top5_tokens": ", ".join(top5_str_list),
                "entropy": entropy,
                "token_ce": token_ce,
                "is_correct_top1": is_top1,
                "is_supervised": is_supervised,
            }
        )

    return rows
