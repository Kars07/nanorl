"""Logprob mismatch probe comparing trainer vs sampler selected-token logprobs."""

from typing import Any, Dict

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


def compute_trainer_selected_logprobs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute teacher-forcing selected-token logprobs on training model."""
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        try:
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits if hasattr(out, "logits") else out
        except TypeError:
            logits = model(input_ids)

    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_tokens = input_ids[:, 1:].contiguous()

    # Memory-efficient logprob: logprob(target) = target_logit - logsumexp(logits)
    target_logits = shift_logits.gather(dim=-1, index=shift_tokens.unsqueeze(-1)).squeeze(-1)
    log_sum_exp = torch.logsumexp(shift_logits, dim=-1)

    logprobs = target_logits - log_sum_exp
    return logprobs


def run_logprob_mismatch_probe(
    model_a: PreTrainedModel,
    model_b: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt_text: str = "Solve the equation: 2x + 6 = 14. What is x?",
    completion_text: str = " To solve 2x + 6 = 14, subtract 6: 2x = 8, so x = 4. #### 4",
) -> Dict[str, Any]:
    """Compare selected-token logprobs between two model configurations on identical prompt+completion."""
    messages = [{"role": "user", "content": prompt_text}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    full_text = formatted_prompt + completion_text
    tokens = tokenizer.encode(full_text, return_tensors="pt")

    prompt_tokens = tokenizer.encode(formatted_prompt, return_tensors="pt")
    prompt_len = prompt_tokens.shape[1]

    logprobs_a = compute_trainer_selected_logprobs(model_a, tokens, torch.ones_like(tokens))[0]
    logprobs_b = compute_trainer_selected_logprobs(model_b, tokens, torch.ones_like(tokens))[0]

    # Focus on completion tokens span
    comp_logprobs_a = logprobs_a[prompt_len - 1 :].cpu().numpy()
    comp_logprobs_b = logprobs_b[prompt_len - 1 :].cpu().numpy()

    abs_diff = np.abs(comp_logprobs_a - comp_logprobs_b)
    max_abs_diff = float(np.max(abs_diff))
    mean_abs_diff = float(np.mean(abs_diff))
    worst_pos = int(np.argmax(abs_diff))

    tok_ids = tokens[0, prompt_len:].tolist()
    worst_token = tokenizer.decode([tok_ids[worst_pos]]) if worst_pos < len(tok_ids) else "N/A"

    return {
        "prompt_text": prompt_text,
        "completion_text": completion_text,
        "prompt_length": prompt_len,
        "completion_length": len(comp_logprobs_a),
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "worst_disagreement_index": worst_pos,
        "worst_disagreement_token": worst_token,
        "logprobs_a": comp_logprobs_a.tolist(),
        "logprobs_b": comp_logprobs_b.tolist(),
        "abs_diffs": abs_diff.tolist(),
    }
