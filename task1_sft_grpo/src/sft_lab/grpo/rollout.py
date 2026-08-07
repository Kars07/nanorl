"""Rollout generator for GRPO prompt groups."""

from typing import Any, Dict, List

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from sft_lab.grpo.advantages import compute_group_relative_advantages
from sft_lab.grpo.logprobs import compute_reference_logprobs
from sft_lab.grpo.rewards import compute_math_reward
from sft_lab.grpo.types import RolloutBatch


def generate_grpo_rollouts(
    policy_model: PreTrainedModel,
    ref_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts_data: List[Dict[str, Any]],
    num_generations: int = 4,
    max_prompt_length: int = 256,
    max_completion_length: int = 128,
    temperature: float = 0.7,
    top_p: float = 1.0,
    use_std_normalization: bool = True,
    device: str = "cuda",
) -> RolloutBatch:
    """Generate G completions per prompt, compute old and reference logprobs, rewards, and group advantages."""

    policy_model.eval()

    all_input_ids = []
    all_comp_masks = []
    all_rewards = []
    all_group_ids = []

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is None or im_end_id == tokenizer.unk_token_id:
        im_end_id = tokenizer.eos_token_id

    eos_token_ids = [tokenizer.eos_token_id]
    if im_end_id != tokenizer.eos_token_id:
        eos_token_ids.append(im_end_id)
    eos_set = set(eos_token_ids)
    if tokenizer.pad_token_id is not None:
        eos_set.add(tokenizer.pad_token_id)

    for prompt_entry in prompts_data:
        prompt_id = prompt_entry["id"]
        prompt_text = prompt_entry["prompt"]
        target_answer = prompt_entry["target_answer"]

        # Format prompt with template
        messages = [{"role": "user", "content": prompt_text}]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
        prompt_len = prompt_inputs["input_ids"].shape[1]

        # Generate G completions for this prompt
        with torch.no_grad():
            gen_outputs = policy_model.generate(
                **prompt_inputs,
                num_return_sequences=num_generations,
                max_new_tokens=max_completion_length,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                eos_token_id=eos_token_ids,
            )

        for g in range(num_generations):
            seq_ids = gen_outputs[g]
            seq_len = seq_ids.shape[0]

            # Find first EOS or pad token in completion span (at or after prompt_len)
            first_eos_idx = seq_len - 1
            for tok_idx in range(prompt_len, seq_len):
                if seq_ids[tok_idx].item() in eos_set:
                    first_eos_idx = tok_idx
                    break

            completion_ids = seq_ids[prompt_len : first_eos_idx + 1]
            completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)

            # Compute reward
            reward = compute_math_reward(completion_text, target_answer)

            # Create completion mask (0 for prompt and trailing pad, 1 for real completion action tokens)
            comp_mask = torch.zeros(seq_len - 1, dtype=torch.long, device=device)
            # Active target tokens: position prompt_len to first_eos_idx (corresponding to logits prompt_len - 1 to first_eos_idx - 1)
            comp_mask[prompt_len - 1 : first_eos_idx] = 1

            all_input_ids.append(seq_ids)
            all_comp_masks.append(comp_mask)
            all_rewards.append(reward)
            all_group_ids.append(prompt_id)

    # Pad sequences to batch max length
    batch_max_len = max(seq.shape[0] for seq in all_input_ids)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    padded_ids_list = []
    padded_masks_list = []
    padded_comp_masks_list = []

    for seq, comp_m in zip(all_input_ids, all_comp_masks):
        pad_len = batch_max_len - seq.shape[0]

        pad_ids = torch.full((pad_len,), pad_id, dtype=torch.long, device=device)
        full_seq = torch.cat([seq, pad_ids], dim=0)

        attn_m = torch.ones(full_seq.shape[0], dtype=torch.long, device=device)
        if pad_len > 0:
            attn_m[-pad_len:] = 0

        pad_comp_m = torch.zeros(pad_len, dtype=torch.long, device=device)
        full_comp_m = torch.cat([comp_m, pad_comp_m], dim=0)

        padded_ids_list.append(full_seq)
        padded_masks_list.append(attn_m)
        padded_comp_masks_list.append(full_comp_m)

    batch_input_ids = torch.stack(padded_ids_list, dim=0)
    batch_attn_mask = torch.stack(padded_masks_list, dim=0)
    batch_comp_mask = torch.stack(padded_comp_masks_list, dim=0)

    # Compute old policy logprobs on policy_model device
    old_logprobs = compute_reference_logprobs(policy_model, batch_input_ids, batch_attn_mask)

    # Compute reference policy logprobs on ref_model device
    ref_device = next(ref_model.parameters()).device
    ref_ids = batch_input_ids.to(ref_device)
    ref_mask = batch_attn_mask.to(ref_device)
    ref_logprobs = compute_reference_logprobs(ref_model, ref_ids, ref_mask).to(device)

    # Compute advantages
    advantages = compute_group_relative_advantages(
        rewards=all_rewards,
        group_ids=all_group_ids,
        use_std_normalization=use_std_normalization,
    ).to(device)

    return RolloutBatch(
        input_ids=batch_input_ids,
        attention_mask=batch_attn_mask,
        completion_mask=batch_comp_mask,
        old_logprobs=old_logprobs,
        reference_logprobs=ref_logprobs,
        advantages=advantages,
        group_ids=all_group_ids,
        rewards=all_rewards,
    )
