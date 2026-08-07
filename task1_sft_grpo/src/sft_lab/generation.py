"""Generation and evaluation utilities for LLM inference."""

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


def generate_completions(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: list[str],
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    top_p: float = 1.0,
    do_sample: bool = False,
    device: str | None = None,
) -> list[str]:
    """Generate completion text for a list of prompt strings."""

    if device is None:
        device = next(model.parameters()).device

    completions = []
    model.eval()

    for prompt in prompts:
        # Format prompt with system/user template if needed
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt_text = prompt

        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

        with torch.no_grad():
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "pad_token_id": tokenizer.pad_token_id
                if tokenizer.pad_token_id is not None
                else tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if do_sample and temperature > 0:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = top_p
            else:
                gen_kwargs["do_sample"] = False

            outputs = model.generate(**inputs, **gen_kwargs)

        input_len = inputs["input_ids"].shape[1]
        gen_tokens = outputs[0, input_len:]
        completion = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        completions.append(completion)

    return completions
