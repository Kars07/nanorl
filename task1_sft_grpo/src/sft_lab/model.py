"""Model loader and PyTorch model utilities."""

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def load_model_and_tokenizer(
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    tokenizer_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    dtype: str = "bfloat16",
    device: str | None = None,
    use_gradient_checkpointing: bool = False,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load model and tokenizer for Qwen2.5-0.5B-Instruct with memory optimizations."""

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32,
    }
    torch_dtype = dtype_map.get(dtype, torch.float32)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )

    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()

    model.to(device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return model, tokenizer
