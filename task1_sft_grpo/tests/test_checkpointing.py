"""Unit tests for checkpoint saving and loading."""

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sft_lab.checkpointing import load_checkpoint, save_checkpoint


def test_save_and_load_checkpoint(tmp_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = str(tmp_path / "test_ckpt")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    save_checkpoint(
        model,
        tokenizer,
        save_dir,
        optimizer=optimizer,
        config={"test": 123},
        step=42,
        epoch=2,
    )

    assert os.path.exists(os.path.join(save_dir, "model.safetensors")) or os.path.exists(
        os.path.join(save_dir, "pytorch_model.bin")
    )
    assert os.path.exists(os.path.join(save_dir, "optimizer.pt"))
    assert os.path.exists(os.path.join(save_dir, "trainer_state.pt"))
    assert os.path.exists(os.path.join(save_dir, "environment_report.json"))

    loaded_model, loaded_tok, trainer_state = load_checkpoint(save_dir, device=device)
    assert trainer_state["step"] == 42
    assert trainer_state["epoch"] == 2
