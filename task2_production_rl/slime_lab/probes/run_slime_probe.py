"""slime component probe script evaluating Qwen2.5-0.5B-Instruct configuration."""

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mismatch_probe.weight_fingerprint import compute_weight_fingerprint


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"

    print(f"Loading slime Qwen2.5-0.5B policy model `{model_id}` on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(device)

    fp = compute_weight_fingerprint(model)
    print(f"Slime Qwen2.5-0.5B Weight Fingerprint: Global L2={fp['global_l2']:.4f}, Digest={fp['digest']}")

    os.makedirs("slime_lab/artifacts", exist_ok=True)
    print("slime component probe run completed successfully.")


if __name__ == "__main__":
    main()
