"""Prime-RL component probe script evaluating Qwen2.5-0.5B-Instruct adaptation."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mismatch_probe.logprob_probe import run_logprob_mismatch_probe
from mismatch_probe.report import generate_mismatch_report
from mismatch_probe.weight_fingerprint import compute_weight_fingerprint


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"

    print(f"Loading policy model `{model_id}` on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(device)

    # Compute base model weight fingerprint
    fp_base = compute_weight_fingerprint(model)

    # Create model_b with slight perturbation to simulate updated policy
    model_b = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(device)

    # Run logprob mismatch probe between model and perturbed model_b
    probe_results = run_logprob_mismatch_probe(
        model_a=model,
        model_b=model_b,
        tokenizer=tokenizer,
        prompt_text="Solve: 15 + 27",
        completion_text=" 15 + 27 = 42. #### 42",
    )

    generate_mismatch_report(probe_results, fp_base)
    print("Prime-RL probe run completed successfully.")


if __name__ == "__main__":
    main()
