# Training / Inference Boundary & Numerical Discrepancies

This document explains why identical model weights do not guarantee bitwise identical forward-pass logprobs between training frameworks (e.g. PyTorch FSDP2 / Megatron) and inference engines (e.g. vLLM / SGLang).

---

## 1. Mismatch Sources: Weights vs. Kernels vs. Templates

```text
Identical Output Discrepancy Hierarchy:
1. Tokenizer / Chat Template Mismatch (Human / Config Error)
2. Token ID / BOS / EOS Padding Mismatch (Data Pipeline Error)
3. Weight Parameter Mismatch (Unsynced / Stale State Dict)
4. Numerical Kernel Mismatch (FlashAttention vs PagedAttention vs FP16/BF16 Accumulation)
```

---

## 2. Diagnostic Protocol Sequence

To diagnose logprob discrepancies between trainer and sampler engines, follow this mandatory diagnostic order:

1. **Verify Token Sequence Identity**: Save exact integer token IDs `input_ids` produced during rollout and pass identical integer tokens to trainer teacher forcing.
2. **Verify Tokenizer & Chat Template**: Ensure identical special tokens (`<|im_start|>`, `<|im_end|>`) and chat templates.
3. **Verify Weight Fingerprint**: Run `compute_weight_fingerprint()` on both trainer and sampler models to confirm parameter identity.
4. **Verify Dtype & Precision**: Ensure both engines run under identical precision (e.g. `bfloat16`).
5. **Inspect Per-Token Logprobs**: Execute `run_logprob_mismatch_probe()` to isolate max/mean absolute logprob differences and locate the exact token index of worst disagreement.

---

## 3. Tolerances in Production RL

- Bitwise logprob equality is rarely achieved between vLLM (PagedAttention FP16 accumulation) and PyTorch eager/FlashAttention.
- Expected logprob tolerance:
  $$\max | \log \pi_{\text{trainer}} - \log \pi_{\text{sampler}} | < 10^{-3}$$
- Discrepancies exceeding $10^{-1}$ indicate a true configuration or weight mismatch bug!
