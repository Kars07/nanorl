# Custom GRPO vs HuggingFace TRL GRPOTrainer Comparison

This document details the architectural and mathematical comparison between our custom from-scratch GRPO implementation (`src/sft_lab/grpo/`) and Hugging Face TRL's `GRPOTrainer`.

## Feature & Component Comparison Matrix

| Component | Our Implementation | TRL GRPOTrainer Behavior | Same / Different | Engineering Rationale / Why |
| --- | --- | --- | --- | --- |
| **Group Generation** | Prompt-level `generate(num_generations=G)` with sampling temp/top_p | HuggingFace `vLLM` or native HF `generate(num_return_sequences=G)` | **Same** | Generates G completion rollouts per prompt. |
| **Advantage Normalization** | $A_i = \frac{r_i - \mu_{group}}{\sigma_{group} + \epsilon}$ (with zero-variance safety check) | Group-level z-score normalization across prompt completions | **Same** | Standard DeepSeekMath GRPO advantage normalization formula. |
| **Loss Masking** | Active only on completion action tokens (`completion_mask == 1`); prompt and padding receive 0 weight | Masked cross-entropy / policy loss on generated tokens | **Same** | Prevents prompt tokens from receiving policy gradient updates. |
| **Policy Ratio & Clipping** | PPO-style clipped surrogate: $\min(r \cdot A, \text{clip}(r, 1-\epsilon, 1+\epsilon) \cdot A)$ | PPO-style clipped surrogate loss | **Same** | Prevents destructively large policy updates per rollout batch over $K$ epochs. |
| **KL Divergence Estimator** | Schulman $k_3$ estimator: $\exp(r_{ref} - r_{cur}) - (r_{ref} - r_{cur}) - 1$ | Reverse KL or Schulman $k_3$ non-negative estimator | **Same** | Provides non-negative, zero-centered penalty relative to reference SFT model. |
| **Optimization Epochs $K$** | Explicit $K$-epoch loop over static rollout batch without re-generating rollouts | Configurable `num_generations` & mini-batch gradient updates | **Same** | On-policy/near-policy rollout buffer re-use. |

## Variants & Modern Alternatives

1. **Dr-GRPO / DAPO**:
   - Modern variants disable group $\sigma$ division when reward variance is low or zero across prompts to preserve natural gradient scale across varying prompt difficulties.
2. **TRL Defaults**:
   - TRL supports optional vLLM integration for fast rollout generation at scale and optional per-token kl penalty arrays.

## References

- DeepSeekMath Paper: [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
- TRL GRPOTrainer Documentation: [huggingface.co/docs/trl/grpo_trainer](https://huggingface.co/docs/trl/grpo_trainer)
