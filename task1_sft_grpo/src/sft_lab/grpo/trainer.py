"""From-scratch GRPO Trainer implementation."""

from typing import Any, Dict, List

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from sft_lab.config import GRPOConfig
from sft_lab.grpo.logprobs import get_per_token_logprobs
from sft_lab.grpo.objective import compute_grpo_loss
from sft_lab.grpo.rollout import generate_grpo_rollouts


class GRPOTrainer:
    """Production-quality from-scratch GRPO Trainer."""

    def __init__(
        self,
        policy_model: PreTrainedModel,
        ref_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        optimizer: torch.optim.Optimizer,
        config: GRPOConfig,
        device: str = "cuda",
    ):
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.config = config
        self.device = device

        self.policy_version = 0

    def step(self, prompts_batch: List[Dict[str, Any]]) -> Dict[str, float]:
        """Perform one full GRPO rollout collection + K optimization epochs step."""

        # Phase 1: Rollout collection under old policy
        rollout_batch = generate_grpo_rollouts(
            policy_model=self.policy_model,
            ref_model=self.ref_model,
            tokenizer=self.tokenizer,
            prompts_data=prompts_batch,
            num_generations=self.config.num_generations,
            max_prompt_length=self.config.max_prompt_length,
            max_completion_length=self.config.max_completion_length,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            use_std_normalization=self.config.use_std_normalization,
            device=self.device,
        )

        reward_mean = float(np.mean(rollout_batch.rewards))
        reward_std = float(np.std(rollout_batch.rewards))

        # Phase 2: K Optimization Epochs over the collected rollout batch
        self.policy_model.train()

        epoch_metrics = []

        for k in range(self.config.num_epochs):
            self.optimizer.zero_grad()

            out = self.policy_model(
                input_ids=rollout_batch.input_ids,
                attention_mask=rollout_batch.attention_mask,
                use_cache=False,
            )

            current_logprobs = get_per_token_logprobs(out.logits, rollout_batch.input_ids)

            loss, metrics = compute_grpo_loss(
                current_logprobs=current_logprobs,
                old_logprobs=rollout_batch.old_logprobs,
                reference_logprobs=rollout_batch.reference_logprobs,
                advantages=rollout_batch.advantages,
                completion_mask=rollout_batch.completion_mask,
                clip_eps=self.config.clip_eps,
                kl_coeff=self.config.kl_coeff,
            )

            if torch.isnan(loss) or torch.isinf(loss):
                raise ValueError("NaN or Inf detected in GRPO loss computation!")

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.config.grad_clip).item()
            self.optimizer.step()

            metrics["grad_norm"] = grad_norm
            epoch_metrics.append(metrics)

        self.policy_version += 1

        # Aggregate metrics
        avg_loss = float(np.mean([m["loss"] for m in epoch_metrics]))
        avg_policy_loss = float(np.mean([m["policy_loss"] for m in epoch_metrics]))
        avg_kl = float(np.mean([m["kl_mean"] for m in epoch_metrics]))
        avg_ratio = float(np.mean([m["ratio_mean"] for m in epoch_metrics]))
        avg_clip_frac = float(np.mean([m["clipping_fraction"] for m in epoch_metrics]))

        step_report = {
            "policy_version": self.policy_version,
            "reward_mean": reward_mean,
            "reward_std": reward_std,
            "loss": avg_loss,
            "policy_loss": avg_policy_loss,
            "kl_mean": avg_kl,
            "ratio_mean": avg_ratio,
            "clipping_fraction": avg_clip_frac,
        }

        return step_report
