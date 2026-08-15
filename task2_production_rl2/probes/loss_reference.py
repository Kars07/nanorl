from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DefaultLossResult:
    loss: float
    importance_ratio: list[float]
    kept: list[bool]


def grpo_advantages(rewards: list[float]) -> list[float]:
    """Pinned Prime GRPO: reward minus group mean, without std normalization."""
    if not rewards:
        raise ValueError("reward group cannot be empty")
    mean = sum(rewards) / len(rewards)
    return [reward - mean for reward in rewards]


def default_loss(
    trainer_logprobs: list[float],
    inference_logprobs: list[float],
    advantages: list[float],
    loss_mask: list[bool],
    *,
    dppo_mask_high: float,
    dppo_mask_low: float,
    adv_tau: float,
    kl_tau: float,
    loss_weights: list[float] | None = None,
) -> DefaultLossResult:
    """Independent scalar reproduction of pinned `default_loss_fn`."""
    size = len(trainer_logprobs)
    values = [inference_logprobs, advantages, loss_mask]
    if any(len(value) != size for value in values):
        raise ValueError("all token streams must have equal length")
    if loss_weights is not None and len(loss_weights) != size:
        raise ValueError("loss_weights must align with token streams")
    ratios: list[float] = []
    kept: list[bool] = []
    loss = 0.0
    for idx, (trainer, inference, advantage, trainable) in enumerate(
        zip(trainer_logprobs, inference_logprobs, advantages, loss_mask, strict=True)
    ):
        log_ratio = trainer - inference
        ratio = math.exp(log_ratio)
        probability_delta = math.exp(trainer) - math.exp(inference)
        invalid = (
            probability_delta > dppo_mask_high
            if advantage > 0
            else probability_delta < -dppo_mask_low
        )
        keep = trainable and not invalid
        per_token = -(adv_tau * advantage * ratio if keep else 0.0)
        if trainable:
            per_token += kl_tau * log_ratio**2
        if loss_weights is not None:
            per_token *= loss_weights[idx]
        loss += per_token
        ratios.append(ratio)
        kept.append(keep)
    return DefaultLossResult(loss=loss, importance_ratio=ratios, kept=kept)
