"""Data types and containers for GRPO rollouts and batches."""

from dataclasses import dataclass
from typing import List

import torch


@dataclass
class RolloutItem:
    prompt_id: str
    completion_id: str
    group_id: str
    prompt_text: str
    completion_text: str
    token_ids: torch.Tensor  # 1D LongTensor (prompt + completion)
    completion_mask: torch.Tensor  # 1D LongTensor (0 for prompt/padding, 1 for completion)
    old_logprobs: torch.Tensor  # 1D FloatTensor (per completion token)
    reference_logprobs: torch.Tensor  # 1D FloatTensor (per completion token)
    reward: float = 0.0
    advantage: float = 0.0
    policy_version: int = 0


@dataclass
class RolloutBatch:
    input_ids: torch.Tensor  # (B * G, S)
    attention_mask: torch.Tensor  # (B * G, S)
    completion_mask: torch.Tensor  # (B * G, S - 1)
    old_logprobs: torch.Tensor  # (B * G, S - 1)
    reference_logprobs: torch.Tensor  # (B * G, S - 1)
    advantages: torch.Tensor  # (B * G, 1) or (B * G, S - 1)
    group_ids: List[str]
    rewards: List[float]
