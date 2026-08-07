"""Rollout record schema for production RL systems."""

from typing import List, Optional

from pydantic import BaseModel, Field


class RolloutRecord(BaseModel):
    """Standardized schema for rollout trajectory records."""

    prompt_id: str
    group_id: str
    policy_version: int
    prompt_tokens: List[int]
    completion_tokens: List[int]
    full_tokens: List[int]
    prompt_text: str
    completion_text: str
    reward: float
    sampling_logprobs: Optional[List[float]] = None
    trainer_logprobs: Optional[List[float]] = None
    timestamp: float = Field(default_factory=lambda: 0.0)

    @property
    def prompt_length(self) -> int:
        return len(self.prompt_tokens)

    @property
    def completion_length(self) -> int:
        return len(self.completion_tokens)

    @property
    def total_length(self) -> int:
        return len(self.full_tokens)
