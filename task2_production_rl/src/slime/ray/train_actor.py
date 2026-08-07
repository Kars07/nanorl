"""Megatron-style distributed trainer rank Ray actor for slime."""

from typing import Any, Dict, List

import numpy as np
import ray


@ray.remote
class TrainActor:
    """Ray actor representing a Megatron-style training rank."""

    def __init__(self, rank: int = 0):
        self.rank = rank
        self.version = 0
        self.weights = np.ones((10,), dtype=np.float32)

    def train_step(self, mini_batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        self.version += 1
        self.weights += 0.05
        return {
            "rank": self.rank,
            "version": self.version,
            "batch_size": len(mini_batch),
            "weight_sum": float(np.sum(self.weights)),
        }
