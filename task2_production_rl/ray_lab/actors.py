"""Ray worker and actor implementations for production RL lab."""

import time
from typing import Any, Dict, List, Optional

import numpy as np
import ray


@ray.remote
class PolicyWorker:
    """Ray Actor representing a Policy Inference Worker."""

    def __init__(self, version: int = 0, initial_weights: Optional[np.ndarray] = None):
        self.version = version
        self.weights = initial_weights if initial_weights is not None else np.ones((10,), dtype=np.float32)

    def get_version(self) -> int:
        return self.version

    def get_weight_fingerprint(self) -> float:
        return float(np.sum(self.weights))

    def update_weights(self, new_weights: np.ndarray, new_version: int) -> int:
        self.weights = new_weights.copy()
        self.version = new_version
        return self.version

    def generate(self, prompt: str, delay_seconds: float = 0.0) -> Dict[str, Any]:
        if delay_seconds > 0:
            time.sleep(delay_seconds)
        # Mock completion generation
        completion = f"Answer to '{prompt}' with weight sum {np.sum(self.weights):.2f}"
        return {
            "prompt": prompt,
            "completion": completion,
            "policy_version": self.version,
            "weight_fingerprint": float(np.sum(self.weights)),
        }


@ray.remote
class RolloutWorker:
    """Ray Actor representing a Rollout Generation Worker."""

    def __init__(self, worker_id: str):
        self.worker_id = worker_id
        self.rollout_count = 0

    def collect_rollout(
        self,
        policy_actor: Any,
        prompt: str,
        delay_seconds: float = 0.0,
    ) -> Dict[str, Any]:
        self.rollout_count += 1
        # Async call to policy actor
        gen_res = ray.get(policy_actor.generate.remote(prompt, delay_seconds=delay_seconds))
        return {
            "worker_id": self.worker_id,
            "rollout_id": f"{self.worker_id}_{self.rollout_count}",
            "prompt": prompt,
            "completion": gen_res["completion"],
            "policy_version": gen_res["policy_version"],
            "weight_fingerprint": gen_res["weight_fingerprint"],
        }


@ray.remote
class RewardWorker:
    """Ray Actor representing a Reward Evaluation Worker."""

    def __init__(self):
        self.evaluated_count = 0

    def compute_reward(self, rollout_item: Dict[str, Any]) -> Dict[str, Any]:
        self.evaluated_count += 1
        completion = rollout_item.get("completion", "")
        # Mock reward scoring
        reward = 1.0 if "weight sum" in completion else 0.0
        result = dict(rollout_item)
        result["reward"] = reward
        return result


@ray.remote
class LearnerWorker:
    """Ray Actor representing a Central Learner Worker."""

    def __init__(self, initial_version: int = 0):
        self.current_version = initial_version
        self.weights = np.ones((10,), dtype=np.float32)
        self.processed_rollouts = 0
        self.stale_rollouts = 0

    def get_version(self) -> int:
        return self.current_version

    def get_weights(self) -> np.ndarray:
        return self.weights

    def update_policy(self, rollout_batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        self.processed_rollouts += len(rollout_batch)

        lags = []
        for item in rollout_batch:
            r_ver = item.get("policy_version", 0)
            lag = self.current_version - r_ver
            lags.append(lag)
            if lag > 0:
                self.stale_rollouts += 1

        # Simulate SGD update: increment weights and bump policy version
        self.weights = self.weights + 0.1
        self.current_version += 1

        return {
            "new_version": self.current_version,
            "avg_lag": float(np.mean(lags)) if lags else 0.0,
            "max_lag": int(np.max(lags)) if lags else 0,
            "processed_count": self.processed_rollouts,
            "stale_count": self.stale_rollouts,
        }
