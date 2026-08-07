"""Group Relative Policy Optimization (GRPO) advantage & sequence packing algorithm."""

from typing import Any, Dict, List

import numpy as np


class GRPOAlgorithm:
    """Computes group-relative baseline advantages and trajectory packing."""

    def __init__(self, eps: float = 1e-8, use_std_normalization: bool = True):
        self.eps = eps
        self.use_std_normalization = use_std_normalization

    def compute_advantages(self, rewards: List[float]) -> List[float]:
        rewards_arr = np.array(rewards, dtype=np.float32)
        mean_r = np.mean(rewards_arr)

        if self.use_std_normalization:
            std_r = np.std(rewards_arr)
            advs = (rewards_arr - mean_r) / (std_r + self.eps)
        else:
            advs = rewards_arr - mean_r

        return advs.tolist()

    def pack_trajectories(self, trajectories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Pack trajectories into contiguous sequence representations."""
        packed_items = []

        for item in trajectories:
            full_toks = item["full_tokens"]
            prompt_len = len(item["prompt_tokens"])
            seq_len = len(full_toks)
            adv = item["advantage"]

            comp_mask = [0] * (seq_len - 1)
            for i in range(prompt_len - 1, seq_len - 1):
                comp_mask[i] = 1

            packed_items.append(
                {
                    "input_ids": full_toks,
                    "completion_mask": comp_mask,
                    "advantage": adv,
                    "prompt_len": prompt_len,
                    "seq_len": seq_len,
                }
            )

        return {
            "samples": packed_items,
            "num_samples": len(packed_items),
        }
