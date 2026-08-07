"""Group-relative advantage computation for GRPO."""

from typing import Dict, List

import numpy as np
import torch


def compute_group_relative_advantages(
    rewards: List[float],
    group_ids: List[str],
    use_std_normalization: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute group-relative advantages for GRPO.

    Args:
        rewards: List of scalar rewards for each completion in the batch
        group_ids: List of group IDs identifying which prompt each completion belongs to
        use_std_normalization: Whether to divide by group standard deviation
        eps: Numerical stability constant

    Returns:
        (B * G, 1) FloatTensor of advantages (detached from grad graph)
    """
    assert len(rewards) == len(group_ids), (
        f"Mismatch between rewards len ({len(rewards)}) and group_ids len ({len(group_ids)})"
    )

    # Group rewards by group_id
    grouped_rewards: Dict[str, List[int]] = {}
    for idx, gid in enumerate(group_ids):
        if gid not in grouped_rewards:
            grouped_rewards[gid] = []
        grouped_rewards[gid].append(idx)

    advantages = np.zeros(len(rewards), dtype=np.float32)

    for gid, indices in grouped_rewards.items():
        if len(indices) <= 1:
            # Group size 1 provides no relative advantage signal
            for idx in indices:
                advantages[idx] = 0.0
            continue

        g_rewards = np.array([rewards[i] for i in indices], dtype=np.float32)
        g_mean = float(np.mean(g_rewards))
        g_std = float(np.std(g_rewards))

        if use_std_normalization:
            if g_std < eps:
                # Constant reward group: no variance
                for idx in indices:
                    advantages[idx] = 0.0
            else:
                for idx in indices:
                    advantages[idx] = (rewards[idx] - g_mean) / (g_std + eps)
        else:
            for idx in indices:
                advantages[idx] = rewards[idx] - g_mean

    adv_tensor = torch.tensor(advantages, dtype=torch.float32).unsqueeze(-1)
    assert not adv_tensor.requires_grad, "Advantages must not require gradients!"
    return adv_tensor
