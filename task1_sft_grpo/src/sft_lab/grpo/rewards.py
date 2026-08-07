"""Reward functions and verifiers for GRPO RL training."""

import re
from typing import List, Optional


def extract_numeric_answer(text: str) -> Optional[str]:
    """Extract numeric answer after #### or last number in text."""
    if "####" in text:
        after_hash = text.split("####")[-1].strip()
        match = re.search(r"[-+]?\d+(?:\.\d+)?", after_hash)
        if match:
            return match.group(0)

    # Fallback to last number in text
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1]

    return None


def compute_math_reward(completion_text: str, target_answer_text: str) -> float:
    """Compute exact numeric match reward for math reasoning tasks."""
    target_num = extract_numeric_answer(target_answer_text)
    pred_num = extract_numeric_answer(completion_text)

    if target_num is None or pred_num is None:
        return 0.0

    try:
        if abs(float(target_num) - float(pred_num)) < 1e-5:
            return 1.0
    except ValueError:
        pass

    return 0.0


def compute_batch_rewards(completions: List[str], targets: List[str]) -> List[float]:
    """Compute rewards for a batch of generated completions."""
    return [compute_math_reward(c, t) for c, t in zip(completions, targets)]
