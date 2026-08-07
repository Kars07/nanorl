"""Prime-RL environment verifiers for mathematical exact-match reasoning."""

import re


def compute_math_reward(completion_text: str, target_answer: str) -> float:
    """Exact-match numeric verifier for math reasoning."""
    target = target_answer.strip()
    match = re.search(r"####\s*(-?\d+(?:\.\d+)?)", completion_text)

    if match:
        extracted = match.group(1).strip()
        if extracted == target:
            return 1.0

    numbers = re.findall(r"-?\d+(?:\.\d+)?", completion_text)
    if numbers and numbers[-1].strip() == target:
        return 0.5

    return 0.0
