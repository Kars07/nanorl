"""Reproduce bug case: stale_optimizer_state"""


def reproduce_bug():
    """Returns bug injection metadata for stale_optimizer_state."""
    return {"bug_name": "stale_optimizer_state", "injected": True}


if __name__ == "__main__":
    print(reproduce_bug())
