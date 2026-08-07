"""Reproduce bug case: accidental_training_user_tokens"""


def reproduce_bug():
    """Returns bug injection metadata for accidental_training_user_tokens."""
    return {"bug_name": "accidental_training_user_tokens", "injected": True}


if __name__ == "__main__":
    print(reproduce_bug())
