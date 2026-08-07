"""Reproduce bug case: accidental_training_prompt_only"""


def reproduce_bug():
    """Returns bug injection metadata for accidental_training_prompt_only."""
    return {"bug_name": "accidental_training_prompt_only", "injected": True}


if __name__ == "__main__":
    print(reproduce_bug())
