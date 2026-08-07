"""Reproduce bug case: sequence_length_too_small"""


def reproduce_bug():
    """Returns bug injection metadata for sequence_length_too_small."""
    return {"bug_name": "sequence_length_too_small", "injected": True}


if __name__ == "__main__":
    print(reproduce_bug())
