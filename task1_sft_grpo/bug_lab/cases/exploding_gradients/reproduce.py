"""Reproduce bug case: exploding_gradients"""


def reproduce_bug():
    """Returns bug injection metadata for exploding_gradients."""
    return {"bug_name": "exploding_gradients", "injected": True}


if __name__ == "__main__":
    print(reproduce_bug())
