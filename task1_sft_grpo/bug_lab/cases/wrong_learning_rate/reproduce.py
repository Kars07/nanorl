"""Reproduce bug case: wrong_learning_rate"""


def reproduce_bug():
    """Returns bug injection metadata for wrong_learning_rate."""
    return {"bug_name": "wrong_learning_rate", "injected": True}


if __name__ == "__main__":
    print(reproduce_bug())
