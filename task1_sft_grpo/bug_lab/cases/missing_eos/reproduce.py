"""Reproduce bug case: missing_eos"""


def reproduce_bug():
    """Returns bug injection metadata for missing_eos."""
    return {"bug_name": "missing_eos", "injected": True}


if __name__ == "__main__":
    print(reproduce_bug())
