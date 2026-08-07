"""Reproduce bug case: duplicated_bos"""


def reproduce_bug():
    """Returns bug injection metadata for duplicated_bos."""
    return {"bug_name": "duplicated_bos", "injected": True}


if __name__ == "__main__":
    print(reproduce_bug())
