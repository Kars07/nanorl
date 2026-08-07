"""Reproduce bug case: dataset_duplication"""


def reproduce_bug():
    """Returns bug injection metadata for dataset_duplication."""
    return {"bug_name": "dataset_duplication", "injected": True}


if __name__ == "__main__":
    print(reproduce_bug())
