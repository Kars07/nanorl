"""Reproduce bug case: incorrect_label_shifting"""


def reproduce_bug():
    """Returns bug injection metadata for incorrect_label_shifting."""
    return {"bug_name": "incorrect_label_shifting", "injected": True}


if __name__ == "__main__":
    print(reproduce_bug())
