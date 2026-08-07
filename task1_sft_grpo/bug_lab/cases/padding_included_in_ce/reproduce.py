"""Reproduce bug case: padding_included_in_ce"""


def reproduce_bug():
    """Returns bug injection metadata for padding_included_in_ce."""
    return {"bug_name": "padding_included_in_ce", "injected": True}


if __name__ == "__main__":
    print(reproduce_bug())
