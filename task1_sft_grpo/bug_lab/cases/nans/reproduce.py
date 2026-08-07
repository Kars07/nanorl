"""Reproduce bug case: nans"""


def reproduce_bug():
    """Returns bug injection metadata for nans."""
    return {"bug_name": "nans", "injected": True}


if __name__ == "__main__":
    print(reproduce_bug())
