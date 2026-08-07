"""Reproduce bug case: catastrophic_forgetting"""


def reproduce_bug():
    """Returns bug injection metadata for catastrophic_forgetting."""
    return {"bug_name": "catastrophic_forgetting", "injected": True}


if __name__ == "__main__":
    print(reproduce_bug())
