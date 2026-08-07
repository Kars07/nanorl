"""Reproduce bug case: assistant_response_truncated"""


def reproduce_bug():
    """Returns bug injection metadata for assistant_response_truncated."""
    return {"bug_name": "assistant_response_truncated", "injected": True}


if __name__ == "__main__":
    print(reproduce_bug())
