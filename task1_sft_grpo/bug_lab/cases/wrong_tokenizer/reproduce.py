"""Reproduce bug case: wrong_tokenizer"""


def reproduce_bug():
    """Returns bug injection metadata for wrong_tokenizer."""
    return {"bug_name": "wrong_tokenizer", "injected": True}


if __name__ == "__main__":
    print(reproduce_bug())
