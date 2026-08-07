"""Reproduce bug case: wrong_chat_template"""


def reproduce_bug():
    """Returns bug injection metadata for wrong_chat_template."""
    return {"bug_name": "wrong_chat_template", "injected": True}


if __name__ == "__main__":
    print(reproduce_bug())
