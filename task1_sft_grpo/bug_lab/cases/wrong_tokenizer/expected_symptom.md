# Expected Symptom: wrong_tokenizer

- **What was broken**: Injected bug `wrong_tokenizer`.
- **Observable Metric**: Specific failure in labels, loss, or generation.
- **Why aggregate loss can be misleading**: Aggregate loss may appear to decrease even when training is broken.
- **Prevention Test**: `tests/test_bug_injections.py::test_bug_wrong_tokenizer`
