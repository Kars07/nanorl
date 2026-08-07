# Expected Symptom: accidental_training_user_tokens

- **What was broken**: Injected bug `accidental_training_user_tokens`.
- **Observable Metric**: Specific failure in labels, loss, or generation.
- **Why aggregate loss can be misleading**: Aggregate loss may appear to decrease even when training is broken.
- **Prevention Test**: `tests/test_bug_injections.py::test_bug_accidental_training_user_tokens`
