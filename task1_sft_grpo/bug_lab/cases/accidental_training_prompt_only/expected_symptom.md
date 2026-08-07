# Expected Symptom: accidental_training_prompt_only

- **What was broken**: Injected bug `accidental_training_prompt_only`.
- **Observable Metric**: Specific failure in labels, loss, or generation.
- **Why aggregate loss can be misleading**: Aggregate loss may appear to decrease even when training is broken.
- **Prevention Test**: `tests/test_bug_injections.py::test_bug_accidental_training_prompt_only`
