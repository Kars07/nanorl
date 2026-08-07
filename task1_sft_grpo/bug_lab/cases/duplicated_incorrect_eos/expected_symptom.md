# Expected Symptom: duplicated_incorrect_eos

- **What was broken**: Injected bug `duplicated_incorrect_eos`.
- **Observable Metric**: Specific failure in labels, loss, or generation.
- **Why aggregate loss can be misleading**: Aggregate loss may appear to decrease even when training is broken.
- **Prevention Test**: `tests/test_bug_injections.py::test_bug_duplicated_incorrect_eos`
