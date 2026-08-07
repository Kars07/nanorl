# Expected Symptom: incorrect_label_shifting

- **What was broken**: Injected bug `incorrect_label_shifting`.
- **Observable Metric**: Specific failure in labels, loss, or generation.
- **Why aggregate loss can be misleading**: Aggregate loss may appear to decrease even when training is broken.
- **Prevention Test**: `tests/test_bug_injections.py::test_bug_incorrect_label_shifting`
