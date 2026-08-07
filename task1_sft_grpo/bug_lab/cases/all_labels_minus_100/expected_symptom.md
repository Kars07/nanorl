# Expected Symptom: all_labels_minus_100

- **What was broken**: Injected bug `all_labels_minus_100`.
- **Observable Metric**: Specific failure in labels, loss, or generation.
- **Why aggregate loss can be misleading**: Aggregate loss may appear to decrease even when training is broken.
- **Prevention Test**: `tests/test_bug_injections.py::test_bug_all_labels_minus_100`
