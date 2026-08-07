# Expected Symptom: sequence_length_too_small

- **What was broken**: Injected bug `sequence_length_too_small`.
- **Observable Metric**: Specific failure in labels, loss, or generation.
- **Why aggregate loss can be misleading**: Aggregate loss may appear to decrease even when training is broken.
- **Prevention Test**: `tests/test_bug_injections.py::test_bug_sequence_length_too_small`
