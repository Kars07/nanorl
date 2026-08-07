# Expected Symptom: stale_optimizer_state

- **What was broken**: Injected bug `stale_optimizer_state`.
- **Observable Metric**: Specific failure in labels, loss, or generation.
- **Why aggregate loss can be misleading**: Aggregate loss may appear to decrease even when training is broken.
- **Prevention Test**: `tests/test_bug_injections.py::test_bug_stale_optimizer_state`
