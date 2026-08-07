# SFT Acceptance Gate Report

Date: 2026-08-07
Model: Qwen/Qwen2.5-0.5B-Instruct

## Acceptance Criteria Status

| Criterion | Status | Notes / Evidence |
| --- | --- | --- |
| **Dataset Inspector** | **PASS** | `scripts/inspect_dataset.py` passes all structural & fixture checks (`data/fixtures/test_fixtures.jsonl`) |
| **Template Microscope** | **PASS** | `scripts/inspect_template.py` verifies ChatML rendering and token-by-token loss mask |
| **Assistant Labels Visual Verification** | **PASS** | Only assistant content + terminating `<|im_end|>` are assigned non-`-100` labels |
| **Manual CE Parity** | **PASS** | `tests/test_manual_ce.py` confirms manual CE matches PyTorch model CE within `< 1e-4` |
| **Per-Token Loss Decomposition** | **PASS** | `scripts/inspect_logits.py` decomposes target logprobs, entropy, and top-5 tokens |
| **Batch Inspector** | **PASS** | `scripts/inspect_batch.py` asserts shapes and valid token label ranges (`-100` or `0 <= label < vocab_size`) |
| **Tiny Dataset Overfit** | **PASS** | `scripts/overfit_test.py` drops loss from 0.6990 to 0.0004 and reproduces target generation |
| **Checkpoint Save/Reload** | **PASS** | `compare_checkpoints.py` & `overfit_test.py` verify reloaded checkpoint reproduces exact generations |
| **Bug Injection Suite** | **PASS** | All 20 bug cases in `bug_lab/cases/` detected by `tests/test_bug_injections.py` |
| **PyTest Suite** | **PASS** | All unit tests pass in `pytest -q` |
| **Reproduction Instructions** | **PASS** | `README.md` contains exact execution commands |

## Conclusion

**SFT Acceptance Gate: PASSED.**
Proceeding to Phase M — GRPO implementation.
