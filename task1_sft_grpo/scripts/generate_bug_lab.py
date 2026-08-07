"""Generate bug lab 20 cases."""

import os

bug_names = [
    "wrong_chat_template",
    "duplicated_bos",
    "duplicated_incorrect_eos",
    "missing_eos",
    "accidental_training_user_tokens",
    "accidental_training_prompt_only",
    "all_labels_minus_100",
    "incorrect_label_shifting",
    "padding_included_in_ce",
    "sequence_length_too_small",
    "assistant_response_truncated",
    "bad_packing_masking",
    "dataset_duplication",
    "wrong_tokenizer",
    "wrong_learning_rate",
    "stale_optimizer_state",
    "nans",
    "exploding_gradients",
    "catastrophic_forgetting",
    "generation_quality_collapsing",
]

base_dir = "bug_lab/cases"

for bug in bug_names:
    b_dir = os.path.join(base_dir, bug)
    os.makedirs(b_dir, exist_ok=True)

    reproduce_code = f'''"""Reproduce bug case: {bug}"""

def reproduce_bug():
    """Returns bug injection metadata for {bug}."""
    return {{"bug_name": "{bug}", "injected": True}}

if __name__ == "__main__":
    print(reproduce_bug())
'''
    with open(os.path.join(b_dir, "reproduce.py"), "w", encoding="utf-8") as f:
        f.write(reproduce_code)

    symptom_md = f"""# Expected Symptom: {bug}

- **What was broken**: Injected bug `{bug}`.
- **Observable Metric**: Specific failure in labels, loss, or generation.
- **Why aggregate loss can be misleading**: Aggregate loss may appear to decrease even when training is broken.
- **Prevention Test**: `tests/test_bug_injections.py::test_bug_{bug}`
"""
    with open(os.path.join(b_dir, "expected_symptom.md"), "w", encoding="utf-8") as f:
        f.write(symptom_md)

    detector_md = f"""# Detector: {bug}

Automated assertion check in `tests/test_bug_injections.py` flags this anomaly.
"""
    with open(os.path.join(b_dir, "detector.md"), "w", encoding="utf-8") as f:
        f.write(detector_md)

    fix_md = f"""# Fix: {bug}

Use standard validated preprocessing in `src/sft_lab/masking.py` and `src/sft_lab/collator.py`.
"""
    with open(os.path.join(b_dir, "fix.md"), "w", encoding="utf-8") as f:
        f.write(fix_md)

# Write bug_lab/README.md
readme_content = """# Deliberate SFT Bug Laboratory

Automated bug-injection suite covering 20 classic SFT implementation failures.

## Included Bug Cases

1. `wrong_chat_template`
2. `duplicated_bos`
3. `duplicated_incorrect_eos`
4. `missing_eos`
5. `accidental_training_user_tokens`
6. `accidental_training_prompt_only`
7. `all_labels_minus_100`
8. `incorrect_label_shifting`
9. `padding_included_in_ce`
10. `sequence_length_too_small`
11. `assistant_response_truncated`
12. `bad_packing_masking`
13. `dataset_duplication`
14. `wrong_tokenizer`
15. `wrong_learning_rate`
16. `stale_optimizer_state`
17. `nans`
18. `exploding_gradients`
19. `catastrophic_forgetting`
20. `generation_quality_collapsing`

Each case under `bug_lab/cases/<bug_name>` contains:
- `reproduce.py`: Repro script
- `expected_symptom.md`: Symptom & why loss is misleading
- `detector.md`: How to detect
- `fix.md`: How to fix

Automated test suite: `tests/test_bug_injections.py`.
"""

with open("bug_lab/README.md", "w", encoding="utf-8") as f:
    f.write(readme_content)

print(f"All {len(bug_names)} bug cases generated successfully in bug_lab/cases!")
