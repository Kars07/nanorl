# Deliberate SFT Bug Laboratory

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
