# SFT & GRPO Debugging Playbook

This playbook lists common symptoms, diagnostic commands, and immediate fixes.

## 1. Symptom: Model output collapses to empty strings or repetitive tokens
- **Likely Cause**: Assistant loss mask included user tokens or system tokens, or `<|im_end|>` was missing during training.
- **Diagnostic Command**:
  ```bash
  python scripts/inspect_template.py --data_path data/fixtures/valid.jsonl
  python scripts/inspect_loss_mask.py --data_path data/fixtures/valid.jsonl
  ```
- **Fix**: Check `src/sft_lab/masking.py` to ensure only assistant tokens receive non-`-100` labels.

## 2. Symptom: CUDA Out of Memory (OOM)
- **Likely Cause**: Batch size or sequence length too large for GPU VRAM (e.g. 6GB).
- **Diagnostic Command**: Check `environment.json` GPU capacity.
- **Fix**:
  Set `batch_size: 1`, `gradient_accumulation_steps: 4`, and `use_gradient_checkpointing=True` in `load_model_and_tokenizer`.

## 3. Symptom: Loss is NaN or Inf
- **Likely Cause**: Learning rate too high, gradient explosion, or mixed precision underflow in FP16.
- **Fix**: Switch `dtype` to `bfloat16`, reduce learning rate to `2e-5`, and ensure `grad_clip: 1.0` is active.

## 4. Symptom: GRPO Policy Ratio is 100% Clipped
- **Likely Cause**: Learning rate too large or too many optimization epochs $K$ per rollout batch.
- **Fix**: Set `learning_rate: 1e-6` and `num_epochs: 1` in `configs/grpo_debug.yaml`.
