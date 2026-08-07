# SFT Learner Walkthrough — From Conversation to Parameter Updates

This document teaches the complete mathematical and tensor data pipeline of Supervised Fine-Tuning (SFT) using `Qwen/Qwen2.5-0.5B-Instruct`.

---

## 1. Raw Conversation to Tokens

A conversation in ChatML format:

```json
[
  {"role": "user", "content": "What is 2 + 2?"},
  {"role": "assistant", "content": "2 + 2 = 4."}
]
```

Is rendered by `tokenizer.apply_chat_template` into:

```text
<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is 2 + 2?<|im_end|>\n<|im_start|>assistant\n2 + 2 = 4.<|im_end|>\n
```

When tokenized:
- `<|im_start|>` $\rightarrow$ Token ID `151644`
- `assistant` $\rightarrow$ Token ID `77091`
- `2 + 2 = 4.` $\rightarrow$ Token IDs `[17, 483, 17, 284, 19, 13]`
- `<|im_end|>` $\rightarrow$ Token ID `151645`

---

## 2. Assistant Spans to Loss Labels & Why `-100` Works

PyTorch `torch.nn.functional.cross_entropy` accepts an `ignore_index` parameter, defaulted to `-100`. Any label set to `-100` is completely ignored during loss summation and gradient computation.

In our laboratory (`src/sft_lab/masking.py`):
- All system, user, and header tokens receive label `-100`.
- Only assistant response tokens (`2 + 2 = 4.`) and terminating `<|im_end|>` receive non-`-100` target labels.

Visual label mapping:

| Token | ID | Role | Target Label | Trained? | Reason |
| --- | --- | --- | --- | --- | --- |
| `<|im_start|>` | 151644 | user | -100 | NO | User prompt token |
| `user` | 872 | user | -100 | NO | Header |
| `What...` | ... | user | -100 | NO | User prompt token |
| `<|im_start|>` | 151644 | assistant | -100 | NO | Assistant header |
| `assistant` | 77091 | assistant | -100 | NO | Assistant header |
| `2` | 17 | assistant | 17 | YES | Assistant content |
| `+` | 483 | assistant | 483 | YES | Assistant content |
| `2` | 17 | assistant | 17 | YES | Assistant content |
| `=` | 284 | assistant | 284 | YES | Assistant content |
| `4` | 19 | assistant | 19 | YES | Assistant content |
| `.` | 13 | assistant | 13 | YES | Assistant content |
| `<|im_end|>` | 151645 | assistant | 151645 | YES | Assistant EOS (learn to stop) |

---

## 3. Causal Shift & Cross-Entropy

Causal language models predict the next token $x_{t+1}$ given prefix $x_{\le t}$.

Therefore:
- `logits[:, :-1, :]` (predictions at position $0 \dots T-2$)
- `labels[:, 1:]` (targets at position $1 \dots T-1$)

Per-token Cross-Entropy:

$$\text{CE}(x_t) = -\log \frac{\exp(z_{\text{target}})}{\sum_j \exp(z_j)}$$

Total batch loss is the mean of non-`-100` per-token cross-entropies:

$$\mathcal{L} = \frac{1}{N_{\text{supervised}}} \sum_{i \in \text{supervised}} \text{CE}(x_i)$$

---

## 4. Why Lower Average Loss Can Hide a Broken Model

If a dataset contains 90% prompt tokens and 10% assistant tokens, and loss is computed across all tokens (without masking), loss will drop dramatically as the model easily predicts repetitive user header patterns (`<|im_start|>user\n`), while failing completely on assistant reasoning!

This is why our dataset inspector (`scripts/inspect_dataset.py`) and loss mask inspector (`scripts/inspect_loss_mask.py`) explicitly verify that only assistant content is supervised.

---

## 5. TransformerLens Compatibility Note

TransformerLens was tested for `Qwen/Qwen2.5-0.5B-Instruct`. Since official TransformerLens architecture hooks for Qwen2.5 require custom layer wrappers, native PyTorch hooks (`src/sft_lab/hooks.py`) serve as the authoritative activation and gradient inspection mechanism in this laboratory.
