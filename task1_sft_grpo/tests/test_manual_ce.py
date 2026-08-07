"""Unit tests for manual CE loss calculation and model loss numerical parity."""

import torch

from sft_lab.collator import SFTDataCollator
from sft_lab.dataset import SFTDataset
from sft_lab.metrics import compute_manual_causal_lm_loss
from sft_lab.model import load_model_and_tokenizer


def test_manual_ce_assertions():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model_and_tokenizer("Qwen/Qwen2.5-0.5B-Instruct", dtype="float32", device=device)
    model.eval()

    dataset = SFTDataset("data/fixtures/valid.jsonl", tokenizer, max_seq_length=256)
    collator = SFTDataCollator(tokenizer, max_seq_length=256)
    batch = collator([dataset[0]])

    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    with torch.no_grad():
        out = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask, use_cache=False)
        logits = out.logits
        model_loss = out.loss.item()

    # Logits assertions
    assert logits.dim() == 3, f"Logits rank must be 3, got {logits.dim()}"
    b, seq, vocab = logits.shape
    assert (b, seq) == input_ids.shape
    assert vocab >= tokenizer.vocab_size

    manual_loss = compute_manual_causal_lm_loss(logits, labels).item()

    # Numerical parity tolerance: <= 1e-4
    diff = abs(model_loss - manual_loss)
    assert diff < 1e-4, f"Model loss ({model_loss}) and manual CE ({manual_loss}) mismatch: diff {diff}"


def test_ignored_label_sensitivity():
    """Verify changing an ignored label (-100) does not change CE, but changing a supervised label does."""
    logits = torch.randn(1, 10, 100)
    labels = torch.tensor([[-100, -100, -100, 5, 12, 45, -100, -100, -100, -100]])

    loss_orig = compute_manual_causal_lm_loss(logits, labels).item()

    # Mutate ignored position
    labels_mut_ignored = labels.clone()
    labels_mut_ignored[0, 1] = 99  # was -100
    labels_mut_ignored[0, 1] = -100
    loss_mut_ignored = compute_manual_causal_lm_loss(logits, labels_mut_ignored).item()

    assert loss_orig == loss_mut_ignored, "Mutating ignored position should not change CE loss"

    # Mutate supervised target
    labels_mut_sup = labels.clone()
    labels_mut_sup[0, 3] = 99  # target was 5
    loss_mut_sup = compute_manual_causal_lm_loss(logits, labels_mut_sup).item()

    assert loss_orig != loss_mut_sup, "Mutating supervised target must change CE loss"
