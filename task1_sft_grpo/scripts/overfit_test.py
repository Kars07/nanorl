"""Overfit-before-scale test script."""

import json
import os

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from sft_lab.checkpointing import load_checkpoint, save_checkpoint
from sft_lab.collator import SFTDataCollator
from sft_lab.config import SFTConfig
from sft_lab.dataset import SFTDataset
from sft_lab.generation import generate_completions
from sft_lab.metrics import compute_manual_causal_lm_loss, decompose_per_token_logits
from sft_lab.model import load_model_and_tokenizer
from sft_lab.seed import set_seed


def main():
    config_path = "configs/sft_overfit.yaml"
    if not os.path.exists(config_path):
        config = SFTConfig(
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            tokenizer_id="Qwen/Qwen2.5-0.5B-Instruct",
            dataset_name_or_path="data/processed/overfit_subset.jsonl",
            num_epochs=30,
            batch_size=1,
            gradient_accumulation_steps=2,
            learning_rate=2e-4,
            output_dir="artifacts/checkpoints/sft_overfit",
            dtype="bfloat16",
        )
    else:
        config = SFTConfig.from_yaml(config_path)

    set_seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model {config.model_id} on {device}...")
    model, tokenizer = load_model_and_tokenizer(
        config.model_id,
        config.tokenizer_id,
        dtype=config.dtype,
        device=device,
        use_gradient_checkpointing=True,
    )

    dataset = SFTDataset(config.dataset_name_or_path, tokenizer, max_seq_length=config.max_seq_length)
    collator = SFTDataCollator(tokenizer, max_seq_length=config.max_seq_length)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collator)

    # Test prompt for before/after generation check
    test_ex = dataset[0]
    test_prompt = test_ex["messages"][0]["content"]
    test_target = test_ex["messages"][1]["content"]

    print("\n--- BEFORE TRAINING ---")
    gen_before = generate_completions(model, tokenizer, [test_prompt], max_new_tokens=64, do_sample=False)[0]
    print(f"Prompt: {repr(test_prompt)}")
    print(f"Target: {repr(test_target)}")
    print(f"Generation before: {repr(gen_before)}")

    # Initial loss & target logprobs
    eval_batch = collator([test_ex])
    eval_input = eval_batch["input_ids"].to(device)
    eval_labels = eval_batch["labels"].to(device)
    eval_mask = eval_batch["attention_mask"].to(device)

    model.eval()
    with torch.no_grad():
        out_init = model(input_ids=eval_input, attention_mask=eval_mask)
        init_loss = compute_manual_causal_lm_loss(out_init.logits, eval_labels).item()
        rows_init = decompose_per_token_logits(out_init.logits, eval_labels, eval_input, tokenizer)[0]
        init_target_logprob = rows_init["target_logprob"] if rows_init["is_supervised"] else 0.0

    print(f"Initial loss: {init_loss:.4f}, Target logprob: {init_target_logprob:.4f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    total_steps = (len(loader) // config.gradient_accumulation_steps + 1) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=5, num_training_steps=total_steps)

    model.train()
    step = 0
    final_loss = 0.0

    print("\nStarting overfit training...")
    for epoch in range(config.num_epochs):
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step_idx, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attn_mask = batch["attention_mask"].to(device)

            out = model(input_ids=input_ids, attention_mask=attn_mask)
            loss = compute_manual_causal_lm_loss(out.logits, labels)
            loss_scaled = loss / config.gradient_accumulation_steps

            if torch.isnan(loss) or torch.isinf(loss):
                raise ValueError("Loss is NaN or Inf during overfit training!")

            loss_scaled.backward()
            epoch_loss += loss.item()

            if (step_idx + 1) % config.gradient_accumulation_steps == 0 or (step_idx + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

        avg_epoch_loss = epoch_loss / len(loader)
        final_loss = avg_epoch_loss
        if (epoch + 1) % 5 == 0 or epoch == config.num_epochs - 1:
            print(f"Epoch {epoch + 1:2d}/{config.num_epochs}: Average Loss = {avg_epoch_loss:.4f}")

    # Post-training evaluation
    model.eval()
    with torch.no_grad():
        out_final = model(input_ids=eval_input, attention_mask=eval_mask)
        rows_final = decompose_per_token_logits(out_final.logits, eval_labels, eval_input, tokenizer)[0]
        final_target_logprob = rows_final["target_logprob"] if rows_final["is_supervised"] else 0.0

    gen_after = generate_completions(model, tokenizer, [test_prompt], max_new_tokens=64, do_sample=False)[0]

    print("\n--- AFTER TRAINING ---")
    print(f"Final loss: {final_loss:.4f}, Target logprob: {final_target_logprob:.4f}")
    print(f"Generation after: {repr(gen_after)}")

    # Save checkpoint
    save_checkpoint(
        model,
        tokenizer,
        config.output_dir,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config.model_dump(),
        step=step,
        epoch=config.num_epochs,
    )
    print(f"Checkpoint saved to {config.output_dir}")

    # Reload checkpoint and test reproduction
    model_reloaded, tokenizer_reloaded, _ = load_checkpoint(config.output_dir, device=device)
    gen_reloaded = generate_completions(
        model_reloaded, tokenizer_reloaded, [test_prompt], max_new_tokens=64, do_sample=False
    )[0]

    print(f"Reloaded generation: {repr(gen_reloaded)}")
    assert gen_reloaded == gen_after, f"Reloaded generation mismatch! Expected '{gen_after}', got '{gen_reloaded}'"

    # Assert overfit pass criteria
    assert final_loss < init_loss * 0.5 or final_loss < 0.25, (
        f"Overfit test failed! Starting loss: {init_loss:.4f}, Ending loss: {final_loss:.4f}"
    )

    report = {
        "init_loss": init_loss,
        "final_loss": final_loss,
        "init_target_logprob": init_target_logprob,
        "final_target_logprob": final_target_logprob,
        "gen_before": gen_before,
        "gen_after": gen_after,
        "gen_reloaded": gen_reloaded,
        "passed": True,
    }
    os.makedirs("artifacts/reports", exist_ok=True)
    with open("artifacts/reports/overfit_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n[bold green]OVERFIT TEST PASSED SUCCESSFULLY![/bold green]")


if __name__ == "__main__":
    main()
