"""Data collator for SFT batching with dynamic padding."""

from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from transformers import PreTrainedTokenizer

from sft_lab.masking import build_sft_labels_and_metadata


@dataclass
class SFTDataCollator:
    """Collates list of conversation records into padded PyTorch tensors."""

    tokenizer: PreTrainedTokenizer
    max_seq_length: int = 512
    assistant_only_loss: bool = True
    pad_to_multiple_of: int | None = None

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids_list = []
        labels_list = []
        attention_mask_list = []

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        for item in batch:
            messages = item["messages"]
            processed = build_sft_labels_and_metadata(
                messages=messages,
                tokenizer=self.tokenizer,
                max_seq_length=self.max_seq_length,
                assistant_only_loss=self.assistant_only_loss,
            )
            input_ids_list.append(processed["input_ids"])
            labels_list.append(processed["labels"])
            attention_mask_list.append(processed["attention_mask"])

        # Determine batch max sequence length
        batch_max_len = max(len(ids) for ids in input_ids_list)
        if self.pad_to_multiple_of is not None and self.pad_to_multiple_of > 0:
            batch_max_len = (
                (batch_max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of
            ) * self.pad_to_multiple_of

        batch_input_ids = []
        batch_labels = []
        batch_attention_mask = []

        for ids, lbls, mask in zip(input_ids_list, labels_list, attention_mask_list):
            pad_len = batch_max_len - len(ids)
            padded_ids = ids + [pad_token_id] * pad_len
            padded_lbls = lbls + [-100] * pad_len
            padded_mask = mask + [0] * pad_len

            batch_input_ids.append(padded_ids)
            batch_labels.append(padded_lbls)
            batch_attention_mask.append(padded_mask)

        input_ids_tensor = torch.tensor(batch_input_ids, dtype=torch.long)
        labels_tensor = torch.tensor(batch_labels, dtype=torch.long)
        attention_mask_tensor = torch.tensor(batch_attention_mask, dtype=torch.long)

        # Basic shape and label assertions
        assert input_ids_tensor.shape == labels_tensor.shape == attention_mask_tensor.shape, (
            f"Shape mismatch: input_ids {input_ids_tensor.shape}, labels {labels_tensor.shape}, attention_mask {attention_mask_tensor.shape}"
        )

        return {
            "input_ids": input_ids_tensor,
            "labels": labels_tensor,
            "attention_mask": attention_mask_tensor,
        }
