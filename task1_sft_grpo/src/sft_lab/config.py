"""Typed and validated configurations for SFT and GRPO experiments."""

import os

import yaml
from pydantic import BaseModel, Field, model_validator


class SFTConfig(BaseModel):
    """Validation schema for SFT configuration."""

    model_id: str = Field("Qwen/Qwen2.5-0.5B-Instruct", description="HF Model ID")
    tokenizer_id: str = Field("Qwen/Qwen2.5-0.5B-Instruct", description="HF Tokenizer ID")
    seed: int = Field(42, ge=0, description="Random seed")
    max_seq_length: int = Field(512, gt=0, description="Maximum sequence length")
    dataset_name_or_path: str = Field(..., description="Dataset path or HF dataset name")
    train_split: str = Field("train", description="Dataset train split name")
    eval_split: str = Field("eval", description="Dataset eval split name")
    assistant_only_loss: bool = Field(True, description="Whether to mask non-assistant tokens in loss")
    batch_size: int = Field(2, gt=0, description="Per-device batch size")
    gradient_accumulation_steps: int = Field(1, gt=0, description="Gradient accumulation steps")
    learning_rate: float = Field(2e-5, gt=0.0, description="Learning rate")
    weight_decay: float = Field(0.01, ge=0.0, description="Weight decay")
    num_epochs: int = Field(3, gt=0, description="Number of training epochs")
    warmup_ratio: float = Field(0.05, ge=0.0, le=1.0, description="Warmup ratio")
    output_dir: str = Field("artifacts/checkpoints/sft", description="Output directory for checkpoints")
    dtype: str = Field("bfloat16", description="Model dtype (float32, float16, bfloat16)")
    grad_clip: float = Field(1.0, ge=0.0, description="Gradient clipping norm")
    allow_zero_supervised_tokens: bool = Field(False, description="Allow examples with 0 supervised tokens")

    @model_validator(mode="after")
    def validate_config(self) -> "SFTConfig":
        if self.dtype not in ["float32", "float16", "bfloat16"]:
            raise ValueError(f"Unsupported dtype: {self.dtype}")
        return self

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "SFTConfig":
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def save_yaml(self, yaml_path: str) -> None:
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.model_dump(), f)


class GRPOConfig(BaseModel):
    """Validation schema for GRPO configuration."""

    model_id: str = Field("Qwen/Qwen2.5-0.5B-Instruct", description="Policy Model ID")
    ref_model_id: str = Field("Qwen/Qwen2.5-0.5B-Instruct", description="Reference Model ID")
    seed: int = Field(42, ge=0, description="Random seed")
    max_prompt_length: int = Field(256, gt=0, description="Maximum prompt sequence length")
    max_completion_length: int = Field(128, gt=0, description="Maximum completion sequence length")
    num_generations: int = Field(4, ge=2, description="Number of completions G per prompt")
    temperature: float = Field(0.7, gt=0.0, description="Sampling temperature")
    top_p: float = Field(1.0, gt=0.0, le=1.0, description="Top-p sampling")
    learning_rate: float = Field(1e-6, gt=0.0, description="Learning rate")
    num_epochs: int = Field(1, gt=0, description="Optimization epochs K per rollout batch")
    batch_size: int = Field(2, gt=0, description="Prompt batch size")
    gradient_accumulation_steps: int = Field(1, gt=0, description="Gradient accumulation steps")
    clip_eps: float = Field(0.2, gt=0.0, lt=1.0, description="PPO/GRPO clipping epsilon")
    kl_coeff: float = Field(0.04, ge=0.0, description="KL penalty coefficient")
    use_std_normalization: bool = Field(True, description="Normalize advantages by group std")
    output_dir: str = Field("artifacts/checkpoints/grpo", description="Output directory")
    dtype: str = Field("bfloat16", description="Model dtype")
    grad_clip: float = Field(1.0, ge=0.0, description="Gradient clipping norm")

    @model_validator(mode="after")
    def validate_config(self) -> "GRPOConfig":
        if self.dtype not in ["float32", "float16", "bfloat16"]:
            raise ValueError(f"Unsupported dtype: {self.dtype}")
        return self

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "GRPOConfig":
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def save_yaml(self, yaml_path: str) -> None:
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.model_dump(), f)
