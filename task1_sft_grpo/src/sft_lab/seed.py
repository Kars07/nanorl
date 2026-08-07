"""Determinism helper and random seed setter."""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic_cudnn: bool = True) -> None:
    """Set random seed across python, numpy, torch, and CUDA.

    Note on nondeterminism:
    - CUDA ops such as scatter/gather or specific GEMM implementations may still be non-deterministic.
    - Setting torch.use_deterministic_algorithms(True) can cause errors for unsupported CUDA kernels.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_cudnn:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def generate_environment_report(
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    tokenizer_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    seed: int = 42,
    dtype: str = "float32",
    model_revision: str | None = None,
) -> dict:
    """Generate environment details as a dictionary."""
    import platform
    import sys

    import transformers

    cuda_available = torch.cuda.is_available()
    gpu_model = None
    gpu_count = 0
    compute_capability = None
    cuda_version = None

    if cuda_available:
        gpu_count = torch.cuda.device_count()
        gpu_model = torch.cuda.get_device_name(0)
        cc = torch.cuda.get_device_capability(0)
        compute_capability = f"{cc[0]}.{cc[1]}"
        cuda_version = torch.version.cuda

    report = {
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_available": cuda_available,
        "cuda_version": cuda_version,
        "gpu_model": gpu_model,
        "gpu_count": gpu_count,
        "compute_capability": compute_capability,
        "model_id": model_id,
        "tokenizer_id": tokenizer_id,
        "model_revision": model_revision or "main",
        "dtype": dtype,
        "seed": seed,
        "os": platform.platform(),
    }
    return report
