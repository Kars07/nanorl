"""Generate environment report for task2_production_rl."""

import json
import os
import platform
import sys

import ray
import torch
import transformers


def generate_env_report():
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
        "ray_version": ray.__version__,
        "cuda_available": cuda_available,
        "cuda_version": cuda_version,
        "gpu_model": gpu_model,
        "gpu_count": gpu_count,
        "compute_capability": compute_capability,
        "os": platform.platform(),
    }

    os.makedirs("artifacts/environment", exist_ok=True)
    out_path = "artifacts/environment/environment.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Environment report saved to {out_path}")


if __name__ == "__main__":
    generate_env_report()
