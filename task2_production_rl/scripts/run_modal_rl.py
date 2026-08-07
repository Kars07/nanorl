"""Modal remote multi-GPU runner for Task 2 production RL tests."""

import modal

app = modal.App("task2-production-rl-runner")

# Build image with PyTorch, Ray, Transformers, vLLM, and Task 2 dependencies
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.0.0",
    "transformers>=4.40.0",
    "datasets>=2.14.0",
    "ray[default]>=2.10.0",
    "pydantic>=2.0.0",
    "pytest>=7.0.0",
    "rich>=13.0.0",
)


@app.function(
    image=image,
    gpu="A10G:2",
    timeout=600,
)
def run_remote_multi_gpu_test():
    import ray
    import torch

    print("=== MODAL REMOTE MULTI-GPU RUNNER ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"GPU Count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    ray.init(ignore_reinit_error=True)
    print("Ray multi-GPU cluster initialized successfully on Modal!")

    return {
        "status": "SUCCESS",
        "gpu_count": torch.cuda.device_count(),
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "ray_version": ray.__version__,
    }


@app.local_entrypoint()
def main():
    print("Launching multi-GPU Modal job...")
    result = run_remote_multi_gpu_test.remote()
    print("Remote execution result:", result)


if __name__ == "__main__":
    main()
