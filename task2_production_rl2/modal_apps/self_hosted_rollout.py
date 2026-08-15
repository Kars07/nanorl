from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import modal

PRIME_RL_REV = "8c1f196dd39699726ee8ff52f6ee2495c5fa38df"
VERIFIERS_REV = "7251c60934d2c42af85d42a1da3da62269b7957e"
MODEL = "Qwen/Qwen3-0.6B"
PRIME_RL_DIR = "/opt/prime-rl"
ENV_DIR = "/opt/tiny_repo_repair_v1"
HARNESS_DIR = "/opt/learning_harness"
LEARNING_ENV_DIRS = {
    name: f"/opt/{name}"
    for name in (
        "tiny_terminal_v1",
        "tiny_browser_v1",
        "tiny_tooluse_v1",
        "tiny_long_horizon_v1",
        "proposer_solver_v1",
    )
}
REPO_REPAIR_RL_CONFIG = "/opt/configs/repo_repair_smoke.toml"
PRODUCTION_RL_CONFIG = "/opt/configs/repo_repair_production.toml"
GEPA_CONFIG = "/opt/configs/learning_harness_gepa.toml"
MULTI_SOURCE_CONFIG = "/opt/configs/multi_source_3_1_1.toml"
ALGORITHM_CONFIGS = {
    "max-rl": "/opt/configs/max_rl_repo_repair.toml",
    "hierarchical-grpo": "/opt/configs/hierarchical_grpo_proposer_solver.toml",
    "per-env-grpo-echo": "/opt/configs/per_environment_grpo_echo.toml",
    "echo-terminal": "/opt/configs/echo_terminal.toml",
}
BASELINE_SUMMARY = "/opt/artifacts/self_hosted_baseline_summary.json"
TRAIN_TRACE_JSONL = "/opt/artifacts/repo_repair_step1_traces.jsonl"
TASK2_RUNTIME_DIR = "/opt/task2_runtime_src"
HARBOR_RUNNER = "/opt/task2_scripts/run_harbor_e2b.py"
PYTHON = f"{PRIME_RL_DIR}/.venv/bin/python"
VLLM_ROUTER_WHEEL = (
    "https://github.com/PrimeIntellect-ai/router/releases/download/v0.1.26/"
    "vllm_router-0.1.26-cp38-abi3-manylinux_2_28_x86_64.whl"
)
FLASH_ATTN_WHEEL = (
    "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/"
    "download/v0.9.4/flash_attn-2.8.3+cu128torch2.11-"
    "cp312-cp312-linux_x86_64.whl"
)

app = modal.App("task2-self-hosted-rollout")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04", add_python="3.12"
    )
    .entrypoint([])
    .apt_install("git", "curl")
    .pip_install("uv==0.11.21")
    .run_commands(
        f"git clone https://github.com/PrimeIntellect-ai/prime-rl.git {PRIME_RL_DIR}",
        f"cd {PRIME_RL_DIR} && git checkout {PRIME_RL_REV}",
        (
            f"cd {PRIME_RL_DIR} && "
            "git -c url.https://github.com/.insteadOf=git@github.com: "
            "submodule update --init deps/prime-envs deps/pydantic-config "
            "deps/renderers deps/verifiers"
        ),
        f"cd {PRIME_RL_DIR} && uv sync --frozen --no-dev",
        f"uv pip install --python {PYTHON} --no-deps {VLLM_ROUTER_WHEEL}",
        f"uv pip install --python {PYTHON} --no-deps {FLASH_ATTN_WHEEL}",
        f"uv pip install --python {PYTHON} e2b==2.35.0",
        f"uv pip install --python {PYTHON} harbor==0.20.0",
    )
    .add_local_dir(
        "environments/tiny_repo_repair_v1", remote_path=ENV_DIR, copy=True
    )
    .add_local_dir(
        "harnesses/learning_harness", remote_path=HARNESS_DIR, copy=True
    )
    .add_local_dir(
        "environments/tiny_terminal_v1",
        remote_path=LEARNING_ENV_DIRS["tiny_terminal_v1"], copy=True,
    )
    .add_local_dir(
        "environments/tiny_browser_v1",
        remote_path=LEARNING_ENV_DIRS["tiny_browser_v1"], copy=True,
    )
    .add_local_dir(
        "environments/tiny_tooluse_v1",
        remote_path=LEARNING_ENV_DIRS["tiny_tooluse_v1"], copy=True,
    )
    .add_local_dir(
        "environments/tiny_long_horizon_v1",
        remote_path=LEARNING_ENV_DIRS["tiny_long_horizon_v1"], copy=True,
    )
    .add_local_dir(
        "environments/proposer_solver_v1",
        remote_path=LEARNING_ENV_DIRS["proposer_solver_v1"], copy=True,
    )
    .add_local_file(
        "configs/rl/repo_repair_smoke.toml",
        remote_path=REPO_REPAIR_RL_CONFIG,
        copy=True,
    )
    .add_local_file(
        "configs/rl/repo_repair_production.toml",
        remote_path=PRODUCTION_RL_CONFIG,
        copy=True,
    )
    .add_local_file(
        "configs/gepa/learning_harness.toml",
        remote_path=GEPA_CONFIG,
        copy=True,
    )
    .add_local_file(
        "configs/rl/multi_source_3_1_1.toml",
        remote_path=MULTI_SOURCE_CONFIG,
        copy=True,
    )
    .add_local_file(
        "configs/rl/max_rl_repo_repair.toml",
        remote_path=ALGORITHM_CONFIGS["max-rl"],
        copy=True,
    )
    .add_local_file(
        "configs/rl/hierarchical_grpo_proposer_solver.toml",
        remote_path=ALGORITHM_CONFIGS["hierarchical-grpo"],
        copy=True,
    )
    .add_local_file(
        "configs/rl/per_environment_grpo_echo.toml",
        remote_path=ALGORITHM_CONFIGS["per-env-grpo-echo"],
        copy=True,
    )
    .add_local_file(
        "configs/rl/echo_terminal.toml",
        remote_path=ALGORITHM_CONFIGS["echo-terminal"],
        copy=True,
    )
    .add_local_file(
        "artifacts/evals/self_hosted_baseline_summary.json",
        remote_path=BASELINE_SUMMARY,
        copy=True,
    )
    .add_local_file(
        "artifacts/rollouts/repo_repair_mixed_step1/all_traces.jsonl",
        remote_path=TRAIN_TRACE_JSONL,
        copy=True,
    )
    .run_commands(
        f"uv pip install --python {PYTHON} --no-deps {ENV_DIR}",
        f"uv pip install --python {PYTHON} --no-deps {HARNESS_DIR}",
        *(
            f"uv pip install --python {PYTHON} --no-deps {path}"
            for path in LEARNING_ENV_DIRS.values()
        ),
        (
            f"uv pip install --python {PYTHON} "
            f"{PRIME_RL_DIR}/deps/verifiers/environments/reverse_text"
        ),
    )
    # These sources change while iterating on Harbor. Keep them after the stable
    # environment-install layer so a runner edit does not reinstall every package.
    .add_local_dir(
        "task2_runtime",
        remote_path=f"{TASK2_RUNTIME_DIR}/task2_runtime",
        copy=True,
    )
    .add_local_file(
        "scripts/run_harbor_e2b.py", remote_path=HARBOR_RUNNER, copy=True
    )
    .env(
        {
            "HF_HOME": "/cache/huggingface",
            "HF_HUB_CACHE": "/cache/huggingface/hub",
            "VLLM_CACHE_ROOT": "/cache/vllm",
            "HF_HUB_DOWNLOAD_TIMEOUT": "300",
        }
    )
)

hf_cache = modal.Volume.from_name("task2-huggingface-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("task2-vllm-cache", create_if_missing=True)
outputs = modal.Volume.from_name("task2-prime-rl-outputs", create_if_missing=True)
task1_models = modal.Volume.from_name("task2-task1-models", create_if_missing=True)


@app.function(image=image, cpu=1, memory=2048, timeout=300)
def dependency_probe() -> dict[str, str]:
    script = (
        "import importlib.metadata as m, inspect; "
        "from e2b.envd.client_sync import SyncHTTPTransport; "
        "print(m.version('e2b')); print(m.version('pyqwest')); "
        "print(m.version('connectrpc')); print(inspect.signature(SyncHTTPTransport))"
    )
    result = subprocess.run(
        [PYTHON, "-c", script], text=True, capture_output=True, timeout=60
    )
    return {
        "exit_code": str(result.returncode),
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


@app.function(
    image=image,
    gpu="A10",
    cpu=8,
    memory=32768,
    timeout=3600,
    volumes={
        "/cache/huggingface": hf_cache,
        "/cache/vllm": vllm_cache,
        "/outputs": outputs,
    },
)
def evaluate_harbor_e2b(e2b_api_key: str) -> dict[str, object]:
    """Run upstream Harbor hello-world using Prime inference + an E2B microVM."""
    run_dir = Path("/outputs/harbor-e2b")
    run_dir.mkdir(parents=True, exist_ok=True)
    inference_log = run_dir / "inference.log"
    runner_log = run_dir / "runner.log"
    trace_path = run_dir / "result.json"
    key_path = Path("/tmp/task2-harbor-e2b-key")
    key_path.write_text(e2b_api_key, encoding="utf-8")
    key_path.chmod(0o600)

    model = "Qwen/Qwen3-1.7B"
    inference_command = [
        "uv", "run", "--frozen", "inference",
        "--vllm.model", model,
        "--vllm.max-model-len", "4096",
        "--vllm.gpu-memory-utilization", "0.85",
    ]
    env = {
        **os.environ,
        "E2B_KEY_FILE": str(key_path),
        "PYTHONPATH": TASK2_RUNTIME_DIR,
        "PYTHONUNBUFFERED": "1",
    }
    with inference_log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            inference_command,
            cwd=PRIME_RL_DIR,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    try:
        _wait_for_inference(process)
        runner_command = [
            "uv", "run", "--frozen", "python", HARBOR_RUNNER,
            "--model", model,
            "--base-url", "http://127.0.0.1:8000/v1",
            "--output", str(trace_path),
        ]
        completed = subprocess.run(
            runner_command,
            cwd=PRIME_RL_DIR,
            env=env,
            text=True,
            capture_output=True,
            timeout=1200,
        )
        runner_log.write_text(
            completed.stdout + "\n[stderr]\n" + completed.stderr,
            encoding="utf-8",
        )
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=30)
        key_path.unlink(missing_ok=True)
        hf_cache.commit()
        vllm_cache.commit()

    result = json.loads(trace_path.read_text(encoding="utf-8")) if trace_path.exists() else {}
    summary = {
        "prime_rl_revision": PRIME_RL_REV,
        "verifiers_revision": VERIFIERS_REV,
        "inference_command": " ".join(inference_command),
        "runner_command": " ".join(runner_command),
        "runner_exit_code": completed.returncode,
        "passed": result.get("passed", False),
        "reward": result.get("reward"),
        "trace_id": result.get("trace", {}).get("id"),
        "model_calls": len(result.get("trace", {}).get("calls", [])),
        "sandbox_type": result.get("trace", {}).get("info", {}).get("runtime"),
        "strict_loader_error": result.get("strict_loader_error"),
        "runner_stdout_tail": completed.stdout[-8000:],
        "runner_stderr_tail": completed.stderr[-8000:],
        "inference_log_tail": inference_log.read_text(
            encoding="utf-8", errors="replace"
        )[-12000:],
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    outputs.commit()
    return summary


@app.function(image=image, cpu=2, memory=4096, timeout=300)
def inspect_hierarchical_validator() -> dict[str, object]:
    needle = "requires a proposer-solver env"
    matches: list[dict[str, str]] = []
    for path in Path(PRIME_RL_DIR).rglob("*.py"):
        source = path.read_text(encoding="utf-8", errors="replace")
        if needle in source:
            index = source.index(needle)
            matches.append({
                "path": str(path),
                "context": source[max(0, index - 1000):index + 1000],
            })
    probe = subprocess.run(
        [PYTHON, "-c", "import proposer_solver_v1; print(proposer_solver_v1.__file__); print(dir(proposer_solver_v1))"],
        text=True, capture_output=True, timeout=60,
    )
    return {
        "matches": matches,
        "import_exit_code": probe.returncode,
        "import_stdout": probe.stdout,
        "import_stderr": probe.stderr,
    }


@app.function(image=image, cpu=2, memory=4096, timeout=600, volumes={"/outputs": outputs})
def inspect_prime_source_mixing() -> dict[str, object]:
    """Sample the pinned Prime TrainSource itself for the configured 3:1:1 mix."""
    script = r'''
import json
from collections import Counter
from types import SimpleNamespace
import verifiers.v1 as vf
from prime_rl.orchestrator.train_source import TrainSource

ratios = {"repo-repair": 3, "terminal": 1, "browser": 1}
envs = []
for name, ratio in ratios.items():
    tasks = [vf.Task(vf.TaskData(idx=i, prompt=f"{name}-{i}")) for i in range(10)]
    envs.append(SimpleNamespace(
        name=name, tasks=iter(tasks), num_tasks=len(tasks),
        config=SimpleNamespace(ratio=ratio),
    ))
source = TrainSource(envs)
state = source.state_dict()
draws = [source.next_example()["env_name"] for _ in range(10000)]
counts = Counter(draws)
observed = {name: counts[name] / len(draws) for name in ratios}
expected = {name: ratio / sum(ratios.values()) for name, ratio in ratios.items()}
restored = TrainSource([
    SimpleNamespace(
        name=name,
        tasks=iter([vf.Task(vf.TaskData(idx=i, prompt=f"{name}-{i}")) for i in range(10)]),
        num_tasks=10,
        config=SimpleNamespace(ratio=ratio),
    ) for name, ratio in ratios.items()
])
restored.load_state_dict(state)
reproducible = [restored.next_example()["env_name"] for _ in range(20)] == draws[:20]
print(json.dumps({
    "class": f"{TrainSource.__module__}.{TrainSource.__name__}",
    "draws": len(draws), "counts": counts, "observed": observed,
    "expected": expected,
    "within_two_percent": all(abs(observed[k] - expected[k]) < 0.02 for k in ratios),
    "checkpoint_rng_reproducible": reproducible,
}))
'''
    completed = subprocess.run(
        [PYTHON, "-c", script], cwd=PRIME_RL_DIR,
        text=True, capture_output=True, timeout=500,
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr)
    result = json.loads(completed.stdout)
    path = Path("/outputs/source-mixing-3-1-1.json")
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    outputs.commit()
    return result


@app.function(image=image, cpu=4, memory=8192, timeout=900, volumes={"/outputs": outputs})
def dry_run_multi_source() -> dict[str, object]:
    run_dir = Path("/outputs/multi-source-3-1-1-dry-run")
    command = [
        "uv", "run", "--frozen", "rl", "@", MULTI_SOURCE_CONFIG,
        "--output-dir", str(run_dir), "--no-wandb", "--dry-run",
    ]
    completed = subprocess.run(
        command, cwd=PRIME_RL_DIR, text=True, capture_output=True, timeout=800,
    )
    result = {
        "command": " ".join(command),
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "modal-summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    outputs.commit()
    if completed.returncode:
        raise RuntimeError(json.dumps(result, indent=2))
    return result


@app.function(image=image, cpu=4, memory=8192, timeout=900, volumes={"/outputs": outputs})
def dry_run_production_repo_repair() -> dict[str, object]:
    """Resolve the real 100-train/25-validation production config with Prime."""
    run_dir = Path("/outputs/repo-repair-production-dry-run")
    command = [
        "uv", "run", "--frozen", "rl", "@", PRODUCTION_RL_CONFIG,
        "--output-dir", str(run_dir), "--no-wandb", "--dry-run",
    ]
    completed = subprocess.run(
        command, cwd=PRIME_RL_DIR, text=True, capture_output=True, timeout=800,
    )
    result = {
        "command": " ".join(command),
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "modal-summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    outputs.commit()
    if completed.returncode:
        raise RuntimeError(json.dumps(result, indent=2))
    return result


@app.function(image=image, cpu=4, memory=8192, timeout=1200, volumes={"/outputs": outputs})
def dry_run_algorithm_configs() -> dict[str, object]:
    """Resolve typed Prime configs for the supported algorithm paths we exercise."""
    results: dict[str, object] = {}
    for name, config in ALGORITHM_CONFIGS.items():
        run_dir = Path("/outputs/algorithm-dry-runs") / name
        command = [
            "uv", "run", "--frozen", "rl", "@", config,
            "--output-dir", str(run_dir), "--no-wandb", "--dry-run",
        ]
        completed = subprocess.run(
            command, cwd=PRIME_RL_DIR, text=True, capture_output=True, timeout=900,
        )
        results[name] = {
            "command": " ".join(command),
            "exit_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    root = Path("/outputs/algorithm-dry-runs")
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    outputs.commit()
    if any(item["exit_code"] for item in results.values()):
        raise RuntimeError(json.dumps(results, indent=2))
    return results


@app.function(image=image, cpu=2, memory=8192, timeout=600, volumes={"/outputs": outputs})
def inspect_actual_training_sample() -> dict[str, object]:
    """Run pinned Prime's real Trace -> TrainingSample conversion on the saved 4-call trace."""
    script = r'''
import json
import asyncio
from pathlib import Path
from prime_rl.configs.algorithm import GRPOAlgoConfig
from prime_rl.orchestrator.algo.grpo import GRPOAlgorithm
from prime_rl.orchestrator.envs import ROLLOUT_TYPE
from prime_rl.orchestrator.trajectories import trace_to_samples

records = [json.loads(line) for line in Path("/opt/artifacts/repo_repair_step1_traces.jsonl").read_text().splitlines() if line.strip()]
trace_data = next(record for record in records if record.get("ok") and len(record.get("calls", [])) >= 4)
group_id = trace_data["info"]["group_id"]
group = [ROLLOUT_TYPE.model_validate(record) for record in records if record.get("ok") and record["info"]["group_id"] == group_id]
for rollout in group:
    rollout.samples = trace_to_samples(rollout, env_name=rollout.info.get("env_name", "tiny-repo-repair-v1"))
asyncio.run(GRPOAlgorithm(GRPOAlgoConfig(), None).finalize_group(group))
trace = next(rollout for rollout in group if len(rollout.calls) >= 4)
samples = trace.samples
payload = {
    "trace_id": trace.id,
    "model_calls": len(trace.calls),
    "branches": len(trace.branches),
    "samples": [
        {
            "token_count": len(sample.token_ids),
            "trainable_tokens": sum(sample.mask),
            "context_tokens": len(sample.mask) - sum(sample.mask),
            "aligned": len(sample.token_ids) == len(sample.mask) == len(sample.logprobs),
            "sampled_spans": [],
            "token_ids": sample.token_ids,
            "mask": sample.mask,
            "inference_logprobs": sample.logprobs,
            "advantages": sample.advantages,
            "rl_weights": sample.rl_weights,
            "ce_weights": sample.ce_weights,
            "ref_kl_weights": sample.ref_kl_weights,
        }
        for sample in samples
    ],
}
# Replace the placeholder above with contiguous true-mask spans without changing Prime output.
for sample in payload["samples"]:
    spans = []
    start = None
    for idx, enabled in enumerate(sample["mask"] + [False]):
        if enabled and start is None:
            start = idx
        elif not enabled and start is not None:
            spans.append([start, idx])
            start = None
    sample["sampled_spans"] = spans
path = Path("/outputs/self-hosted-eval/training-sample-inspection.json")
path.write_text(json.dumps(payload, indent=2))
print(json.dumps(payload))
'''
    result = subprocess.run(
        [PYTHON, "-c", script], cwd=PRIME_RL_DIR, text=True, capture_output=True, timeout=500
    )
    outputs.commit()
    if result.returncode:
        raise RuntimeError(result.stderr)
    payload = json.loads(
        Path("/outputs/self-hosted-eval/training-sample-inspection.json").read_text()
    )
    # Keep the complete, auditable arrays in the Modal volume while returning a
    # compact console result that is practical to inspect after a remote run.
    return {
        "trace_id": payload["trace_id"],
        "model_calls": payload["model_calls"],
        "branches": payload["branches"],
        "samples": [
            {
                "token_count": sample["token_count"],
                "trainable_tokens": sample["trainable_tokens"],
                "context_tokens": sample["context_tokens"],
                "aligned": sample["aligned"],
                "sampled_spans": sample["sampled_spans"],
                "advantage_min": min(sample["advantages"]),
                "advantage_max": max(sample["advantages"]),
                "nonzero_advantages": sum(
                    value != 0 for value in sample["advantages"]
                ),
            }
            for sample in payload["samples"]
        ],
        "full_artifact": (
            "/outputs/self-hosted-eval/training-sample-inspection.json"
        ),
    }


@app.function(image=image, cpu=2, memory=8192, timeout=600, volumes={"/outputs": outputs})
def inspect_max_rl_training_sample() -> dict[str, object]:
    """Materialize MaxRL's real token streams for the effective optimizer cohort."""
    script = r'''
import asyncio
import json
from pathlib import Path

from prime_rl.configs.algorithm import MaxRLAlgoConfig
from prime_rl.orchestrator.algo.max_rl import MaxRLAlgorithm
from prime_rl.orchestrator.envs import ROLLOUT_TYPE
from prime_rl.orchestrator.trajectories import trace_to_samples

path = Path("/outputs/algorithm-training/max-rl/rollouts/step_1/train/effective/traces.jsonl")
records = []
for line in path.read_text().splitlines():
    value = json.loads(line)
    records.extend(value.get("traces", [value]))
group = [ROLLOUT_TYPE.model_validate(record) for record in records]
for rollout in group:
    rollout.samples = trace_to_samples(
        rollout, env_name=rollout.info.get("env_name", "repo-repair")
    )
asyncio.run(MaxRLAlgorithm(MaxRLAlgoConfig(), None).finalize_group(group))

payload = {"algorithm": "max_rl", "group_size": len(group), "rollouts": []}
for rollout in group:
    samples = []
    for sample in rollout.samples:
        trainable = [
            advantage
            for advantage, enabled in zip(sample.advantages, sample.mask, strict=True)
            if enabled
        ]
        samples.append({
            "token_count": len(sample.token_ids),
            "trainable_tokens": sum(sample.mask),
            "advantage_min": min(trainable),
            "advantage_max": max(trainable),
            "advantage_values": sorted(set(trainable)),
            "rl_weight_values": (
                None if sample.rl_weights is None else sorted(set(sample.rl_weights))
            ),
        })
    payload["rollouts"].append({
        "trace_id": rollout.id,
        "reward": rollout.reward,
        "samples": samples,
    })
out = Path("/outputs/algorithm-training/max-rl/sample-inspection.json")
out.write_text(json.dumps(payload, indent=2))
print(json.dumps(payload))
'''
    completed = subprocess.run(
        [PYTHON, "-c", script],
        cwd=PRIME_RL_DIR,
        text=True,
        capture_output=True,
        timeout=500,
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr)
    result = json.loads(completed.stdout)
    outputs.commit()
    return result


@app.function(image=image, cpu=2, memory=8192, timeout=600, volumes={"/outputs": outputs})
def inspect_hierarchical_training_samples() -> dict[str, object]:
    """Replay complete saved groups through Prime's hierarchical credit assignment."""
    script = r'''
import asyncio
import json
from collections import defaultdict
from pathlib import Path

from prime_rl.configs.algorithm import HierarchicalGRPOAlgoConfig
from prime_rl.orchestrator.algo.hierarchical_grpo import HierarchicalGRPOAlgorithm
from prime_rl.orchestrator.envs import ROLLOUT_TYPE
from prime_rl.orchestrator.trajectories import trace_to_samples

path = Path("/outputs/algorithm-training/hierarchical-grpo/rollouts/step_1/train/all/traces.jsonl")
records = []
for line in path.read_text().splitlines():
    value = json.loads(line)
    records.extend(value.get("traces", [value]))
groups = defaultdict(list)
for record in records:
    rollout = ROLLOUT_TYPE.model_validate(record)
    # These orchestration fields are excluded from Trace serialization and
    # mirrored into info by Prime. Rehydrate them before replaying group credit.
    rollout.episode_id = record["info"]["episode_id"]
    rollout.env_name = record["info"]["env_name"]
    rollout.policy_version = record["info"]["policy_version"]
    groups[record["info"]["group_id"]].append(rollout)

payload = {"algorithm": "hierarchical_grpo", "groups": []}
for group_id, group in groups.items():
    for rollout in group:
        rollout.samples = trace_to_samples(
            rollout, env_name=rollout.info.get("env_name", "proposer-solver")
        )
    algo = HierarchicalGRPOAlgorithm(
        HierarchicalGRPOAlgoConfig(episode_agents=["solver"]), None
    )
    asyncio.run(algo.finalize_group(group))
    item = {"group_id": group_id, "rollouts": []}
    for rollout in group:
        samples = []
        for sample in rollout.samples:
            trainable = [
                advantage
                for advantage, enabled in zip(sample.advantages, sample.mask, strict=True)
                if enabled
            ]
            samples.append({
                "trainable_tokens": sum(sample.mask),
                "advantage_values": sorted(set(trainable)),
            })
        item["rollouts"].append({
            "trace_id": rollout.id,
            "episode_id": rollout.info["episode_id"],
            "agent": rollout.agent.name,
            "reward": rollout.reward,
            "samples": samples,
        })
    payload["groups"].append(item)

out = Path("/outputs/algorithm-training/hierarchical-grpo/sample-inspection.json")
out.write_text(json.dumps(payload, indent=2))
print(json.dumps(payload))
'''
    completed = subprocess.run(
        [PYTHON, "-c", script],
        cwd=PRIME_RL_DIR,
        text=True,
        capture_output=True,
        timeout=500,
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr)
    result = json.loads(completed.stdout)
    outputs.commit()
    return result


@app.function(image=image, cpu=2, memory=8192, timeout=600, volumes={"/outputs": outputs})
def inspect_echo_training_samples() -> dict[str, object]:
    """Replay the saved ECHO cohort through pinned Prime's real signal routing."""
    script = r'''
import asyncio
import json
from pathlib import Path

from prime_rl.configs.algorithm import EchoAlgoConfig
from prime_rl.orchestrator.algo.echo import EchoAlgorithm
from prime_rl.orchestrator.envs import ROLLOUT_TYPE
from prime_rl.orchestrator.trajectories import trace_to_samples

path = Path("/outputs/algorithm-training/echo-terminal/rollouts/step_1/train/effective/traces.jsonl")
records = []
for line in path.read_text().splitlines():
    value = json.loads(line)
    records.extend(value.get("traces", [value]))
group = [ROLLOUT_TYPE.model_validate(record) for record in records]
algo = EchoAlgorithm(EchoAlgoConfig(), None)

async def finalize():
    for rollout in group:
        rollout.samples = trace_to_samples(
            rollout, env_name=rollout.info.get("env_name", "terminal-echo")
        )
        await algo.finalize_rollout(rollout)
    await algo.finalize_group(group)

asyncio.run(finalize())
payload = {"algorithm": "echo", "group_size": len(group), "rollouts": []}
for rollout in group:
    samples = []
    for sample in rollout.samples:
        ce_weights = sample.ce_weights or [0.0] * len(sample.token_ids)
        samples.append({
            "token_count": len(sample.token_ids),
            "rl_trainable_tokens": sum(sample.mask),
            "ce_trainable_tokens": sum(weight > 0 for weight in ce_weights),
            "ce_weight_values": sorted(set(ce_weights)),
            "advantage_values": sorted(set(sample.advantages)),
            "rl_weight_values": (
                None if sample.rl_weights is None else sorted(set(sample.rl_weights))
            ),
        })
    payload["rollouts"].append({
        "trace_id": rollout.id,
        "reward": rollout.reward,
        "turns": rollout.num_turns,
        "samples": samples,
    })
payload["ce_trainable_tokens"] = sum(
    sample["ce_trainable_tokens"]
    for rollout in payload["rollouts"]
    for sample in rollout["samples"]
)
out = Path("/outputs/algorithm-training/echo-terminal/sample-inspection.json")
out.write_text(json.dumps(payload, indent=2))
print(json.dumps(payload))
'''
    completed = subprocess.run(
        [PYTHON, "-c", script],
        cwd=PRIME_RL_DIR,
        text=True,
        capture_output=True,
        timeout=500,
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr)
    result = json.loads(completed.stdout)
    outputs.commit()
    return result


@app.function(image=image, cpu=2, memory=8192, timeout=600, volumes={"/outputs": outputs})
def inspect_per_environment_training_samples() -> dict[str, object]:
    """Audit Prime's env-specific ECHO/GRPO routing in both optimizer cohorts."""
    script = r'''
import asyncio
import json
from pathlib import Path

from prime_rl.configs.algorithm import EchoAlgoConfig, GRPOAlgoConfig
from prime_rl.orchestrator.algo.echo import EchoAlgorithm
from prime_rl.orchestrator.algo.grpo import GRPOAlgorithm
from prime_rl.orchestrator.envs import ROLLOUT_TYPE
from prime_rl.orchestrator.trajectories import trace_to_samples

root = Path("/outputs/algorithm-training/per-env-grpo-echo/rollouts")
payload = {"steps": []}
for step in (1, 2):
    path = root / f"step_{step}/train/effective/traces.jsonl"
    records = []
    for line in path.read_text().splitlines():
        value = json.loads(line)
        records.extend(value.get("traces", [value]))
    group = [ROLLOUT_TYPE.model_validate(record) for record in records]
    env_names = sorted({record["info"]["env_name"] for record in records})
    if env_names == ["terminal-echo"]:
        algorithm = "echo"
        algo = EchoAlgorithm(EchoAlgoConfig(), None)
    elif env_names == ["browser-grpo"]:
        algorithm = "grpo"
        algo = GRPOAlgorithm(GRPOAlgoConfig(), None)
    else:
        raise AssertionError(f"unexpected environment cohort: {env_names}")

    async def finalize():
        for rollout in group:
            rollout.samples = trace_to_samples(
                rollout, env_name=rollout.info["env_name"]
            )
            await algo.finalize_rollout(rollout)
        await algo.finalize_group(group)
    asyncio.run(finalize())

    item = {
        "step": step,
        "environment": env_names[0],
        "algorithm": algorithm,
        "rollouts": len(group),
        "rewards": [rollout.reward for rollout in group],
        "rl_trainable_tokens": 0,
        "rl_nonzero_advantage_tokens": 0,
        "ce_trainable_tokens": 0,
        "ce_weight_values": set(),
    }
    for rollout in group:
        for sample in rollout.samples:
            item["rl_trainable_tokens"] += sum(sample.mask)
            item["rl_nonzero_advantage_tokens"] += sum(
                enabled and advantage != 0
                for enabled, advantage in zip(sample.mask, sample.advantages, strict=True)
            )
            if sample.ce_weights is not None:
                item["ce_trainable_tokens"] += sum(
                    weight > 0 for weight in sample.ce_weights
                )
                item["ce_weight_values"].update(sample.ce_weights)
    item["ce_weight_values"] = sorted(item["ce_weight_values"])
    payload["steps"].append(item)

out = Path("/outputs/algorithm-training/per-env-grpo-echo/sample-inspection.json")
out.write_text(json.dumps(payload, indent=2))
print(json.dumps(payload))
'''
    completed = subprocess.run(
        [PYTHON, "-c", script], cwd=PRIME_RL_DIR, text=True,
        capture_output=True, timeout=500,
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr)
    result = json.loads(completed.stdout)
    outputs.commit()
    return result


def _local_e2b_key() -> str:
    key = os.environ.get("E2B_API_KEY")
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not key:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            name, separator, value = line.partition("=")
            if separator and name.strip() == "e2b_api":
                key = value.strip().strip("'\"")
                break
    if not key:
        raise RuntimeError(".env must define e2b_api or E2B_API_KEY must be set")
    return key


def _wait_for_inference(process: subprocess.Popen[str], timeout: int = 900) -> None:
    deadline = time.monotonic() + timeout
    url = "http://127.0.0.1:8000/health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Prime-RL inference exited with code {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2)
    raise TimeoutError(f"Prime-RL inference did not become healthy at {url}")


@app.function(
    image=image,
    gpu="A10",
    cpu=8,
    memory=32768,
    timeout=3600,
    volumes={
        "/cache/huggingface": hf_cache,
        "/cache/vllm": vllm_cache,
        "/outputs": outputs,
    },
)
def evaluate_on_self_hosted_prime_inference(e2b_api_key: str) -> dict[str, object]:
    run_dir = Path("/outputs/self-hosted-eval")
    run_dir.mkdir(parents=True, exist_ok=True)
    inference_log = run_dir / "inference.log"
    eval_log = run_dir / "eval.log"
    key_path = Path("/tmp/task2-e2b-key")
    key_path.write_text(e2b_api_key, encoding="utf-8")
    key_path.chmod(0o600)

    inference_command = [
        "uv",
        "run",
        "--frozen",
        "inference",
        "--vllm.model",
        MODEL,
        "--vllm.max-model-len",
        "4096",
        "--vllm.gpu-memory-utilization",
        "0.85",
    ]
    command_env = {
        **os.environ,
        "E2B_KEY_FILE": str(key_path),
        "TASK2_LOCAL_INFERENCE_KEY": "local-no-auth",
        "PYTHONUNBUFFERED": "1",
    }
    with inference_log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            inference_command,
            cwd=PRIME_RL_DIR,
            env=command_env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    try:
        _wait_for_inference(process)
        eval_command = [
            "uv",
            "run",
            "--frozen",
            "eval",
            "tiny-repo-repair-v1",
            "-m",
            MODEL,
            "-n",
            "1",
            "-r",
            "1",
            "--client.base-url",
            "http://127.0.0.1:8000/v1",
            "--client.api-key-var",
            "TASK2_LOCAL_INFERENCE_KEY",
            "--env.agent.max-turns",
            "8",
            "--env.agent.harness.id",
            "null",
            "--env.agent.runtime.type",
            "subprocess",
            "--no-push",
        ]
        evaluated = subprocess.run(
            eval_command,
            cwd=PRIME_RL_DIR,
            env=command_env,
            text=True,
            capture_output=True,
            timeout=1200,
        )
        eval_log.write_text(
            evaluated.stdout + "\n[stderr]\n" + evaluated.stderr,
            encoding="utf-8",
        )
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        hf_cache.commit()
        vllm_cache.commit()
        outputs.commit()

    artifacts: dict[str, str] = {}
    trace_ok = False
    for path in Path(PRIME_RL_DIR, "outputs").rglob("*"):
        if path.is_file() and path.stat().st_size <= 2_000_000:
            content = path.read_text(
                encoding="utf-8", errors="replace"
            )
            artifacts[str(path.relative_to(PRIME_RL_DIR))] = content
            if path.suffix in {".json", ".jsonl"}:
                for line in content.splitlines():
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    candidates = value if isinstance(value, list) else [value]
                    trace_ok = trace_ok or any(
                        isinstance(candidate, dict) and candidate.get("ok") is True
                        for candidate in candidates
                    )
    summary = {
        "prime_rl_revision": PRIME_RL_REV,
        "verifiers_revision": VERIFIERS_REV,
        "model": MODEL,
        "inference_command": " ".join(inference_command),
        "eval_command": " ".join(eval_command),
        "eval_exit_code": evaluated.returncode,
        "trace_ok": trace_ok,
        "inference_log_tail": inference_log.read_text(
            encoding="utf-8", errors="replace"
        )[-12000:],
        "eval_stdout": evaluated.stdout,
        "eval_stderr": evaluated.stderr,
        "artifacts": artifacts,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    outputs.commit()
    return summary


@app.function(
    image=image,
    gpu="A10",
    cpu=8,
    memory=32768,
    timeout=3600,
    volumes={
        "/cache/huggingface": hf_cache,
        "/cache/vllm": vllm_cache,
        "/outputs": outputs,
    },
)
def run_gepa_self_hosted(dry_run: bool = False) -> dict[str, object]:
    """Run pinned Verifiers v1 GEPA against our local Prime vLLM process."""
    run_name = "gepa-learning-harness-dry-run" if dry_run else "gepa-learning-harness"
    run_dir = Path("/outputs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "TASK2_LOCAL_INFERENCE_KEY": "local-no-auth",
        "PYTHONUNBUFFERED": "1",
    }
    inference_log = run_dir / "inference.log"
    inference_command = [
        "uv", "run", "--frozen", "inference",
        "--vllm.model", MODEL,
        "--vllm.max-model-len", "4096",
        "--vllm.gpu-memory-utilization", "0.85",
    ]
    with inference_log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            inference_command, cwd=PRIME_RL_DIR, env=env,
            stdout=handle, stderr=subprocess.STDOUT, text=True,
        )
    command = [
        "uv", "run", "--frozen", "gepa", "@", GEPA_CONFIG,
        "--output-dir", str(run_dir),
    ]
    if dry_run:
        command.append("--dry-run")
    try:
        _wait_for_inference(process)
        completed = subprocess.run(
            command, cwd=PRIME_RL_DIR, env=env,
            text=True, capture_output=True, timeout=2400,
        )
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=30)
    result = {
        "command": " ".join(command),
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "inference_command": " ".join(inference_command),
        "inference_log_tail": inference_log.read_text(
            encoding="utf-8", errors="replace"
        )[-8000:],
    }
    (run_dir / "modal-summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    outputs.commit()
    if completed.returncode:
        raise RuntimeError(json.dumps(result, indent=2))
    return result


@app.function(image=image, cpu=4, memory=16384, timeout=900, volumes={"/outputs": outputs})
def inspect_checkpoint_fingerprints() -> dict[str, object]:
    """Fingerprint the same stable tensors before and after the resumed optimizer step."""
    script = r'''
import hashlib, json
from pathlib import Path
from safetensors import safe_open
root = Path("/outputs/official-reverse-text-20-step/weights")
selected = ("model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight", "model.norm.weight")
result = {"transport": "Prime in-node NCCL broadcast", "steps": {}}
for step in (20, 21):
    stats = {}
    with safe_open(root / f"step_{step}" / "model.safetensors", framework="pt", device="cpu") as tensors:
        keys = set(tensors.keys())
        for name in selected:
            if name not in keys:
                continue
            value = tensors.get_tensor(name).float()
            stats[name] = {"shape": list(value.shape), "mean": float(value.mean()), "std": float(value.std()), "sha256": hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()}
    result["steps"][str(step)] = stats
before, after = result["steps"]["20"], result["steps"]["21"]
result["changed"] = {name: before[name]["sha256"] != after[name]["sha256"] for name in before.keys() & after.keys()}
result["all_selected_changed"] = bool(result["changed"]) and all(result["changed"].values())
print(json.dumps(result))
'''
    completed = subprocess.run(
        [PYTHON, "-c", script], cwd=PRIME_RL_DIR,
        text=True, capture_output=True, timeout=800,
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr)
    result = json.loads(completed.stdout)
    path = Path("/outputs/official-reverse-text-20-step/weight-fingerprints.json")
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    outputs.commit()
    return result


@app.function(
    image=image,
    gpu="A10:2",
    cpu=16,
    memory=65536,
    timeout=14400,
    volumes={
        "/cache/huggingface": hf_cache,
        "/cache/vllm": vllm_cache,
        "/outputs": outputs,
    },
)
def train_official_reverse_text() -> dict[str, object]:
    run_dir = Path("/outputs/official-reverse-text-20-step")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "modal-command.log"
    command = [
        "uv",
        "run",
        "--frozen",
        "rl",
        "@",
        f"{PRIME_RL_DIR}/examples/basic/reverse-text/rl.toml",
        "--output-dir",
        str(run_dir),
        "--no-wandb",
    ]
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=PRIME_RL_DIR,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    with log_path.open("w", encoding="utf-8") as handle:
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
            handle.flush()
            if "Saved checkpoint" in line or "step_" in line:
                outputs.commit()
    exit_code = process.wait()
    outputs.commit()
    hf_cache.commit()
    vllm_cache.commit()
    files = [
        str(path.relative_to(run_dir))
        for path in run_dir.rglob("*")
        if path.is_file()
    ]
    return {
        "command": " ".join(command),
        "exit_code": exit_code,
        "gpu": "A10:2",
        "wall_seconds": time.monotonic() - started,
        "files": files,
        "log_tail": log_path.read_text(encoding="utf-8", errors="replace")[-20000:],
    }


@app.function(
    image=image,
    gpu="A10:2",
    cpu=16,
    memory=65536,
    timeout=7200,
    volumes={
        "/cache/huggingface": hf_cache,
        "/cache/vllm": vllm_cache,
        "/outputs": outputs,
    },
)
def resume_official_reverse_text() -> dict[str, object]:
    """Resume the real official run from its step-20 trainer/orchestrator checkpoint."""
    run_dir = Path("/outputs/official-reverse-text-20-step")
    log_path = run_dir / "modal-resume-step21.log"
    command = [
        "uv", "run", "--frozen", "rl", "@",
        f"{PRIME_RL_DIR}/examples/basic/reverse-text/rl.toml",
        "--output-dir", str(run_dir), "--no-wandb",
        "--max-steps", "21", "--ckpt.resume-step", "20",
    ]
    started = time.monotonic()
    process = subprocess.run(
        command,
        cwd=PRIME_RL_DIR,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        text=True,
        capture_output=True,
        timeout=7000,
    )
    log_path.write_text(
        process.stdout + "\n[stderr]\n" + process.stderr,
        encoding="utf-8",
    )
    outputs.commit()
    return {
        "command": " ".join(command),
        "exit_code": process.returncode,
        "wall_seconds": time.monotonic() - started,
        "step_21_checkpoint": (run_dir / "checkpoints/step_21/STABLE").exists(),
        "step_21_weights": (run_dir / "weights/step_21/STABLE").exists(),
        "log_tail": log_path.read_text(encoding="utf-8", errors="replace")[-20000:],
    }


@app.function(
    image=image,
    gpu="A10",
    cpu=8,
    memory=32768,
    timeout=1800,
    volumes={
        "/cache/huggingface": hf_cache,
        "/cache/vllm": vllm_cache,
        "/outputs": outputs,
    },
)
def reload_official_checkpoint() -> dict[str, object]:
    """Boot Prime inference from the exported local HF weights and make a real request."""
    checkpoint = Path("/outputs/official-reverse-text-20-step/weights/step_20")
    stable = checkpoint / "STABLE"
    if not stable.exists():
        raise FileNotFoundError(f"stable checkpoint marker missing: {stable}")
    run_dir = Path("/outputs/official-reverse-text-20-step/reload-verification")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "inference.log"
    command = [
        "uv", "run", "--frozen", "inference",
        "--vllm.model", str(checkpoint),
        "--vllm.max-model-len", "2048",
        "--vllm.gpu-memory-utilization", "0.85",
    ]
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=PRIME_RL_DIR,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    try:
        _wait_for_inference(process)
        request_body = json.dumps(
            {
                "model": str(checkpoint),
                "messages": [{"role": "user", "content": "Reverse: modal"}],
                "max_tokens": 32,
                "temperature": 0,
            }
        ).encode()
        request = urllib.request.Request(
            "http://127.0.0.1:8000/v1/chat/completions",
            data=request_body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.loads(response.read())
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=30)
    result = {
        "command": " ".join(command),
        "checkpoint": str(checkpoint),
        "stable": stable.exists(),
        "response": payload,
        "inference_log_tail": log_path.read_text(
            encoding="utf-8", errors="replace"
        )[-12000:],
    }
    (run_dir / "summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    outputs.commit()
    return result


@app.function(
    image=image,
    gpu="A10",
    cpu=8,
    memory=32768,
    timeout=3600,
    volumes={
        "/cache/huggingface": hf_cache,
        "/cache/vllm": vllm_cache,
        "/outputs": outputs,
    },
)
def evaluate_repo_checkpoint(e2b_api_key: str) -> dict[str, object]:
    checkpoint = Path("/outputs/repo-repair-mixed/weights/step_2")
    if not (checkpoint / "STABLE").exists():
        raise FileNotFoundError(f"stable checkpoint missing: {checkpoint}")
    run_dir = Path("/outputs/repo-repair-mixed/checkpoint-eval")
    run_dir.mkdir(parents=True, exist_ok=True)
    key_path = Path("/tmp/task2-e2b-key")
    key_path.write_text(e2b_api_key, encoding="utf-8")
    key_path.chmod(0o600)
    env = {
        **os.environ,
        "E2B_KEY_FILE": str(key_path),
        "TASK2_LOCAL_INFERENCE_KEY": "local-no-auth",
        "PYTHONUNBUFFERED": "1",
    }
    inference_log = run_dir / "inference.log"
    inference_command = [
        "uv", "run", "--frozen", "inference",
        "--vllm.model", str(checkpoint),
        "--vllm.max-model-len", "4096",
        "--vllm.gpu-memory-utilization", "0.85",
        # A checkpoint path cannot be auto-mapped back to the Qwen family, so
        # Prime's parser auto-resolution needs this explicit serving override.
        "--vllm.tool-call-parser", "hermes",
        "--vllm.enable-auto-tool-choice", "true",
    ]
    with inference_log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            inference_command, cwd=PRIME_RL_DIR, env=env,
            stdout=handle, stderr=subprocess.STDOUT, text=True,
        )
    try:
        _wait_for_inference(process)
        eval_command = [
            "uv", "run", "--frozen", "eval", "tiny-repo-repair-v1",
            "-m", str(checkpoint), "-n", "4", "-r", "1",
            "--client.base-url", "http://127.0.0.1:8000/v1",
            "--client.api-key-var", "TASK2_LOCAL_INFERENCE_KEY",
            "--env.taskset.start", "16", "--env.taskset.num-tasks", "4",
            "--env.taskset.task.e2b.allow-internet-access", "false",
            "--env.taskset.task.e2b.timeout-seconds", "240",
            "--env.agent.max-turns", "4",
            "--env.agent.harness.id", "null",
            "--env.agent.runtime.type", "subprocess",
            "--sampling.max-tokens", "512",
            "--output-dir", str(run_dir), "--no-push",
        ]
        evaluated = subprocess.run(
            eval_command, cwd=PRIME_RL_DIR, env=env,
            text=True, capture_output=True, timeout=2400,
        )
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=30)
    summary = {
        "checkpoint": str(checkpoint),
        "stable": True,
        "inference_command": " ".join(inference_command),
        "eval_command": " ".join(eval_command),
        "eval_exit_code": evaluated.returncode,
        "eval_stdout": evaluated.stdout,
        "eval_stderr": evaluated.stderr,
        "inference_log_tail": inference_log.read_text(
            encoding="utf-8", errors="replace"
        )[-12000:],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    outputs.commit()
    return summary


@app.function(
    image=image,
    gpu="A10",
    cpu=8,
    memory=32768,
    timeout=3600,
    volumes={
        "/cache/huggingface": hf_cache,
        "/cache/vllm": vllm_cache,
        "/outputs": outputs,
    },
)
def evaluate_learning_environment_suite(tasksets: list[str] | None = None) -> dict[str, object]:
    """Evaluate every non-repo learning environment on self-hosted Prime inference."""
    run_dir = Path("/outputs/environment-suite" if tasksets is None else "/outputs/environment-rerun")
    run_dir.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "TASK2_LOCAL_INFERENCE_KEY": "local-no-auth",
        "PYTHONUNBUFFERED": "1",
    }
    inference_log = run_dir / "inference.log"
    suite_model = "Qwen/Qwen3-1.7B"
    inference_command = [
        "uv", "run", "--frozen", "inference",
        "--vllm.model", suite_model,
        "--vllm.max-model-len", "4096",
        "--vllm.gpu-memory-utilization", "0.85",
    ]
    with inference_log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            inference_command, cwd=PRIME_RL_DIR, env=env,
            stdout=handle, stderr=subprocess.STDOUT, text=True,
        )
    selected_tasksets = tuple(tasksets) if tasksets else (
        "tiny-terminal-v1",
        "tiny-browser-v1",
        "tiny-tooluse-v1",
        "tiny-long-horizon-v1",
        "proposer-solver-v1",
    )
    results: dict[str, object] = {}
    try:
        _wait_for_inference(process)
        for taskset in selected_tasksets:
            output_dir = run_dir / taskset
            command = [
                "uv", "run", "--frozen", "eval", taskset,
                "-m", suite_model, "-n", "1", "-r", "1",
                "--client.base-url", "http://127.0.0.1:8000/v1",
                "--client.api-key-var", "TASK2_LOCAL_INFERENCE_KEY",
                "--sampling.max-tokens", "1536",
                "--output-dir", str(output_dir), "--no-push",
            ]
            if taskset != "proposer-solver-v1":
                command[command.index("--sampling.max-tokens"):command.index("--sampling.max-tokens")] = [
                    "--env.agent.max-turns", "8",
                    "--env.agent.harness.id", "null",
                    "--env.agent.runtime.type", "subprocess",
                ]
            else:
                role_flags: list[str] = []
                for role in ("proposer", "solver"):
                    role_flags.extend([
                        f"--env.{role}.max-turns", "4",
                        f"--env.{role}.harness.id", "null",
                        f"--env.{role}.runtime.type", "subprocess",
                    ])
                role_flags.extend(["--env.n", "2"])
                command[command.index("--sampling.max-tokens"):command.index("--sampling.max-tokens")] = role_flags
            completed = subprocess.run(
                command, cwd=PRIME_RL_DIR, env=env,
                text=True, capture_output=True, timeout=600,
            )
            trace_path = output_dir / "traces.jsonl"
            traces = []
            if trace_path.exists():
                for line in trace_path.read_text(encoding="utf-8").splitlines():
                    record = json.loads(line)
                    traces.extend(record.get("traces", [record]))
            results[taskset] = {
                "command": " ".join(command),
                "exit_code": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "trace_count": len(traces),
                "ok": all(trace.get("ok", False) for trace in traces) and bool(traces),
                "reward": [trace.get("rewards", {}) for trace in traces],
                "calls": [len(trace.get("calls", [])) for trace in traces],
                "tools": [[tool.get("name") for tool in trace.get("tools", [])] for trace in traces],
                "called_tools": [[
                    call.get("name")
                    for node in trace.get("nodes", [])
                    for call in (node.get("message", {}).get("tool_calls") or [])
                ] for trace in traces],
            }
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=30)
    result = {
        "inference_command": " ".join(inference_command),
        "results": results,
        "all_commands_succeeded": all(
            item["exit_code"] == 0 for item in results.values()
        ),
        "all_traces_ok": all(item["ok"] for item in results.values()),
    }
    (run_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    outputs.commit()
    return result


@app.function(
    image=image,
    gpu="A10",
    cpu=8,
    memory=32768,
    timeout=3600,
    volumes={
        "/cache/huggingface": hf_cache,
        "/cache/vllm": vllm_cache,
        "/outputs": outputs,
    },
)
def evaluate_harness_suite() -> dict[str, object]:
    """Run one trusted task through null, bash, and the custom interception harness."""
    root = Path("/outputs/harness-suite")
    root.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "TASK2_LOCAL_INFERENCE_KEY": "local-no-auth",
        "PYTHONUNBUFFERED": "1",
    }
    inference_log = root / "inference.log"
    inference_command = [
        "uv", "run", "--frozen", "inference",
        "--vllm.model", MODEL,
        "--vllm.max-model-len", "2048",
        "--vllm.gpu-memory-utilization", "0.85",
    ]
    with inference_log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            inference_command, cwd=PRIME_RL_DIR, env=env,
            stdout=handle, stderr=subprocess.STDOUT, text=True,
        )
    results: dict[str, object] = {}
    try:
        _wait_for_inference(process)
        for harness in ("null", "bash", "learning-harness"):
            output_dir = root / harness
            command = [
                "uv", "run", "--frozen", "eval", "learning-harness",
                "-m", MODEL, "-n", "1", "-r", "1",
                "--client.base-url", "http://127.0.0.1:8000/v1",
                "--client.api-key-var", "TASK2_LOCAL_INFERENCE_KEY",
                "--env.agent.max-turns", "2",
                "--env.agent.harness.id", harness,
                "--env.agent.runtime.type", "subprocess",
                "--sampling.max-tokens", "64",
                "--output-dir", str(output_dir), "--no-push",
            ]
            completed = subprocess.run(
                command, cwd=PRIME_RL_DIR, env=env,
                text=True, capture_output=True, timeout=900,
            )
            traces = []
            trace_path = output_dir / "traces.jsonl"
            if trace_path.exists():
                for line in trace_path.read_text(encoding="utf-8").splitlines():
                    record = json.loads(line)
                    traces.extend(record.get("traces", [record]))
            results[harness] = {
                "command": " ".join(command),
                "exit_code": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "trace_count": len(traces),
                "ok": all(trace.get("ok", False) for trace in traces) and bool(traces),
                "calls": [len(trace.get("calls", [])) for trace in traces],
                "runtime": [trace.get("agent", {}).get("runtime", {}) for trace in traces],
                "rewards": [trace.get("rewards", {}) for trace in traces],
            }
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=30)
    result = {"inference_command": " ".join(inference_command), "results": results}
    (root / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    outputs.commit()
    return result


@app.function(
    image=image,
    gpu="A10",
    cpu=8,
    memory=32768,
    timeout=5400,
    volumes={
        "/cache/huggingface": hf_cache,
        "/cache/vllm": vllm_cache,
        "/outputs": outputs,
        "/models/task1": task1_models,
    },
)
def evaluate_sft_and_reference(e2b_api_key: str) -> dict[str, object]:
    """Evaluate the actual Task-1 SFT checkpoint and a larger local reference."""
    key_path = Path("/tmp/task2-e2b-key")
    key_path.write_text(e2b_api_key, encoding="utf-8")
    key_path.chmod(0o600)
    env = {
        **os.environ,
        "E2B_KEY_FILE": str(key_path),
        "TASK2_LOCAL_INFERENCE_KEY": "local-no-auth",
        "PYTHONUNBUFFERED": "1",
    }
    models = (
        ("task1-sft", "/models/task1/sft_overfit", "hermes"),
        ("qwen3-1.7b-reference", "Qwen/Qwen3-1.7B", None),
    )
    root = Path("/outputs/model-comparison")
    root.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {}
    for label, model, parser in models:
        run_dir = root / label
        run_dir.mkdir(parents=True, exist_ok=True)
        inference_log = run_dir / "inference.log"
        inference_command = [
            "uv", "run", "--frozen", "inference",
            "--vllm.model", model,
            "--vllm.max-model-len", "4096",
            "--vllm.gpu-memory-utilization", "0.85",
        ]
        if parser:
            inference_command.extend([
                "--vllm.tool-call-parser", parser,
                "--vllm.enable-auto-tool-choice", "true",
            ])
        with inference_log.open("w", encoding="utf-8") as handle:
            process = subprocess.Popen(
                inference_command, cwd=PRIME_RL_DIR, env=env,
                stdout=handle, stderr=subprocess.STDOUT, text=True,
            )
        completed: subprocess.CompletedProcess[str] | None = None
        startup_error = ""
        try:
            _wait_for_inference(process, timeout=900)
            command = [
                "uv", "run", "--frozen", "eval", "tiny-repo-repair-v1",
                "-m", model, "-n", "4", "-r", "1",
                "--client.base-url", "http://127.0.0.1:8000/v1",
                "--client.api-key-var", "TASK2_LOCAL_INFERENCE_KEY",
                "--env.taskset.start", "16",
                "--env.taskset.num-tasks", "4",
                "--env.taskset.task.e2b.allow-internet-access", "false",
                "--env.taskset.task.e2b.timeout-seconds", "240",
                "--env.agent.max-turns", "6",
                "--env.agent.harness.id", "null",
                "--env.agent.runtime.type", "subprocess",
                "--sampling.max-tokens", "512",
                "--output-dir", str(run_dir), "--no-push",
            ]
            completed = subprocess.run(
                command, cwd=PRIME_RL_DIR, env=env,
                text=True, capture_output=True, timeout=1800,
            )
        except Exception as exc:
            startup_error = f"{type(exc).__name__}: {exc}"
            command = []
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=30)
        trace_path = run_dir / "traces.jsonl"
        traces = []
        if trace_path.exists():
            for line in trace_path.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                traces.extend(record.get("traces", [record]))
        results[label] = {
            "model": model,
            "inference_command": " ".join(inference_command),
            "eval_command": " ".join(command),
            "exit_code": completed.returncode if completed else None,
            "stdout": completed.stdout if completed else "",
            "stderr": completed.stderr if completed else "",
            "startup_error": startup_error,
            "trace_count": len(traces),
            "error_count": sum(not trace.get("ok", False) for trace in traces),
            "rewards": [trace.get("rewards", {}) for trace in traces],
            "calls": [len(trace.get("calls", [])) for trace in traces],
        }
        outputs.commit()
    result = {"results": results}
    (root / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    outputs.commit()
    hf_cache.commit()
    vllm_cache.commit()
    return result


@app.function(
    image=image,
    gpu="A10:2",
    cpu=16,
    memory=65536,
    timeout=14400,
    volumes={
        "/cache/huggingface": hf_cache,
        "/cache/vllm": vllm_cache,
        "/outputs": outputs,
    },
)
def train_repo_repair(e2b_api_key: str, dry_run: bool = False) -> dict[str, object]:
    run_name = "repo-repair-mixed-dry-run" if dry_run else "repo-repair-mixed"
    run_dir = Path("/outputs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    key_path = Path("/tmp/task2-e2b-key")
    key_path.write_text(e2b_api_key, encoding="utf-8")
    key_path.chmod(0o600)
    log_path = run_dir / "modal-command.log"
    command = [
        "uv", "run", "--frozen", "rl", "@", REPO_REPAIR_RL_CONFIG,
        "--output-dir", str(run_dir), "--no-wandb",
    ]
    if dry_run:
        command.append("--dry-run")
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=PRIME_RL_DIR,
        env={
            **os.environ,
            "E2B_KEY_FILE": str(key_path),
            "TASK2_LOCAL_INFERENCE_KEY": "local-no-auth",
            "PYTHONUNBUFFERED": "1",
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    with log_path.open("w", encoding="utf-8") as handle:
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
            handle.flush()
            if "Saved checkpoint" in line or "step_" in line:
                outputs.commit()
    exit_code = process.wait()
    outputs.commit()
    hf_cache.commit()
    vllm_cache.commit()
    return {
        "command": " ".join(command),
        "exit_code": exit_code,
        "gpu": "A10:2",
        "wall_seconds": time.monotonic() - started,
        "files": [
            str(path.relative_to(run_dir))
            for path in run_dir.rglob("*")
            if path.is_file()
        ],
        "log_tail": log_path.read_text(encoding="utf-8", errors="replace")[-20000:],
    }


@app.function(
    image=image,
    gpu="A10:2",
    cpu=16,
    memory=65536,
    timeout=14400,
    volumes={
        "/cache/huggingface": hf_cache,
        "/cache/vllm": vllm_cache,
        "/outputs": outputs,
    },
)
def train_algorithm_smoke(
    algorithm: str, e2b_api_key: str
) -> dict[str, object]:
    """Run real optimizer steps for a validated algorithm configuration."""
    if algorithm not in ALGORITHM_CONFIGS:
        raise ValueError(
            f"unknown algorithm {algorithm!r}; expected one of {sorted(ALGORITHM_CONFIGS)}"
        )
    run_dir = Path("/outputs/algorithm-training") / algorithm
    run_dir.mkdir(parents=True, exist_ok=True)
    key_path = Path("/tmp/task2-e2b-key")
    key_path.write_text(e2b_api_key, encoding="utf-8")
    key_path.chmod(0o600)
    log_path = run_dir / "modal-command.log"
    max_steps = 2 if algorithm == "per-env-grpo-echo" else 1
    command = [
        "uv", "run", "--frozen", "rl", "@", ALGORITHM_CONFIGS[algorithm],
        "--max-steps", str(max_steps),
        "--output-dir", str(run_dir),
        "--no-wandb",
    ]
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=PRIME_RL_DIR,
        env={
            **os.environ,
            "E2B_KEY_FILE": str(key_path),
            "TASK2_LOCAL_INFERENCE_KEY": "local-no-auth",
            "PYTHONUNBUFFERED": "1",
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    try:
        with log_path.open("w", encoding="utf-8") as handle:
            for line in process.stdout:
                print(line, end="", flush=True)
                handle.write(line)
                handle.flush()
                if "Saved checkpoint" in line or "step_" in line:
                    outputs.commit()
        exit_code = process.wait()
    finally:
        key_path.unlink(missing_ok=True)
        outputs.commit()
        hf_cache.commit()
        vllm_cache.commit()
    log = log_path.read_text(encoding="utf-8", errors="replace")
    trainer_log_path = run_dir / "logs" / "trainer.log"
    trainer_log = (
        trainer_log_path.read_text(encoding="utf-8", errors="replace")
        if trainer_log_path.exists()
        else ""
    )
    optimizer_evidence = [
        line for line in trainer_log.splitlines()
        if "Step " in line and "Loss " in line
    ]
    return {
        "algorithm": algorithm,
        "command": " ".join(command),
        "exit_code": exit_code,
        "gpu": "A10:2",
        "wall_seconds": time.monotonic() - started,
        "files": [
            str(path.relative_to(run_dir))
            for path in run_dir.rglob("*")
            if path.is_file()
        ],
        "optimizer_step_observed": bool(optimizer_evidence),
        "optimizer_evidence": optimizer_evidence[-20:],
        "trainer_log_tail": trainer_log[-20000:],
        "log_tail": log[-30000:],
    }


@app.local_entrypoint()
def main(mode: str = "eval") -> None:
    if mode == "probe":
        print(json.dumps(dependency_probe.remote(), indent=2))
        return
    if mode == "eval-harbor-e2b":
        result = evaluate_harbor_e2b.remote(_local_e2b_key())
        print(json.dumps(result, indent=2))
        if result["runner_exit_code"] != 0 or not result["passed"]:
            raise SystemExit(1)
        return
    if mode in {
        "train-max-rl",
        "train-hierarchical-grpo",
        "train-per-env-grpo-echo",
        "train-echo-terminal",
    }:
        algorithm = {
            "train-max-rl": "max-rl",
            "train-hierarchical-grpo": "hierarchical-grpo",
            "train-per-env-grpo-echo": "per-env-grpo-echo",
            "train-echo-terminal": "echo-terminal",
        }[mode]
        result = train_algorithm_smoke.remote(algorithm, _local_e2b_key())
        print(json.dumps(result, indent=2))
        raise SystemExit(
            int(result["exit_code"])
            if result["exit_code"]
            else int(not result["optimizer_step_observed"])
        )
    if mode == "inspect-hierarchical-validator":
        print(json.dumps(inspect_hierarchical_validator.remote(), indent=2))
        return
    if mode == "inspect-sample":
        print(json.dumps(inspect_actual_training_sample.remote(), indent=2))
        return
    if mode == "inspect-max-rl-sample":
        print(json.dumps(inspect_max_rl_training_sample.remote(), indent=2))
        return
    if mode == "inspect-hierarchical-samples":
        print(json.dumps(inspect_hierarchical_training_samples.remote(), indent=2))
        return
    if mode == "inspect-echo-samples":
        print(json.dumps(inspect_echo_training_samples.remote(), indent=2))
        return
    if mode == "inspect-per-env-samples":
        print(json.dumps(inspect_per_environment_training_samples.remote(), indent=2))
        return
    if mode == "inspect-weight-fingerprints":
        print(json.dumps(inspect_checkpoint_fingerprints.remote(), indent=2))
        return
    if mode == "inspect-source-mixing":
        print(json.dumps(inspect_prime_source_mixing.remote(), indent=2))
        return
    if mode == "dry-run-multi-source":
        print(json.dumps(dry_run_multi_source.remote(), indent=2))
        return
    if mode == "dry-run-production":
        print(json.dumps(dry_run_production_repo_repair.remote(), indent=2))
        return
    if mode == "dry-run-algorithms":
        print(json.dumps(dry_run_algorithm_configs.remote(), indent=2))
        return
    if mode == "train-official":
        result = train_official_reverse_text.remote()
        print(json.dumps(result, indent=2))
        raise SystemExit(int(result["exit_code"]))
    if mode == "resume-official":
        result = resume_official_reverse_text.remote()
        print(json.dumps(result, indent=2))
        raise SystemExit(int(result["exit_code"]))
    if mode == "reload-official":
        result = reload_official_checkpoint.remote()
        print(json.dumps(result, indent=2))
        return
    if mode == "eval-repo-checkpoint":
        result = evaluate_repo_checkpoint.remote(_local_e2b_key())
        print(json.dumps(result, indent=2))
        raise SystemExit(int(result["eval_exit_code"]))
    if mode in {"gepa", "gepa-dry-run"}:
        result = run_gepa_self_hosted.remote(mode == "gepa-dry-run")
        print(json.dumps(result, indent=2))
        raise SystemExit(int(result["exit_code"]))
    if mode == "eval-environment-suite":
        result = evaluate_learning_environment_suite.remote()
        print(json.dumps(result, indent=2))
        if not result["all_commands_succeeded"] or not result["all_traces_ok"]:
            raise SystemExit(1)
        return
    if mode == "eval-browser":
        result = evaluate_learning_environment_suite.remote(["tiny-browser-v1"])
        print(json.dumps(result, indent=2))
        item = result["results"]["tiny-browser-v1"]
        if item["exit_code"] or not item["ok"]:
            raise SystemExit(1)
        return
    if mode == "eval-model-comparison":
        result = evaluate_sft_and_reference.remote(_local_e2b_key())
        print(json.dumps(result, indent=2))
        failed = any(
            item["startup_error"] or item["exit_code"] != 0
            for item in result["results"].values()
        )
        if failed:
            raise SystemExit(1)
        return
    if mode == "eval-harness-suite":
        result = evaluate_harness_suite.remote()
        print(json.dumps(result, indent=2))
        failed = any(
            item["exit_code"] != 0 or not item["ok"]
            for item in result["results"].values()
        )
        if failed:
            raise SystemExit(1)
        return
    if mode in {"train-repo-repair", "dry-run-repo-repair"}:
        result = train_repo_repair.remote(
            _local_e2b_key(), mode == "dry-run-repo-repair"
        )
        print(json.dumps(result, indent=2))
        raise SystemExit(int(result["exit_code"]))
    if mode != "eval":
        raise ValueError(
            "mode must be eval, probe, train-official, resume-official, "
            "reload-official, eval-repo-checkpoint, eval-environment-suite, "
            "eval-model-comparison, "
            "eval-harness-suite, "
            "gepa, gepa-dry-run, "
            "train-repo-repair, or dry-run-repo-repair"
        )
    result = evaluate_on_self_hosted_prime_inference.remote(_local_e2b_key())
    print(json.dumps(result, indent=2))
    if int(result["eval_exit_code"]) != 0 or not bool(result["trace_ok"]):
        raise SystemExit(1)
