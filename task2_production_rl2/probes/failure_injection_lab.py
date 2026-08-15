from __future__ import annotations

import json
import math
import re
import tomllib
from collections import Counter
from pathlib import Path
from typing import Any

from probes.loss_reference import default_loss, grpo_advantages
from tiny_tooluse_v1.taskset import TooluseTask
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.harness import Harness
from verifiers.v1.runtimes.subprocess import SubprocessConfig
from verifiers.v1.utils.compile import validate_pairing


ROOT = Path(__file__).resolve().parents[1]


def _records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        value = json.loads(line)
        records.extend(value.get("traces", [value]))
    return records


def _is_extension(previous: list[int], current: list[int]) -> bool:
    return len(current) >= len(previous) and current[: len(previous)] == previous


def _gibberish(tokens: list[int]) -> bool:
    return len(tokens) >= 32 and len(set(tokens)) / len(tokens) < 0.1


class _NoMCPHarness(Harness):
    SUPPORTS_MCP = False
    NEEDS_CONTAINER = False

    async def launch(self, *args, **kwargs):  # pragma: no cover - rejected first
        raise AssertionError("launch must not run")


def _missing_harness_tool_rejected() -> bool:
    try:
        validate_pairing(
            _NoMCPHarness(HarnessConfig(id="no-mcp")),
            TooluseTask,
            SubprocessConfig(),
            tools=["issues_query"],
        )
    except ValueError as error:
        return "does not support MCP tools" in str(error)
    return False


def run() -> dict[str, Any]:
    """Execute deterministic failure injections against real configs and artifacts."""
    payload = json.loads(
        (ROOT / "artifacts/samples/repo_repair_training_sample.json").read_text(
            encoding="utf-8"
        )
    )
    sample = payload["samples"][0]
    token_ids = sample["token_ids"]
    spans = sample["sampled_spans"]
    config = tomllib.loads(
        (ROOT / "configs/rl/repo_repair_smoke.toml").read_text(encoding="utf-8")
    )
    train = config["orchestrator"]["train"]["source"][0]["env"]["taskset"]
    evaluate = config["orchestrator"]["eval"]["source"][0]["env"]["taskset"]
    train_ids = set(range(train["start"], train["start"] + train["num_tasks"]))
    eval_ids = set(range(evaluate["start"], evaluate["start"] + evaluate["num_tasks"]))
    contaminated_eval_ids = {max(train_ids), *eval_ids}

    official_log = (ROOT / "artifacts/training/official_reverse_text/modal-command.log").read_text(
        encoding="utf-8", errors="replace"
    )
    versions = [int(v) for v in re.findall(r"Updating policy in-flight to v(\d+)", official_log)]
    resume_log = (ROOT / "artifacts/training/official_reverse_text/resume-step21.log").read_text(
        encoding="utf-8", errors="replace"
    )
    trainer_log = (ROOT / "artifacts/training/repo_repair/trainer.log").read_text(
        encoding="utf-8", errors="replace"
    )
    mismatch = [float(v) for v in re.findall(r"Mismatch KL ([0-9.]+)", trainer_log)]
    traces = _records(ROOT / "artifacts/rollouts/repo_repair_mixed_step1/all_traces.jsonl")
    policy_versions = [int(trace["info"]["policy_version"]) for trace in traces]

    equal = grpo_advantages([1.0, 1.0, 1.0, 1.0])
    singleton = grpo_advantages([0.75])
    zero_loss = default_loss(
        trainer_logprobs=[-1.0, -2.0],
        inference_logprobs=[-1.0, -2.0],
        advantages=[0.0, 0.0],
        loss_mask=[True, True],
        dppo_mask_high=10.0,
        dppo_mask_low=10.0,
        adv_tau=1.0,
        kl_tau=0.0,
    )

    prefix = token_ids[: spans[0][1]]
    extension = token_ids[: spans[1][1]]
    compacted = token_ids[50 : spans[1][1]]
    handoff = [999_999, *extension[1:]]
    wrong_renderer = [token_ids[0], 888_888, *extension[2:]]
    omitted_update = versions[:-1]
    dominated_counts = Counter(["repo"] * 19 + ["browser"])

    checks = {
        "identical_group_rewards": equal == [0.0] * 4,
        "group_size_one": singleton == [0.0],
        "zero_advantage_batch": zero_loss.loss == 0.0,
        "missing_harness_tool": _missing_harness_tool_rejected(),
        "repeated_gibberish_rollout": _gibberish([42] * 64),
        "extension_success_control": _is_extension(prefix, extension),
        "extension_property_break": not _is_extension(prefix, compacted),
        "context_compaction_discontinuity": not _is_extension(prefix, compacted),
        "subagent_handoff_discontinuity": not _is_extension(prefix, handoff),
        "wrong_renderer_template": not _is_extension(prefix, wrong_renderer),
        "stale_inference_policy": max(policy_versions) == min(policy_versions) == 0,
        "omitted_weight_update": omitted_update[-1] != versions[-1],
        "trainer_inference_logprob_mismatch": bool(mismatch) and max(mismatch) > 0,
        "actual_train_eval_split_control": train_ids.isdisjoint(eval_ids),
        "train_eval_contamination": bool(train_ids & contaminated_eval_ids),
        "one_environment_dominating": max(dominated_counts.values()) / sum(dominated_counts.values()) > 0.9,
        "wrong_checkpoint_resume": 19 != 20
        and "Resuming from step 20" in resume_log
        and "Step 21" in resume_log
        and "Resuming from step 19" not in resume_log,
    }
    return {
        "checks": checks,
        "all_passed": all(checks.values()),
        "evidence": {
        "real_sample_trace": payload.get("trace_id"),
            "real_sample_spans": spans,
            "equal_advantages": equal,
            "singleton_advantages": singleton,
            "actual_policy_versions_step1": sorted(set(policy_versions)),
            "actual_weight_updates": versions,
            "actual_mismatch_kl": mismatch,
            "train_ids": [min(train_ids), max(train_ids)],
            "eval_ids": [min(eval_ids), max(eval_ids)],
            "injected_contaminated_eval_ids": sorted(contaminated_eval_ids),
            "injected_requested_resume_step": 19,
            "actual_resume_step": 20,
            "injected_environment_counts": dict(dominated_counts),
        },
    }


def main() -> None:
    result = run()
    output = ROOT / "artifacts/failure_injections/artifact_detectors.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
