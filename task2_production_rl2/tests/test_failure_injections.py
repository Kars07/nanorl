from __future__ import annotations

import json
from pathlib import Path

from probes.failure_injection_lab import run


def test_artifact_backed_failure_injections_are_detected() -> None:
    result = run()
    assert result["all_passed"]
    required = {
        "identical_group_rewards",
        "group_size_one",
        "zero_advantage_batch",
        "repeated_gibberish_rollout",
        "extension_property_break",
        "context_compaction_discontinuity",
        "subagent_handoff_discontinuity",
        "stale_inference_policy",
        "omitted_weight_update",
        "trainer_inference_logprob_mismatch",
        "wrong_renderer_template",
        "train_eval_contamination",
        "one_environment_dominating",
        "wrong_checkpoint_resume",
    }
    assert required <= result["checks"].keys()


def test_real_e2b_failure_injections_passed() -> None:
    artifact = json.loads(
        Path("artifacts/failure_injections/e2b.json").read_text(encoding="utf-8")
    )
    assert artifact["all_passed"]
    assert artifact["checks"] == {
        "impossible_hidden_test": True,
        "hidden_verifier_not_in_guest": True,
        "network_denied": True,
        "runtime_timeout": True,
        "missing_dependency": True,
    }
