import json
import math
import tomllib
from pathlib import Path

from probes.loss_reference import default_loss, grpo_advantages


def test_prime_grpo_has_no_std_normalization() -> None:
    assert grpo_advantages([0.0, 1.0, 1.0, 0.0]) == [-0.5, 0.5, 0.5, -0.5]
    assert grpo_advantages([1.0, 1.0]) == [0.0, 0.0]


def test_default_loss_importance_ratio_mask_and_kl() -> None:
    result = default_loss(
        trainer_logprobs=[math.log(0.4), math.log(0.1), math.log(0.3)],
        inference_logprobs=[math.log(0.2), math.log(0.3), math.log(0.3)],
        advantages=[1.0, -1.0, 2.0],
        loss_mask=[True, True, False],
        dppo_mask_high=0.1,
        dppo_mask_low=0.1,
        adv_tau=1.0,
        kl_tau=0.2,
    )
    assert result.kept == [False, False, False]
    expected = 0.2 * (math.log(2.0) ** 2 + math.log(1 / 3) ** 2)
    assert math.isclose(result.loss, expected, rel_tol=1e-12)


def test_default_loss_weights_apply_after_components() -> None:
    result = default_loss(
        trainer_logprobs=[0.0], inference_logprobs=[0.0], advantages=[2.0],
        loss_mask=[True], dppo_mask_high=1.0, dppo_mask_low=1.0,
        adv_tau=0.5, kl_tau=1.0, loss_weights=[3.0],
    )
    assert result.loss == -3.0


def test_echo_configs_preserve_ce_samples_and_dispatch_per_environment() -> None:
    dedicated = tomllib.loads(Path("configs/rl/echo_terminal.toml").read_text())
    mixed = tomllib.loads(
        Path("configs/rl/per_environment_grpo_echo.toml").read_text()
    )
    assert dedicated["orchestrator"]["post_batch_filters"] == []
    assert mixed["orchestrator"]["post_batch_filters"] == []
    assert mixed["orchestrator"]["algo"]["type"] == "grpo"
    sources = {source["name"]: source for source in mixed["orchestrator"]["train"]["source"]}
    assert "algo" not in sources["browser-grpo"]
    assert sources["terminal-echo"]["algo"]["type"] == "echo"


def test_actual_per_environment_optimizer_artifact_has_both_loss_streams() -> None:
    payload = json.loads(
        Path(
            "artifacts/training/algorithm_runs/per_env_grpo_echo/sample-inspection.json"
        ).read_text()
    )
    echo, grpo = payload["steps"]
    assert (echo["environment"], echo["algorithm"]) == ("terminal-echo", "echo")
    assert echo["ce_trainable_tokens"] > 0
    assert echo["ce_weight_values"] == [0.0, 0.1]
    assert (grpo["environment"], grpo["algorithm"]) == ("browser-grpo", "grpo")
    assert grpo["ce_trainable_tokens"] == 0
    assert grpo["rl_nonzero_advantage_tokens"] == 0
