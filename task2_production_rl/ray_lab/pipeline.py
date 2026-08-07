"""Ray versioned mini-pipeline simulation."""

import json
import os
from typing import Any, Dict

import ray

from ray_lab.actors import LearnerWorker, PolicyWorker, RewardWorker, RolloutWorker


def run_versioned_pipeline(num_steps: int = 5, introduce_lag: bool = True) -> Dict[str, Any]:
    """Run a versioned Ray mini-pipeline tracking policy lag and stale rollouts."""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    policy_worker = PolicyWorker.remote(version=0)
    rollout_worker = RolloutWorker.remote(worker_id="rollout_1")
    reward_worker = RewardWorker.remote()
    learner_worker = LearnerWorker.remote(initial_version=0)

    trace_log = []

    for step in range(num_steps):
        current_learner_ver = ray.get(learner_worker.get_version.remote())

        # Deliberately introduce artificial delay on specific steps to simulate rollout lag
        delay = 0.2 if (introduce_lag and step == 2) else 0.0

        # Rollout generation
        rollout_future = rollout_worker.collect_rollout.remote(
            policy_worker,
            prompt=f"Prompt at step {step}",
            delay_seconds=delay,
        )

        # In asynchronous execution, learner might update while rollout is in flight
        if introduce_lag and step == 2:
            # Simulate intermediate learner update before rollout completes
            dummy_batch = [{"policy_version": current_learner_ver}]
            _ = ray.get(learner_worker.update_policy.remote(dummy_batch))
            # Sync policy worker weights to latest learner version
            new_weights = ray.get(learner_worker.get_weights.remote())
            new_ver = ray.get(learner_worker.get_version.remote())
            ray.get(policy_worker.update_weights.remote(new_weights, new_ver))

        rollout_item = ray.get(rollout_future)

        # Score reward
        scored_item = ray.get(reward_worker.compute_reward.remote(rollout_item))

        # Learner update
        learner_metrics = ray.get(learner_worker.update_policy.remote([scored_item]))

        # Sync policy worker to new learner weights
        new_weights = ray.get(learner_worker.get_weights.remote())
        new_ver = ray.get(learner_worker.get_version.remote())
        ray.get(policy_worker.update_weights.remote(new_weights, new_ver))

        log_entry = {
            "step": step,
            "rollout_policy_version": scored_item["policy_version"],
            "learner_policy_version": learner_metrics["new_version"],
            "lag": learner_metrics["max_lag"],
            "reward": scored_item["reward"],
        }
        trace_log.append(log_entry)

    summary = {
        "num_steps": num_steps,
        "trace": trace_log,
        "total_stale_rollouts": ray.get(learner_worker.update_policy.remote([]))["stale_count"],
    }

    os.makedirs("artifacts/traces", exist_ok=True)
    out_path = "artifacts/traces/ray_pipeline_trace.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    summary = run_versioned_pipeline(num_steps=5, introduce_lag=True)
    print("Ray pipeline execution completed:")
    print(json.dumps(summary, indent=2))
