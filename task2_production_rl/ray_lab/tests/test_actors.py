"""Unit tests for Ray actors."""

import numpy as np
import pytest
import ray

from ray_lab.actors import LearnerWorker, PolicyWorker, RewardWorker


@pytest.fixture(scope="module")
def ray_session():
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


def test_policy_worker_state(ray_session):
    policy = PolicyWorker.remote(version=1, initial_weights=np.ones((10,)))
    v1 = ray.get(policy.get_version.remote())
    fp1 = ray.get(policy.get_weight_fingerprint.remote())

    assert v1 == 1
    assert abs(fp1 - 10.0) < 1e-5

    # Update weights
    new_w = np.full((10,), 2.0, dtype=np.float32)
    v2 = ray.get(policy.update_weights.remote(new_w, 2))
    fp2 = ray.get(policy.get_weight_fingerprint.remote())

    assert v2 == 2
    assert abs(fp2 - 20.0) < 1e-5


def test_reward_worker(ray_session):
    reward_w = RewardWorker.remote()
    item = {"completion": "weight sum 10.0"}
    res = ray.get(reward_w.compute_reward.remote(item))
    assert res["reward"] == 1.0


def test_learner_worker_lag_detection(ray_session):
    learner = LearnerWorker.remote(initial_version=5)
    rollout_batch = [
        {"policy_version": 5},  # lag 0
        {"policy_version": 3},  # lag 2 (stale)
    ]
    res = ray.get(learner.update_policy.remote(rollout_batch))

    assert res["new_version"] == 6
    assert res["max_lag"] == 2
    assert res["stale_count"] == 1
