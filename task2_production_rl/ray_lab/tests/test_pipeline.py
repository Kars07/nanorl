"""Unit tests for Ray pipeline execution."""

import pytest
import ray

from ray_lab.pipeline import run_versioned_pipeline


@pytest.fixture(scope="module")
def ray_session():
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


def test_versioned_pipeline_lag_detection(ray_session):
    summary = run_versioned_pipeline(num_steps=4, introduce_lag=True)

    assert summary["num_steps"] == 4
    assert len(summary["trace"]) == 4

    lags = [entry["lag"] for entry in summary["trace"]]
    assert any(lag > 0 for lag in lags), "Pipeline should detect non-zero policy lag when delay is introduced"
