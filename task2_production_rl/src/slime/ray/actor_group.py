"""Ray Actor Group controller managing training ranks and rollout engines for slime."""

from typing import Any, Dict, List

import ray

from slime.ray.train_actor import TrainActor


class RayActorGroup:
    """Central Ray Controller for slime cluster execution."""

    def __init__(self, num_train_ranks: int = 1):
        if not ray.is_initialized():
            ray.init(
                ignore_reinit_error=True,
                runtime_env={
                    "env_vars": {"PYTHONPATH": "."},
                    "excludes": ["artifacts", "artifacts/checkpoints", ".venv"],
                },
            )

        self.train_ranks = [TrainActor.remote(rank=i) for i in range(num_train_ranks)]

    def run_training_step(self, mini_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        futures = [actor.train_step.remote(mini_batch) for actor in self.train_ranks]
        return ray.get(futures)


def main():
    print("Launching slime RayActorGroup controller...")
    group = RayActorGroup(num_train_ranks=1)
    dummy_batch = [{"prompt": "Test", "completion": "Answer"}]
    results = group.run_training_step(dummy_batch)
    print("slime RayActorGroup step completed:", results)


if __name__ == "__main__":
    main()
