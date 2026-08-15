from __future__ import annotations

import json

from verifiers.v1.tasksets.harbor import HarborConfig, HarborTaskset


def main() -> None:
    # hello-world declares a Dockerfile. Verifiers v1 intentionally refuses to
    # build it, so metadata-only inspection opts into the documented fallback.
    # This does not claim that the task has been evaluated in the fallback image.
    config = HarborConfig(dataset="harbor/hello-world", ignore_dockerfile=True)
    taskset = HarborTaskset(config=config)
    task = next(iter(taskset.head(1)))
    data = task.data
    print(
        json.dumps(
            {
                "taskset_class": type(taskset).__name__,
                "task_class": type(task).__name__,
                "idx": data.idx,
                "name": data.name,
                "prompt": data.prompt,
                "task_dir": data.task_dir,
                "image": data.image,
                "upload_environment": data.upload_environment,
                "network_allow": data.network_allow,
                "network_block": data.network_block,
                "verifier_separate": data.verifier is not None,
                "artifacts": [
                    artifact.model_dump(mode="json") for artifact in data.artifacts
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
