"""Prime-RL Orchestrator Process coordinating rollouts, verifiers, advantages, and training batches."""

from typing import Any, Dict

from prime_rl.orchestrator.algo.grpo import GRPOAlgorithm
from prime_rl.orchestrator.dispatcher import RolloutDispatcher
from prime_rl.orchestrator.envs import compute_math_reward
from prime_rl.transport.zmq import ZMQPushSender


class Orchestrator:
    """Central Orchestrator Process driving RL data collection and trajectory packing."""

    def __init__(
        self,
        inference_url: str = "http://127.0.0.1:8000",
        trainer_port: int = 5555,
        num_generations: int = 2,
    ):
        self.dispatcher = RolloutDispatcher(inference_url)
        self.algo = GRPOAlgorithm()
        self.trainer_sender = ZMQPushSender("127.0.0.1", trainer_port)
        self.num_generations = num_generations

    def process_prompt_group(self, prompt: str, target_answer: str) -> Dict[str, Any]:
        group_rollouts = []
        rewards = []

        for g in range(self.num_generations):
            rollout = self.dispatcher.request_rollout(prompt, max_new_tokens=128)
            reward = compute_math_reward(rollout["completion"], target_answer)

            rollout["reward"] = reward
            group_rollouts.append(rollout)
            rewards.append(reward)

        advantages = self.algo.compute_advantages(rewards)

        for item, adv in zip(group_rollouts, advantages):
            item["advantage"] = adv

        packed_batch = self.algo.pack_trajectories(group_rollouts)
        self.trainer_sender.send_json(packed_batch)

        total_packed_len = sum(s["seq_len"] for s in packed_batch["samples"])

        return {
            "prompt": prompt,
            "rewards": rewards,
            "advantages": advantages,
            "packed_seq_len": total_packed_len,
        }


def main():
    print("Starting Prime-RL Orchestrator Process...")
    orchestrator = Orchestrator()

    prompts = [
        ("Solve: 15 + 27", "42"),
        ("Calculate: 12 * 4", "48"),
    ]

    for p, ans in prompts:
        res = orchestrator.process_prompt_group(p, ans)
        print(f"Orchestrated prompt '{p}': rewards={res['rewards']}, packed_len={res['packed_seq_len']}")


if __name__ == "__main__":
    main()
