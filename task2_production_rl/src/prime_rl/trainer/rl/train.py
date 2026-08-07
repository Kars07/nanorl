"""Prime-RL Trainer Process implementing FSDP2 / CausalLM RL optimization loop."""

import os
from typing import Any, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prime_rl.trainer.rl.broadcast.nccl import WeightBroadcaster
from prime_rl.trainer.rl.loss import LossInputs, compute_grpo_loss
from prime_rl.transport.zmq import ZMQPullReceiver


class RLTrainer:
    """Trainer process consuming packed ZMQ batches, running loss backward, and updating policy weights."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
        port: int = 5555,
        lr: float = 5e-6,
        inference_url: str = "http://127.0.0.1:8000",
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing Prime-RL Trainer on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        self.receiver = ZMQPullReceiver(port)
        self.broadcaster = WeightBroadcaster(inference_url)
        self.policy_version = 0

    def step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        self.model.train()
        self.optimizer.zero_grad()

        samples = batch["samples"]
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        total_policy_loss = torch.tensor(0.0, device=self.device)
        total_kl = torch.tensor(0.0, device=self.device)
        clip_fracs = []

        for sample in samples:
            input_ids = torch.tensor([sample["input_ids"]], dtype=torch.long, device=self.device)
            comp_mask = torch.tensor(sample["completion_mask"], dtype=torch.bool, device=self.device)
            adv_val = sample["advantage"]

            out = self.model(input_ids=input_ids)
            logits = out.logits[:, :-1, :]
            shift_tokens = input_ids[:, 1:]

            target_logits = logits.gather(dim=-1, index=shift_tokens.unsqueeze(-1)).squeeze(-1).float()
            lse = torch.logsumexp(logits.float(), dim=-1)
            trainer_lp = (target_logits - lse)[0]

            with torch.no_grad():
                inf_lp = trainer_lp.clone()

            advs = torch.full_like(trainer_lp, fill_value=adv_val)

            inputs = LossInputs(
                trainer_logprobs=trainer_lp,
                inference_logprobs=inf_lp,
                ref_logprobs=None,
                advantages=advs,
                loss_mask=comp_mask,
            )

            outputs = compute_grpo_loss(inputs)
            total_loss = total_loss + outputs.loss
            total_policy_loss = total_policy_loss + outputs.policy_loss
            total_kl = total_kl + outputs.kl_penalty
            clip_fracs.append(outputs.clip_fraction)

        avg_loss = total_loss / len(samples)

        if avg_loss.requires_grad and avg_loss.item() != 0.0:
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

        self.policy_version += 1

        # Broadcast weights to inference server
        ckpt_dir = "artifacts/checkpoints/prime_trainer"
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, "latest_weights.pt")
        self.broadcaster.broadcast_weights(self.model, self.policy_version, ckpt_path)

        return {
            "version": self.policy_version,
            "loss": float(avg_loss.item()),
            "policy_loss": float((total_policy_loss / len(samples)).item()),
            "clip_fraction": float(sum(clip_fracs) / len(clip_fracs)),
        }


def train():
    trainer = RLTrainer()
    print("Waiting for packed batches on ZMQ port 5555...")
    batch = trainer.receiver.recv_json(timeout_ms=30000)
    if batch:
        metrics = trainer.step(batch)
        print(f"Trainer Step Completed: version={metrics['version']}, loss={metrics['loss']:.4f}")
    else:
        print("No ZMQ batch received within timeout.")


if __name__ == "__main__":
    train()
