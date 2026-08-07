"""Weight broadcast sender to Inference Server."""

import requests
import torch
import torch.nn as nn


class WeightBroadcaster:
    """Publishes trained weight updates to Prime-RL Inference Server."""

    def __init__(self, inference_url: str = "http://127.0.0.1:8000"):
        self.inference_url = inference_url

    def broadcast_weights(self, model: nn.Module, version: int, save_path: str | None = None) -> bool:
        if save_path:
            torch.save(model.state_dict(), save_path)

        payload = {
            "policy_version": version,
            "weights_path": save_path,
        }

        try:
            res = requests.post(f"{self.inference_url}/update_weights", json=payload, timeout=10)
            return res.status_code == 200
        except Exception as e:
            print(f"Weight broadcast notification failed: {e}")
            return False
