"""Rollout dispatcher sending prompt requests to Prime-RL Inference Server."""

from typing import Any, Dict

import requests


class RolloutDispatcher:
    """Dispatches prompt requests to Inference Server endpoints."""

    def __init__(self, inference_url: str = "http://127.0.0.1:8000"):
        self.inference_url = inference_url

    def request_rollout(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        payload = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        }
        res = requests.post(f"{self.inference_url}/generate", json=payload, timeout=30)
        res.raise_for_status()
        return res.json()
