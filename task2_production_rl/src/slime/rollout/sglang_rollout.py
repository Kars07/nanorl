"""SGLang RadixAttention rollout sampler for slime."""

from typing import Any, Dict, List
import requests

try:
    import sglang as sgl
    HAS_SGLANG = True
except ImportError:
    HAS_SGLANG = False


class SGLangRollout:
    """SGLang sampler engine interface using RadixAttention."""

    def __init__(self, inference_url: str = "http://127.0.0.1:8000"):
        self.inference_url = inference_url
        self.has_sglang = HAS_SGLANG

    def generate_rollout(self, prompt: str, max_new_tokens: int = 128) -> Dict[str, Any]:
        payload = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": 0.7,
        }
        res = requests.post(f"{self.inference_url}/generate", json=payload, timeout=30)
        res.raise_for_status()
        out = res.json()
        out["sglang_radix_attention"] = self.has_sglang
        return out
