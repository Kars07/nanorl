"""Prime-RL Inference Server hosting live policy model."""

from typing import Any, Dict

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from prime_rl.inference.worker.weight_transfer import load_weights_checkpoint_layerwise

app = FastAPI(title="Prime-RL Inference Server")

MODEL_HOLDER: Dict[str, Any] = {}


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 1.0


class WeightUpdateRequest(BaseModel):
    policy_version: int
    weights_path: str | None = None


def init_inference_engine(model_id: str = "Qwen/Qwen2.5-0.5B-Instruct", device: str | None = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing Prime-RL Inference Engine `{model_id}` on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(device)
    model.eval()

    MODEL_HOLDER["tokenizer"] = tokenizer
    MODEL_HOLDER["model"] = model
    MODEL_HOLDER["device"] = device
    MODEL_HOLDER["version"] = 0


@app.post("/generate")
def generate(req: GenerateRequest):
    tokenizer = MODEL_HOLDER["tokenizer"]
    model = MODEL_HOLDER["model"]
    device = MODEL_HOLDER["device"]

    messages = [{"role": "user", "content": req.prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            do_sample=True,
            temperature=req.temperature,
            top_p=req.top_p,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    full_ids = outputs[0].tolist()
    prompt_len = inputs["input_ids"].shape[1]
    completion_ids = full_ids[prompt_len:]
    completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)

    return {
        "prompt": req.prompt,
        "completion": completion_text,
        "prompt_tokens": inputs["input_ids"][0].tolist(),
        "completion_tokens": completion_ids,
        "full_tokens": full_ids,
        "policy_version": MODEL_HOLDER["version"],
    }


@app.post("/update_weights")
def update_weights(req: WeightUpdateRequest):
    if req.weights_path and torch.cuda.is_available():
        state_dict = torch.load(req.weights_path, map_location=MODEL_HOLDER["device"])
        load_weights_checkpoint_layerwise(MODEL_HOLDER["model"], state_dict)

    MODEL_HOLDER["version"] = req.policy_version
    return {"status": "SUCCESS", "version": MODEL_HOLDER["version"]}


def main(host: str = "127.0.0.1", port: int = 8000, model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"):
    init_inference_engine(model_id)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
