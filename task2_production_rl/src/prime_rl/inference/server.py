"""Prime-RL Inference Server utilizing vLLM engine for high-throughput decoding."""

import os
from typing import Any, Dict
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

try:
    from vllm import LLM, SamplingParams
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

from prime_rl.inference.worker.weight_transfer import load_weights_checkpoint_layerwise

app = FastAPI(title="Prime-RL vLLM Inference Server")

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
    print(f"Initializing Prime-RL Inference Server using model `{model_id}` (vLLM Engine: {HAS_VLLM})...")

    if HAS_VLLM:
        # High-throughput vLLM Engine Path
        engine = LLM(
            model=model_id,
            trust_remote_code=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.5,
        )
        MODEL_HOLDER["vllm_engine"] = engine
        MODEL_HOLDER["engine_type"] = "vllm"
    else:
        # Fallback path for environments without vLLM (e.g. Windows native)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(device)
        model.eval()

        MODEL_HOLDER["tokenizer"] = tokenizer
        MODEL_HOLDER["model"] = model
        MODEL_HOLDER["device"] = device
        MODEL_HOLDER["engine_type"] = "hf_fallback"

    MODEL_HOLDER["version"] = 0


@app.post("/generate")
def generate(req: GenerateRequest):
    if MODEL_HOLDER.get("engine_type") == "vllm":
        vllm_engine: LLM = MODEL_HOLDER["vllm_engine"]
        sampling_params = SamplingParams(
            temperature=req.temperature,
            top_p=req.top_p,
            max_tokens=req.max_new_tokens,
        )
        outputs = vllm_engine.generate([req.prompt], sampling_params)
        output = outputs[0]

        completion_text = output.outputs[0].text
        completion_ids = list(output.outputs[0].token_ids)
        prompt_ids = list(output.prompt_token_ids)

        return {
            "prompt": req.prompt,
            "completion": completion_text,
            "prompt_tokens": prompt_ids,
            "completion_tokens": completion_ids,
            "full_tokens": prompt_ids + completion_ids,
            "policy_version": MODEL_HOLDER["version"],
            "engine": "vLLM",
        }
    else:
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
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
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
            "engine": "HuggingFace_Fallback",
        }


@app.post("/update_weights")
def update_weights(req: WeightUpdateRequest):
    if req.weights_path and MODEL_HOLDER.get("engine_type") == "hf_fallback":
        state_dict = torch.load(req.weights_path, map_location=MODEL_HOLDER["device"])
        load_weights_checkpoint_layerwise(MODEL_HOLDER["model"], state_dict)

    MODEL_HOLDER["version"] = req.policy_version
    return {"status": "SUCCESS", "version": MODEL_HOLDER["version"]}


def main(host: str = "127.0.0.1", port: int = 8000, model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"):
    init_inference_engine(model_id)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
