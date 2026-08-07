"""FastAPI Data Buffer server for slime rollout storage and batch sampling."""

import time
from typing import Any, Dict, List

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="slime Data Buffer Server")

BUFFER: List[Dict[str, Any]] = []


class TrajectoryPayload(BaseModel):
    instance_id: str
    prompt: str
    completion: str
    reward: float
    policy_version: int


@app.post("/add_data")
def add_data(data: TrajectoryPayload):
    item = data.dict()
    item["timestamp"] = time.time()
    BUFFER.append(item)
    return {"status": "SUCCESS", "buffer_size": len(BUFFER)}


@app.get("/get_batch")
def get_batch(batch_size: int = 2):
    if len(BUFFER) < batch_size:
        return {"status": "EMPTY", "batch": []}

    batch = [BUFFER.pop(0) for _ in range(batch_size)]
    return {"status": "SUCCESS", "batch": batch}


def main(host: str = "127.0.0.1", port: int = 8080):
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
