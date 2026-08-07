"""Production RL End-to-End Orchestrated Stack Test Runner.

Launches Prime-RL Inference Server, Trainer, and Orchestrator processes,
executes actual RL rollout -> verifier reward -> advantage -> packing -> trainer step -> weight sync loop,
and launches slime RayActorGroup controller training step, capturing process trace logs.
"""

import os
import subprocess
import sys
import time

import requests


def run_production_stack():
    os.makedirs("artifacts/traces", exist_ok=True)

    inf_log = open("artifacts/traces/prime_inference.log", "w", encoding="utf-8")
    trn_log = open("artifacts/traces/prime_trainer.log", "w", encoding="utf-8")
    orc_log = open("artifacts/traces/prime_orchestrator.log", "w", encoding="utf-8")
    slm_log = open("artifacts/traces/slime_trainer.log", "w", encoding="utf-8")

    python_exe = sys.executable

    print("Step 1: Spawning Prime-RL Inference Server on port 8000...")
    inf_proc = subprocess.Popen(
        [python_exe, "-m", "prime_rl.entrypoints.inference"],
        stdout=inf_log,
        stderr=inf_log,
    )

    # Wait for Inference Server readiness
    server_ready = False
    for _ in range(30):
        try:
            r = requests.get("http://127.0.0.1:8000/docs", timeout=1)
            if r.status_code == 200:
                server_ready = True
                break
        except Exception:
            pass
        time.sleep(1)

    assert server_ready, "Prime-RL Inference Server failed to start within timeout"
    print("Inference Server is LIVE and ready!")

    print("\nStep 2: Spawning Prime-RL Trainer on ZMQ port 5555...")
    trn_proc = subprocess.Popen(
        [python_exe, "-m", "prime_rl.entrypoints.trainer"],
        stdout=trn_log,
        stderr=trn_log,
    )
    time.sleep(3)

    print("\nStep 3: Running Prime-RL Orchestrator Process...")
    orc_res = subprocess.run(
        [python_exe, "-m", "prime_rl.entrypoints.orchestrator"],
        stdout=orc_log,
        stderr=orc_log,
        check=True,
    )
    print("Orchestrator process finished successfully!")

    time.sleep(5)

    # Clean up background processes
    trn_proc.terminate()
    inf_proc.terminate()

    inf_log.close()
    trn_log.close()
    orc_log.close()

    print("\nStep 4: Running slime RayActorGroup Controller...")
    slm_res = subprocess.run(
        [python_exe, "-m", "slime.ray.actor_group"],
        stdout=slm_log,
        stderr=slm_log,
        check=True,
    )
    slm_log.close()
    print("slime RayActorGroup step finished successfully!")

    print("\n=== PRODUCTION RL END-TO-END STACK TEST PASSED! ===")


if __name__ == "__main__":
    run_production_stack()
