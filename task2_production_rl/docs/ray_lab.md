# Ray Worker / Actor Learning Lab

This document details the Ray worker and actor architecture implemented in `ray_lab/`.

---

## 1. Actor Architecture (`ray_lab/actors.py`)

- **`PolicyWorker`**: Ray actor hosting policy parameters $\theta$ and version counter $N$. Supports remote weight update calls (`update_weights`) and generation calls (`generate`).
- **`RolloutWorker`**: Ray actor collecting completions from `PolicyWorker` references asynchronously and tagging completions with `policy_version`.
- **`RewardWorker`**: Ray actor evaluating completion rewards.
- **`LearnerWorker`**: Ray actor maintaining master policy version $N_{\text{learner}}$, consuming rollout batches, tracking policy lag ($\text{lag} = N_{\text{learner}} - N_{\text{rollout}}$), and updating model parameters.

---

## 2. Versioned Mini-Pipeline (`ray_lab/pipeline.py`)

Simulates async execution loop:
1. `LearnerWorker` version $N$.
2. `RolloutWorker` collects completions tagged with version $N$.
3. When delays occur, `LearnerWorker` updates parameters to version $N+1$.
4. Rollouts tagged version $N$ arrive at `LearnerWorker`, triggering policy lag detection ($\text{lag} > 0$).

---

## 3. Unit Test Verification

Run Ray lab tests:
```bash
uv run pytest ray_lab/tests/
```
Tests verify actor state persistence between remote calls, weight fingerprint updating, policy lag detection, and worker execution.
