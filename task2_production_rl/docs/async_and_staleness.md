# Asynchronous RL and Policy Staleness

This document covers the mechanics of synchronous vs. asynchronous RL, policy lag, importance sampling ratio degradation, and off-policy risks.

---

## 1. Synchronous vs. Asynchronous RL

```text
Synchronous RL:
Rollout Phase (All GPUs) ──> Trainer Phase (All GPUs) ──> Weight Sync ──> Repeat
(Guarantees lag = 0, but GPUs idle during generation/training switches)

Asynchronous RL:
Rollout Fleet (Generates rollouts continuously under Policy N)
             │ (Async Trajectory Queue)
             ▼
Trainer Fleet (Consumes rollouts, updates Policy N -> N+1 -> N+2)
             │ (Async Weight Broadcast)
             ▼
Rollout Fleet (Receives updated weights async)
(High throughput, GPU idle time = 0, but lag > 0)
```

---

## 2. Policy Lag ($\text{lag}$)

$$\text{lag} = N_{\text{current\_policy}} - N_{\text{rollout\_policy}}$$

- When $\text{lag} = 0$: Strict on-policy RL.
- When $1 \le \text{lag} \le 3$: Moderate off-policy RL (typical in production scaling).
- When $\text{lag} > 5$: Severe policy staleness.

---

## 3. Importance Sampling Ratios and Off-Policy Mismatch

In PPO/GRPO, the policy ratio is:

$$r_t(\theta) = \frac{\pi_\theta(y_t \mid s_t, y_{<t})}{\pi_{\text{old}}(y_t \mid s_t, y_{<t})}$$

If rollouts were generated under policy $N - \text{lag}$ rather than policy $N$:
1. $\pi_{\text{old}}$ reflects policy $N - \text{lag}$.
2. On step 0 of optimizer training, $\pi_\theta$ reflects policy $N$.
3. Ratio $r_t(\theta) \neq 1.0$ at $k=0$ even before taking a gradient step!
4. High policy lag causes PPO clipping ($\text{clip}(r_t, 1-\epsilon, 1+\epsilon)$) to clip nearly 100% of tokens, zeroing out effective policy gradients.

---

## 4. Mitigation Strategies

1. **Max Policy Lag Threshold**: Drop rollouts whose $\text{lag} > \text{max\_lag\_threshold}$.
2. **Dynamic Generation Throttling**: Pause prompt dispatch if trainer queue lag exceeds limits.
3. **Truncated Importance Sampling**: Clip off-policy weights to prevent ratio explosion.
