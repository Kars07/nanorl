# GRPO Learner Walkthrough — Group Relative Policy Optimization

This document teaches the complete mathematical data flow of Group Relative Policy Optimization (GRPO) without external RL abstractions.

---

## 1. Concrete Rollout Group Example

Consider prompt $P$: `"What is 2 + 2?"` (Ground truth answer: `4`).

We sample $G = 4$ completions from the old policy $\pi_{\theta_{\text{old}}}$:

| Completion ID | Generated Completion | Verifier Reward $r_i$ | Group Mean $\mu$ | Group Std $\sigma$ | Group Advantage $A_i$ |
| --- | --- | --- | --- | --- | --- |
| $c_1$ | `"2 + 2 = 4. #### 4"` | **1.0** | 0.50 | 0.50 | $+1.0$ |
| $c_2$ | `"2 + 2 = 5. #### 5"` | **0.0** | 0.50 | 0.50 | $-1.0$ |
| $c_3$ | `"The answer is 4. #### 4"` | **1.0** | 0.50 | 0.50 | $+1.0$ |
| $c_4$ | `"The answer is 6. #### 6"` | **0.0** | 0.50 | 0.50 | $-1.0$ |

$$A_i = \frac{r_i - \mu_{\text{group}}}{\sigma_{\text{group}} + \epsilon} = \frac{1.0 - 0.50}{0.50} = +1.0$$

---

## 2. Policy Ratio & PPO Clipping

For token $t$ in completion $c_i$:

- $\pi_{\theta}(a_t \mid s_t)$: Current policy probability
- $\pi_{\theta_{\text{old}}}(a_t \mid s_t)$: Old rollout policy probability (detached)

$$r_t(\theta) = \exp(\log \pi_{\theta}(a_t \mid s_t) - \log \pi_{\theta_{\text{old}}}(a_t \mid s_t))$$

Clipped Surrogate Objective:

$$\mathcal{L}_{\text{policy}}(\theta) = -\min \left( r_t(\theta) A_i, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_i \right)$$

---

## 3. KL Regularization Penalty

To prevent the policy model from diverging catastrophically from the original SFT base model $\pi_{\text{ref}}$, we add Schulman's $k_3$ unbiased KL estimator:

$$\mathbb{D}_{\text{KL}}(\pi_{\theta} \parallel \pi_{\text{ref}}) = \exp(\log \pi_{\text{ref}} - \log \pi_{\theta}) - (\log \pi_{\text{ref}} - \log \pi_{\theta}) - 1$$

$$\mathcal{L}_{\text{total}}(\theta) = \mathcal{L}_{\text{policy}}(\theta) + \beta \cdot \mathbb{D}_{\text{KL}}(\pi_{\theta} \parallel \pi_{\text{ref}})$$

---

## 4. Why Old and Reference Values Must Be Detached

- `old_logprobs` and `reference_logprobs` represent static historical snapshots collected during rollout generation.
- If `old_logprobs` were attached to the computation graph, backpropagation would erroneously attempt to differentiate through historical generation choices, leading to invalid gradients and memory leaks!
- Therefore: `old_logprobs.requires_grad == False` and `ref_logprobs.requires_grad == False`.

---

## 5. What Happens Across K Optimization Epochs

Over $K$ optimization epochs:
1. `old_logprobs` remain fixed.
2. `current_logprobs` update as parameters $\theta$ change.
3. Ratio $r_t(\theta)$ moves away from $1.0$.
4. PPO clipping activates whenever $r_t(\theta) > 1 + \epsilon$ (for $A_i > 0$) or $r_t(\theta) < 1 - \epsilon$ (for $A_i < 0$), bounding step sizes.
