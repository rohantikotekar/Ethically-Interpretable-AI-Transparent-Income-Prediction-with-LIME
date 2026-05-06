# Writeup - Rohan Vijay Tikotekar

---

## Section 1 — Decisions to Defend

---

### Architecture
**Retained the suggested 3-layer, 256-hidden unit MLP with ReLU activations and a Tanh output.**

**Reason:**
The given state space is low dimensional (19 dimensions tracking object/end-effector poses) so a massive architecture is not necessary. A 256-width network is already highly expressive for this input size. Expanding depth or width would highly increase the risk of overfitting to the small demonstration data (9,666 transitions). The Tanh output is mathematically sound because the action space is strictly bounded by [-1, 1]. This is a hard and differentiable constraint that prevents network saturation outside the permissible action bounds.

---

### Loss Function
**Mean Squared Error (MSE) coupled with a small L2 weight decay 10⁻²**

**Reason:**
MSE assumes that the demonstrator's actions are drawn from a Gaussian distribution with a constant variance. In deterministic continuous control, MSE minimizes the negative log-likelihood so it is the preferred choice. L1 (Huber) loss is more robust to outliers but human teleoperation for simple pick-and-place tasks like `lift-ph` is generally smooth and unimodal. So there is less chance of finding more outliers, hence L1 is not the best choice. Added a lightweight L2 regularization (weight decay) to penalize overly sharp decision boundaries. This helps delay the onset of out-of-distribution catastrophic failure during rollouts.

---

### Training Duration
**15 Epochs**

**Reason:**
Minimizing the training loss to near-zero usually results in the network memorizing the specific high-frequency noise of the demonstrator. It does not focus on learning a generalized control policy. Given the batch size (256) and the dataset scale, 15 epochs allows the optimizer to converge to a smooth mean policy without overfitting.

---

### Stopping Criterion
**Fixed Epoch Limit**

**Reason:**
Within the constrained computational and temporal constraints, repeatedly spinning up MuJoCo environments for mid-training evaluation is slow. Hence a fixed, empirically chosen epoch limit is practical. In a production setting, prioritize early stopping.

---

### Checked Evaluation (Cell 4)

```
Output: 30/30 [01:44<00:00,  5.20s/it, success_rate=0.80]
```

**Failure Mode:**
The primary failure mode is a z-axis misalignment resulting in a premature grasp. The policy approaches the target successfully but halts its descent marginally above the object. Then it closes the gripper on empty space and executes the lift sequence without the payload.

**Diagnosis:**
The BC policy lacks a causal understanding of the grasp mechanics. It has no mechanism to recover from accumulating spatial errors.

---

## Section 2 — Building the Optimizer

When the BC policy fails, it generally executes a successful x, y planar approach but halts its z-axis descent roughly a centimeter above the object. It then closes the gripper (`a = -1`) on empty air and attempts the upward lift sequence without the payload. The BC policy lacks an understanding of grasp mechanics. It has no closed-loop mechanism to recover from small compounding spatial errors (covariate shift). The residual policy is explicitly designed to provide the micro-corrections needed to overcome this terminal z-axis hesitation.

---

### Architecture
**δ(s) only.**

Feeding the base action into the residual network is redundant since it is already a deterministic function of the state `s`. Restricting the residual's input to just the state `s` reduces the parameter count and prevents the RL agent from overfitting.

---

### Activation on δ
**Tanh × Bound**

Used a Tanh output layer scaled by the bound scalar (`raw_delta = tanh(x) * bound`). This mathematically guarantees the network can never output a nudge larger than the permitted safety margin. Using a hard clip (`torch.clamp`) without the scaled Tanh is incorrect in continuous RL — it would kill the actor's gradients for any out-of-bound pre-activation predictions, affecting the learning process.

---

### Bound Magnitude
**0.05**

The BC backbone is already highly competent (80% success) so we do not need macroscopic trajectory generation. A bound of 0.05 is large enough to correct the micro-misalignments but small enough to act as a safety constraint. This prevents the RL agent from completely overriding the safe BC prior and degrading into random exploration.

---

### Algorithm
**TD3 (Twin Delayed DDPG)**

For offline continuous control, standard off-policy algorithms often suffer from catastrophic Q-value overestimation. This architecture strictly bounds the residual to 0.05, which acts as a severe behavior-cloning penalty. The strict bounding regularizes the action space, allowing a standard TD3 setup to remain stable without the need for heavier offline-specific algorithms like AWAC or BCQ.

---

### Reward
**Sparse Terminal**

Retained the default sparse reward. Attempting to shape the reward (e.g., heavily penalizing the distance between the end-effector and the cube) introduces dense, engineered gradients that often conflict with the latent kinematic trajectory already learned by the BC policy. Sparse reward guarantees the residual strictly optimizes for the ultimate goal (a successful lift) without introducing local minima.

---

### Clip δ Inside the Target Q Computation
**YES**

In the TD3 critic update, `Q(s', a')` is evaluated. The `ResidualPolicy.forward()` method explicitly scales the Tanh and hard-clamps the sum to `[-1, 1]`, so `a'` is strictly bounded. Failing to clip the action inside the target Q computation would cause the critic to query Q-values for physically impossible states (e.g., a gripper command of +5 or -5%), leading to massive overestimation bias and divergence.

---

### Critic Target Update
**Soft Polyak (τ = 0.005)**

Hard target network updates (copying weights every N steps) cause severe oscillations in the Q-landscape. This is dangerous when fine-tuning a narrow residual. Polyak averaging (τ = 0.005) smooths the target surface and stabilizes the actor's gradient ascent over the offline dataset.

---

### Training Step Count
**5,000 Steps**

The residual is only learning a localized micro-correction rather than a full trajectory from scratch, so it converges rapidly. Training well past 5,000 steps in a strictly offline regime without environment interaction carries a high risk of over-optimizing the Q-function on the static dataset, leading to brittleness during live rollouts. 5,000 steps provided stable `q_mean` convergence without saturating the delta bounds.

---

### Training Diagnostics

Over 5,000 steps, the residual policy exhibited highly stable convergence:

- **`q_mean`** grew monotonically and stabilized at **+0.729**, confirming that strictly bounding the actions inside the target Q computation successfully prevented the overestimation bias of offline TD3.
- **`delta_mag`** stabilized at approximately **0.049**, indicating that the actor learned to fully utilize its permitted 0.05 correction budget — overcoming the BC backbone's hesitation without fully saturating to a constant scalar.

---

## Section 3

### 1. How did you pick the margin?

The margin was calculated as exactly 2% of the empirical range for each individual action dimension observed in the expert demonstration:

```
margin = (expert_max - expert_min) * 0.02
```

This provides a strict, data-driven buffer — giving the residual policy just enough slack to execute micro-corrections near boundary states without risking kinematic deadlocks.

---

### 2. Why per-dimension instead of a global L2 norm?

Because robotic action dimensions map to independent physical actuators (e.g., arm translation vs. gripper closure). If the network predicts a safe trajectory but violates the limit on the gripper speed, applying a global L2 norm would shrink the entire vector — altering the robot's physical direction in space. Per-dimension clipping strictly halts the violating joint at its limit while perfectly preserving the safe commands of the other joints.

---

### Empirical Results

| Method | Success Rate |
|---|---|
| BC Baseline | 80.0% |
| Residual + Shield | 46.7% |
| **Degradation** | **−33.3%** |

The addition of the residual policy and safety shield induced a **33.3% performance degradation**.

---

### Diagnostic Analysis — Why Did the System Regress?

A successful training phase (stable Q-values, bounded delta, minimizing actor/critic loss) followed by catastrophic live deployment is a symptom of two interacting failures:

---

#### 1. Safety Shield Causing Kinematic Deadlocks

The safety shield established bounds based on the empirical min/max of the expert dataset plus a 2% margin. However, the expert min/max is not the robot's kinematic limit — it is simply the boundary of the human's specific trajectory. When the BC policy encounters a slight out-of-distribution state (covariate shift), it needs to output an action slightly outside the historical dataset to recover. By hard-clipping actions to the dataset's boundaries, the shield acted as a kinematic deadlock. Instead of keeping the robot safe, it artificially trapped the arm whenever it attempted a recovery motion outside the nominal human envelope.

#### 2. Offline RL Overestimation

During training, `q_mean` stabilized at +0.729. However, because standard TD3 was used, the critic evaluated Q-values on a static, offline dataset without live environment interaction. The actor learned to output deltas that maximized this proxy Q-function, exploiting the critic's blind spots. When deployed in the live MuJoCo environment, these optimal offline nudges pushed the end-effector into novel state spaces where the BC backbone's predictions degraded rapidly — inducing early task failure (evidenced by the lower `mean_steps_on_success` of 38.8).

---

### Real-World Compatibility

| Pipeline | P99 Latency |
|---|---|
| Frozen BC Backbone | 0.77 ms |
| Residual + Shield (combined) | 1.49 ms |

---

### Future Work / Corrective Next Steps

Given more time, the following two structural changes would be implemented:

1. **Shield Relaxation** — Transition the safety shield from an empirical dataset boundary to an actual kinematic safety boundary (e.g., maximum safe joint velocities). This would allow the BC policy the necessary slack to execute recovery maneuvers.

2. **TD3+BC Implementation** — Modify the residual's actor loss function to include a behavior-cloning penalty. This would penalize the residual for pushing the state too far from the dataset distribution, anchoring the RL agent closer to the stable BC prior while still allowing micro-corrections.
