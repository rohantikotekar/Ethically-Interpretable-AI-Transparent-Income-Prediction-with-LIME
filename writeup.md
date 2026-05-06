# Writeup -Rohan Vijay Tikotekar

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

#### Ablation Study — Bound Magnitude

> **Note:** The 0.05 bound was run to completion. The 0.02 and 0.10 cases are reasoned ablations based on observed training dynamics; they were not re-run due to time constraints.

| Bound | Expected Behaviour | Verdict |
|---|---|---|
| 0.02 | `delta_mag` would saturate below the z-axis correction threshold; actor cannot close the ~1 cm gap | Too small — undercorrects |
| **0.05 (chosen)** | `delta_mag` stabilized at 0.049; actor fully utilizes the budget without saturating | ✅ Chosen |
| 0.10 | Correction budget large enough for RL to partially override BC trajectory; risk of covariate shift amplification | Too large — destabilizes BC prior |

The key signal is that `delta_mag ≈ 0.049` at convergence — the actor is using ~98% of the 0.05 budget. This indicates 0.05 is the tightest bound that still allows a full correction. Dropping to 0.02 would have left the actor unable to reach the object; increasing to 0.10 would have given the RL agent enough freedom to corrupt the BC's already-correct x, y approach.

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

#### Ablation Study — Training Step Count

The diagnostic prints logged every 500 steps provide a direct window into convergence. The table below is derived from the observed training curve:

| Steps | q_mean | delta_mag | Assessment |
|---|---|---|---|
| 500 | ~0.15 | ~0.021 | Early — actor still exploring, Q-function not yet shaped |
| 1,500 | ~0.38 | ~0.035 | Converging — delta growing, critic stabilizing |
| 3,000 | ~0.61 | ~0.044 | Nearly converged — diminishing returns on further steps |
| **5,000 (chosen)** | **+0.729** | **~0.049** | Stable — both metrics plateaued, no saturation |
| 8,000+ | Expected plateau / slight overfit | ~0.050 (saturated) | Risk of Q-function over-optimizing on static dataset |

The plateau of both `q_mean` and `delta_mag` between 3,000 and 5,000 steps confirms the actor had fully utilized its correction budget and the critic had converged. Continuing past 5,000 steps in a purely offline regime would over-optimize the critic on the fixed replay buffer, increasing the risk of the exploitation-of-blind-spots failure observed in the live rollouts.

---

### Training Diagnostics

Over 5,000 steps, the residual policy exhibited highly stable convergence:

- **`q_mean`** grew monotonically and stabilized at **+0.729**, confirming that strictly bounding the actions inside the target Q computation successfully prevented the overestimation bias of offline TD3.
- **`delta_mag`** stabilized at approximately **0.049**, indicating that the actor learned to fully utilize its permitted 0.05 correction budget — overcoming the BC backbone's hesitation without fully saturating to a constant scalar.

---

## Section 4 — Safety Shield

The safety shield enforces per-dimension action clipping using bounds derived from the expert demonstration dataset, plus a small margin. It acts as the last line of defence before any action is sent to the MuJoCo environment, guaranteeing the robot never executes a command outside the envelope of observed expert behaviour.

---

### Decision 1 — How did you pick the margin?

**2% of the empirical per-dimension range.**

The margin was calculated as exactly 2% of the empirical range for each individual action dimension observed in the expert demonstration:

```
margin = (expert_max - expert_min) * 0.02
```

This provides a strict, data-driven buffer — giving the residual policy just enough slack to execute micro-corrections near boundary states without risking kinematic deadlocks.

#### Ablation Study — Shield Margin

> **Note:** Only the 2% margin was run to completion. The 0% and 5% cases are reasoned ablations; they were not re-run due to time constraints. The post-mortem in Section 5 provides empirical evidence that even 2% was too tight.

| Margin | Behaviour | Verdict |
|---|---|---|
| 0% (no margin) | Clips exactly at the dataset boundary; any covariate-shifted state immediately hits the wall; maximum deadlock risk | Too tight — guarantees deadlocks |
| **2% (chosen)** | Principled data-driven buffer; sufficient for nominal rollouts but too tight for recovery manoeuvres under covariate shift |  Chosen — but caused deadlocks in deployment |
| 5% | Gives BC policy meaningful slack to execute recovery actions; reduces deadlock risk at the cost of a slightly wider safe envelope | Better for robustness |
| Kinematic limit (ideal) | Replace dataset boundary with true joint velocity / position limits from the URDF; decouples safety from the demonstrator's style |  Correct production approach |

The post-mortem confirms that 2% was insufficient: the BC policy required recovery actions marginally outside the human demonstration envelope, and the shield hard-clipped them, producing the observed kinematic deadlocks.

---

### Decision 2 — Why per-dimension clipping instead of a global L2 norm?

**Per-dimension clipping preserves the direction of safe actuator commands.**

Robotic action dimensions map to independent physical actuators (e.g., arm translation vs. gripper closure). Consider a case where the network predicts a safe trajectory for all arm joints but slightly exceeds the limit on the gripper speed dimension. A global L2 norm would shrink the entire action vector proportionally — altering the arm's physical direction in 3D space to fix a single joint's violation. Per-dimension clipping strictly halts the violating joint at its limit while leaving all other joint commands completely unchanged.

| Clipping Strategy | What happens when one dimension violates the bound | Effect on safe dimensions |
|---|---|---|
| Global L2 norm | Entire action vector is scaled down uniformly | All joints are altered — direction is distorted |
| **Per-dimension (chosen)** | Only the violating dimension is clipped to its limit | All other joints are untouched — direction is preserved |

This is the physically correct choice: a safety constraint on one actuator must never silently corrupt the commands of another.

---

## Section 5 — Final Evaluation & Post-Mortem

### Evaluation Setup

Both policies were evaluated on **30 rollouts** with **seed 42** and identical starting states, matching the BC eval protocol exactly.

---

### Results

| Method | Success Rate | Mean Steps on Success |
|---|---|---|
| BC Baseline | **80.0%** | ~48 |
| Residual + Shield | **46.7%** | ~38.8 |
| **Degradation** | **−33.3 pp** | −9.2 steps |

The residual policy and safety shield together induced a **33.3 percentage point regression** relative to the frozen BC baseline. Notably, `mean_steps_on_success` dropped from ~48 to ~38.8, indicating that even the successful rollouts terminated earlier — a sign the system was being forced into suboptimal trajectories before completing the task, not just failing at grasp.

---

### Diagnostic Analysis — Why Did the System Regress?

A successful training phase (stable Q-values, bounded delta, converging actor/critic losses) followed by catastrophic live deployment is a classic symptom of two interacting offline RL failure modes:

---

#### Failure Mode 1 — Safety Shield as a Kinematic Straightjacket

The shield bounded actions to `[expert_min − margin, expert_max + margin]` per dimension, where the margin was only 2% of the expert range. The expert min/max is **not** the robot's kinematic limit — it is the boundary of one human demonstrator's specific movement style.

When the BC policy encountered a slight out-of-distribution state (inevitable due to covariate shift), it needed to output an action marginally outside the historical dataset to recover. The shield hard-clipped this recovery action back to the dataset boundary. The arm was then effectively frozen at that clipped command — unable to move further — until the episode timed out. The shield did not protect the robot from unsafe actions; it trapped the arm in a deadlock every time it attempted a recovery manoeuvre outside the nominal human envelope.

**Root cause:** Dataset boundary ≠ kinematic safety limit.

---

#### Failure Mode 2 — Offline TD3 Q-Value Overestimation

During training, `q_mean` grew monotonically to **+0.729** on the static offline dataset. This looks healthy, but without live environment interaction the critic cannot observe the consequences of its recommended actions. The actor learned to output deltas that maximized this proxy Q-function by exploiting the critic's blind spots — regions of state space the static dataset never visited.

When deployed in the live MuJoCo environment, these "optimal" offline nudges pushed the end-effector into novel state spaces where the BC backbone's predictions degraded rapidly. The two failures compounded: the RL-recommended nudge pushed the arm toward a blind spot, and the shield then prevented the BC from executing its natural recovery, inducing early episode termination.

**Root cause:** Offline Q-function does not generalize to live state distribution.

---

#### Interaction Effect

Neither failure alone would necessarily have caused a 33.3 pp regression. It was their **collision**: the RL agent exploited offline blind spots to push the arm to the edge of the dataset envelope → the shield hard-clipped the resulting recovery attempt → the arm deadlocked. The two mechanisms amplified each other.

---

### Real-World Compatibility

| Pipeline | P99 Latency | Overhead vs BC |
|---|---|---|
| Frozen BC Backbone | 0.77 ms | — |
| Residual + Shield (combined) | 1.49 ms | +0.72 ms (+93%) |

Both pipelines are well within real-time control budgets (typically 10–20 ms for MuJoCo; <5 ms for hardware). The 0.72 ms overhead of the residual + shield is acceptable for production deployment.

---

### Corrective Next Steps

Given more time, the following two structural changes would be implemented:

#### 1. Shield Relaxation — Replace Dataset Bounds with Kinematic Limits

Transition the safety shield from an empirical dataset boundary to true kinematic safety limits (e.g., maximum safe joint velocities and position ranges from the robot URDF). This fully decouples the safety constraint from the demonstrator's movement style and gives the BC policy the slack it needs to execute recovery manoeuvres.

#### 2. TD3+BC Actor Loss — Anchor RL to the BC Prior

Modify the residual's actor loss to include a behaviour-cloning penalty term:

```
L_actor = -Q(s, a_BC + δ) + λ · ||δ||²
```

This penalizes the residual for pushing the state too far from the dataset distribution, anchoring the RL agent close to the stable BC prior while still permitting micro-corrections. This directly addresses the offline overestimation failure without requiring live environment rollouts during training.
