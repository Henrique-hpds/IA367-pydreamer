**On-Policy (PPO) vs Imagined-Rollout (Dreamer) — Behavior & Implementation**

- **Purpose:** Explain the runtime difference between `--ppo_on_policy True` and `--ppo_on_policy False`, how the code implements each path, how PPO is integrated, and practical notes for running and debugging.

**Quick Summary**
- **`ppo_on_policy=True`**: Actor & critic are trained on real trajectories collected by the `generator` processes. Multi-epoch minibatch PPO updates (importance sampling using stored `action_logp`) are executed inside `Dreamer.training_step()` via `ActorCritic.ppo_update()`.
- **`ppo_on_policy=False`** (default Dreamer): The world model is trained on real episodes, but the policy is trained from imagined rollouts produced by the learned world model (`dream()`), usually with a single pass per training batch.

**High-level data flow**
- Generator processes collect episodes (saved as `.npz` via Mlflow artifacts).
- `train.py` DataLoader yields batches of real episodes to `Dreamer.training_step()`.
- `Dreamer.training_step()` always trains the WorldModel (`self.wm.training_step(...)`) from the real batch.

Modes:
- `ppo_on_policy=True`:
  - `features_real = features.select(2, 0)` extracts real-world features (shape: `(T+1, B, F)`).
  - `action_logp` is read from the batch (`obs['action_logp']`) — generators write this if available. If missing, a fallback reconstructs approximate `logp_old` from the current policy.
  - `self.ac.ppo_update(features_real.detach(), obs['action'][1:].detach(), obs['reward'].detach(), obs['terminal'].detach(), logp_old[1:].detach(), optimizer_actor=self.optimizer_actor, optimizer_critic=self.optimizer_critic, ...)` performs multi-epoch minibatch PPO updates.
  - Actor & critic optimizers are created in `Dreamer.init_optimizers()` and stored on the `Dreamer` instance; when `ppo_on_policy=True` they are intentionally withheld from `train.py`'s optimizer list so they are only updated inside `ppo_update()`.
  - Dream rollouts may still be produced for logging, but the primary policy updates come from real data.

- `ppo_on_policy=False`:
  - `Dreamer.dream(...)` is used to produce imagined rollouts (from current world model) and those imagined transitions are fed to `ActorCritic.training_step(...)` to compute actor/critic losses.
  - Actor & critic optimizers are returned to `train.py` and stepped together with world-model/probe optimizers as part of the global optimization step.

**Code pointers (where to look)**
- World model training: `pydreamer/models/worldmodel` code called from `pydreamer/models/dreamer.py` (the call to `self.wm.training_step(...)`).
- On-policy branch: `pydreamer/models/dreamer.py` around the `if getattr(self.conf,'ppo_on_policy',False)` conditional.
- PPO implementation: `pydreamer/models/a2c.py` — `ppo_update()` implements multi-epoch minibatching, GAE computation, clipped surrogate loss.
- Generator outputs: `generator.py` — collects episodes and (optionally) writes `action_logp`.
- Training loop orchestration: `train.py` — yields batches and calls `model.training_step(...)` and handles backward/optimizers.

**Implementation details — what actually changes when toggling the flag**
- Optimizer wiring:
  - `ppo_on_policy=True` → `Dreamer.init_optimizers()` returns optimizer tuples that exclude actor/critic from the top-level optimizer list so `train.py` won't step them.
  - `ppo_on_policy=False` → actor/critic optimizers returned and stepped normally by `train.py`.

- Data used for policy updates:
  - `True` → Real episodes only. Actor/Critic updated with multi-epoch PPO using stored/reconstructed `action_logp` for importance ratios.
  - `False` → Imagined rollouts created from world model and used to update actor/critic (Dreamer approach).

- Advantages & value targets:
  - `True` → GAE computed from real rewards and critic target (within `ppo_update()`).
  - `False` → GAE computed from imagined rewards in the dreamed trajectories.

**Practical notes and gotchas**
- Data freshness: The world model always trains on the batches your DataLoader provides. If you want training to use only newest episodes, change the episode repository / data sampler to prefer recent episodes.
- `action_logp` availability: generator should write `action_logp` for proper importance weights. The code includes a fallback that reconstructs `logp_old` from the current policy if that field is absent — but this is only an approximation.
- Backprop expectations: `train.py` expects to call `.backward()` on all losses returned. When `ppo_on_policy=True`, actor/critic are updated internally; to avoid backward errors we return zero tensors with `requires_grad=True` so the outer loop can still call backward safely without stepping actor/critic.
- Numerical stability: Multi-epoch updates can destabilize if advantages or log-ratios explode. The implementation includes some protections (log-ratio clamping, gradient clipping, logits clamping) — but monitor for NaNs and add further guards if necessary.
- Dream sampling after PPO updates: Dreamed rollouts use the current world model weights at the time `dream()` is called. If you observe sampling failures after PPO updates, try skipping dream logging during on-policy runs or add NaN/Inf checks before sampling.

**Recommended diagnostics to monitor**
- PPO metrics: actor loss, critic loss, policy entropy, KL divergence between old/new policies (add in `ppo_update()`), clip fraction (how often ratio is clipped).
- World-model metrics: reconstruction loss, KL, posterior/prior entropies.
- Value estimates: `policy_value` on real vs imagined states.
- Training stability: check gradients (norms), check for NaN/Inf in parameters/logits.

**Common fixes for errors you may see**
- "element 0 of tensors does not require grad": ensure returned losses are tensors with `requires_grad=True` when actor/critic are updated internally (this repo updates `dreamer.training_step` accordingly).
- "probability tensor contains inf or nan or <0" during sampling in `dream()`: check for NaN/Inf in actor logits, clamp logits, or skip dreaming when on-policy until code stabilizes.
- DataLoader warnings about `pin_memory` with no accelerator: harmless on CPU, or set `pin_memory=False`.

**How to run**
- On-policy PPO (real-trajectory updates):

```powershell
python launch.py --configs defaults atari debug --env_id Atari-Pong --ppo_on_policy True
```

- Imagined-rollout Dreamer (default):

```powershell
python launch.py --configs defaults atari debug --env_id Atari-Pong
```

**If you want stricter "train only on newest episodes" behavior**
- Modify `MlflowEpisodeRepository` / `pydreamer/data.py` sampling logic to sort or filter episodes by timestamp and return only the latest N episodes, or implement an in-memory ring buffer that keeps only the most recent episodes.

**Appendix — concise mapping of responsibilities**
- `generator.py`: collect episodes, optionally store `action_logp`.
- `train.py`: reads episodes, builds batches, calls `model.training_step(...)`, and runs global optimizer steps for returned optimizers.
- `pydreamer/models/dreamer.py`: orchestrates world model training, switches between on-policy PPO vs imagined-rollout actor training, houses `dream()` and `init_optimizers()` logic.
- `pydreamer/models/a2c.py`: `ActorCritic` model — contains policy and value networks, `training_step()` for single-pass updates (imagined rollouts), and `ppo_update()` for multi-epoch on-policy updates.