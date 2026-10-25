# The project at state 0: what the code does, in detail

This document describes the code as it was in 2022, with only the portability and pure speed
fixes listed in the README. It is the reference: every future improvement will be described
relative to it.

---

## 1. The game: Crafter

Crafter (Hafner, 2021) is a simplified 2D Minecraft, designed to measure the range of an agent's
abilities: exploration, survival, long-term planning.

- **World**: 64×64 cells, regenerated for every episode (grassland, forests, lakes, mountains,
  caves, lava). Day and night alternate; at night the image darkens and zombies are more numerous.
- **Observation**: a 64×64 RGB image. It shows a 9×7-cell window around the player and, below
  it, the inventory on 9×2 cells: health, food, drink, energy, items.
- **17 actions**: no-op; move (4 directions); `do`, which hits, collects, drinks or eats
  depending on the facing cell; sleep; place stone, table, furnace or sapling; craft a wood,
  stone or iron pickaxe or sword.
- **Survival**: food, drink and energy decrease over time. If one of them reaches 0, health
  decreases; if all are above 0, health recovers. Zombies (on grass) and skeletons (in caves)
  attack; walking into lava kills instantly.
- **End of an episode**: on death, or after 10,000 steps.
- **Reward**:
  - +1 the first time each of the 22 achievements is unlocked in the episode;
  - ±0.1 per health point gained or lost.

  The maximum reward of an episode is therefore about 22.
- **The 22 achievements**, which form a technology tree:
  - collect: wood, sapling, drink, stone, coal, iron, diamond;
  - place: table, stone, sapling (plant), furnace;
  - make: wood pickaxe and sword, then stone, then iron;
  - eat a cow, eat a plant;
  - defeat a zombie, defeat a skeleton;
  - wake up after sleeping.

  Example chain: wood → table → wood pickaxe → stone → stone pickaxe → iron (also needs a furnace
  and coal) → iron pickaxe → diamond.
- **Crafter score**: geometric mean of the success rates of the 22 achievements, offset by 1%
  (`exp(mean(ln(1 + s_i))) − 1`), computed over all training episodes of the first million
  steps. Because it is a geometric mean, unlocking a rare achievement counts much more than
  doing even better on an easy one. It is the benchmark metric, more informative than the reward.

## 2. Pipeline overview

```
train_ppo.py / train_dqn.py
   │
   ├── Env("train")  ─ src/crafter_wrapper.py
   │      crafter.Env()                 → 64×64 RGB image, uint8
   │      crafter.Recorder(...)         → writes logdir/.../stats.jsonl (1 line per episode)
   │      GrayScale                     → mean of the 3 channels → 64×64
   │      Env (stack of 4 frames, /255) → float32 tensor (4, 64, 64), between 0 and 1
   │
   ├── Env("eval")  ─ same, without Recorder
   │
   ├── agent  ─ agents/*.py + networks/*.py (+ memories/ppo_memory.py)
   │
   └── loop: act → env.step → store → learn → evaluate every --eval-interval steps
```

- **A single environment**, played step by step. The agent picks each action with one network
  pass on a single image (batch of 1).
- **Evaluation**: every `--eval-interval` steps, `eval()` plays `--eval-episodes` episodes
  (20 by default) on the evaluation environment and appends the mean reward to
  `logdir/.../eval_stats.pkl`.
- **Stack of 4 frames**: gives a sense of motion, e.g. which way a zombie is moving. At the start
  of an episode, the 3 previous frames are zeros.
- **Grayscale**: divides the input size by 3. In exchange it loses colour, which distinguishes
  important elements: iron (orange) from coal (black) and stone (grey), water from grass at
  night, etc.

## 3. The convolutional network (shared by the three agents)

"Nature DQN" architecture (Mnih et al., 2015):

| Layer | Parameters | Output |
|---|---|---|
| Input | 4 frames 64×64 | 4 × 64 × 64 |
| Conv 1 | 32 filters 8×8, stride 4, ReLU | 32 × 15 × 15 |
| Conv 2 | 64 filters 4×4, stride 2, ReLU | 64 × 6 × 6 |
| Conv 3 | 64 filters 3×3, stride 1, ReLU | 64 × 4 × 4 = 1,024 |
| Dense | 1,024 → 512, ReLU | 512 |
| Output | 512 → 17 Q-values (DQN), 17 probabilities (PPO actor) or 1 value (PPO critic) | |

About 0.6 million parameters per network, most of them in the dense layer (1,024 × 512). A
useful remark for later: with a stride of 4 in the first layer, each game cell (about 7×7
pixels) is heavily downsampled. The network sees the world very coarsely.

## 4. DQN — `agents/dqn_agent.py`, run with `train_dqn.py --agent dqn`

**Principle.** Learn Q(s, a), the sum of future rewards (discounted by γ) when playing a in
state s and acting optimally afterwards. The target comes from the Bellman equation:
`Q(s,a) ← r + γ · max_a' Q(s', a')`.

**Implementation.**
- **Replay memory**: numpy arrays of 100,000 transitions (s, a, r, s', done). States s and s'
  are stored separately in float32, about 13 GB.
- **Action selection**: ε-greedy. With probability ε the action is random, otherwise `argmax Q`.
- **Learning**: at every environment step, once the memory holds 32 transitions, one update on
  a random batch of 32.
- **Target**: `r + 0.99 · max Q(s') · (1 − done)`, computed with **the same network** that is
  being trained. There is no target network.
- **Loss and optimiser**: Huber (`SmoothL1Loss`) and RMSprop, learning rate 2.5e-4.
- **Exploration**: ε starts at 1 and loses 5e-4 at every update, down to 0.1. It therefore
  reaches 0.1 after about 1,800 steps. The constant `EPSILON_DECAY = 1e-6` in `train_dqn.py` is
  never passed to the agent.
- **Evaluation**: the docstring mentions a greedy policy, but `act()` still applies ε. The agent
  is therefore evaluated with 10% random actions.

**What goes wrong.**
- **Without a target network**, the target moves at every update: the network chases its own
  tail. This is the classic instability that DeepMind's DQN solves by freezing a copy of the
  network for several thousand steps.
- **Exploration stops almost immediately**, whereas Crafter requires a lot of exploration.

## 5. "Double DQN" — `agents/ddqn_agent.py`, run with `train_dqn.py --agent ddqn`

**Expected principle.** Double DQN (van Hasselt, 2016) corrects DQN's overestimation. The
online network selects the best next action, and the target network evaluates it:
`r + γ · Q_target(s', argmax_a Q_online(s', a))`.

**Implementation.**
- **Two networks**: Q1, trained, and Q2, with gradients disabled.
- **Copy**: after **every** update, Q2 receives a copy of Q1. The two networks are therefore
  always identical.
- **Target**: `r + γ · max Q2(s') · done`. This is DQN's `max`, not Double DQN's
  selection/evaluation split.
- **Loss**: squared error written by hand. RMSprop, same ε and same memory as DQN.

**What goes wrong.**
1. **The target is inverted**: `· done` instead of `· (1 − done)`. The future value is only
   added on the transition that ends the episode, i.e. exactly where it should be 0. Everywhere
   else the target is just r. The agent therefore mostly learns the immediate reward, without
   looking further than the next step.
2. **No real target network**, since Q2 = Q1 at all times.
3. **It is not the Double DQN algorithm**: action selection and evaluation are not decoupled.

Despite this, DDQN does slightly better than DQN in 2022 (2.7% vs 2.3%). This is consistent: a
myopic agent that collects what pays immediately (sapling, wood, water) beats an unstable one.

## 6. PPO — `agents/ppo_agent.py`, run with `train_ppo.py`

**Principle.** PPO (Schulman, 2017) directly learns a policy π(a|s), the actor, together with a
value function V(s), the critic. A "rollout" is played with the current policy. For each step,
the advantage A measures how much better than expected the action turned out. The probability
of actions with a positive advantage is then increased, without moving too far from the old
policy: this is the role of *clipping* the ratio π_new / π_old to [0.8, 1.2].

**Implementation.**
- **Two separate networks**, no shared trunk: the actor (CNN then softmax over 17 actions) and
  the critic (CNN then one value). Each has its own Adam optimiser (lr 3e-4).
- **Rollout**: 2,048 steps in a single environment, then learning.
- **Advantages**: GAE with γ = 0.99 and λ = 0.95. In 2022 they were computed with a Python double
  loop in O(n²), about 2 million iterations, **redone at each of the 10 epochs** although the
  result is the same every time. This took about 20 s per rollout. **It is now done once, in
  O(n)**, with bit-identical values: a pure optimisation that does not change learning.
- **Update**: 10 epochs × 32 mini-batches of 64, i.e. 320 gradient steps per rollout.
- **Loss**: `actor_loss (clip 0.2) + 0.5 × critic_loss (MSE)`. There is **no entropy bonus**
  (the comment `# Entropy ?` is still there).
- **Gradients**: clipped to a norm of 0.5 for each of the two networks.
- **Actions**: always sampled from π, including during evaluation.
- **Saving**: only at the very end. In 2022 to `checkpoints/ppo/`; now to
  `logdir/.../checkpoints/`, so that parallel runs do not overwrite each other.

**What goes wrong.**
- **Without an entropy bonus**, the policy quickly becomes almost deterministic and stops
  exploring.
- **Advantages are not normalised**: their scale varies from one rollout to the next, which
  makes the effective learning step irregular.
- **No bootstrap at the end of a rollout**: the advantage of the last step is 0 and the last
  steps are underestimated. The effect is small with 2,048 steps.
- **GAE does not stop at episode boundaries**: the next-state value is correctly cut by
  `(1 − done)`, but the discounted sum of δ continues past the end of an episode. The first
  steps of an episode therefore influence the advantage of the last steps of the previous one.
  The standard formula also multiplies the γλ factor by `(1 − done)`.
- **A single environment**: the 2,048 steps of a rollout are highly correlated (one or two
  episodes), which gives little diversity per update.

PPO was the best 2022 agent (3.2%). It is also the algorithm with the most headroom here: the PPO
published with Crafter reaches 4.6%, and a well-tuned PPO reaches about 15% with the same budget.

## 7. Output files and analysis tools

| File | Written by | Content |
|---|---|---|
| `logdir/<agent>/<n>/stats.jsonl` | `crafter.Recorder` | 1 line per training episode: length, reward, count of each achievement |
| `.../eval_stats.pkl` | `eval()` | mean reward of each evaluation (stacked pickles) |
| `.../loss_log.txt`, `q_values_log.txt` | DQN/DDQN | one value per update (huge files: about 1M lines) |
| `.../actor_/critic_/total_losses_log.txt` | PPO | one value per rollout |
| `.../checkpoints/actor_torch_ppo`, `critic_torch_ppo` | PPO | network weights at the end of training |

- `analysis/crafter_score.py`: official Crafter score and per-achievement success rates, from
  the `stats.jsonl` files. The README scores come from it.
- `analysis/learning_curve.py`: score, reward, episode length and top achievements per window
  of steps, for one or several groups of runs (table + figure).
- `analysis/plot_eval_performance.py`: evaluation reward curve (mean over runs), achievement
  histogram, loss curve.

During training, the terminal shows:
- every `--log-every` steps (5,000 by default), a progress line: speed, estimated time left,
  mean reward of the last 20 episodes, time split (see section 9);
- every 20 episodes, the original `Train mean_reward` line, which `--log-episodes 0` hides;
- at each evaluation, the `eval results: R/ep=…` line.

## 8. 2022 results

Crafter score over the first million steps:

| Agent | Score | Profile (success rates) |
|---|---|---|
| Random | 1.6% | wake up 94%, sapling 51%, wood 24%, drink 10%, table 4% |
| DQN | 2.2 – 2.3% | sapling 93%, wood 30%, drink 11%, cow 8%, table 7% |
| DDQN | 2.6 – 2.7% | sapling 99%, wood 34%, table 12%, cow 11%, zombie 10% |
| PPO (best run) | 3.2% | wake up 94%, wood 59%, table 31%, drink 28%, wood pickaxe 5% |

No agent reliably reaches stone: even the best PPO run leaves 12 of the 22 achievements below
1%, 8 of which are never unlocked.

## 9. Where the time goes, and what the GPU brings

PPO, one run, measured on the RTX 4090 machine (3 runs in parallel):

| Version | Steps/s per run | Split | 1M steps |
|---|---|---|---|
| 2022 code (O(n²) GAE) | ~35 | dominated by `learn` | ~8 h |
| GAE in O(n) | ~105 | env 63%, act 20%, learn 17% | ~2 h 40 |
| + numba (fast world generation) | ~170 | act 38%, env 27%, learn 35% | ~1 h 35 |

- **`env`**: Crafter simulation and image rendering, in Python, on a single CPU core. Most of it
  used to be world generation (about 0.8 s per episode in pure Python, 0.05 s with numba).
- **`act`**: one CNN pass for a single image, twice (actor and critic), with three round trips
  to the GPU (`.item()`). On the GPU, such a tiny computation is dominated by the fixed cost of
  each call, so the GPU gains little here.
- **`learn`**: the network part (320 mini-batches of 64), which benefits from the GPU.

The GPU will only become decisive with several environments running in parallel and larger
batches of images: that is a separate step, because it changes how the agent collects its data.
