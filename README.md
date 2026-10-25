# Crafter — reinforcement learning agents

DQN, Double DQN and PPO agents trained on [Crafter](https://github.com/danijar/crafter), a 2D
Minecraft built as a benchmark: the agent must survive (eat, drink, sleep, fight) and chain 22
achievements of increasing difficulty (wood → table → pickaxe → stone → iron → diamond).

The project started as a master's assignment in AI (2022). The goal now is to improve it
**step by step**, understanding at each step what changes and why.

## Starting point (2022)

Crafter score: geometric mean of the success rates of the 22 achievements, over all training
episodes of the first million steps. Computed by `analysis/crafter_score.py` on the runs in
`results/2022/runs/`.

| Agent | Runs | Crafter score | Mean reward (last 100 episodes) |
|---|---|---|---|
| Random | 4 | 1.6% | 1.4 |
| DQN | 2 (≤ 600k steps) | 2.2 – 2.3% | 2.6 – 2.8 |
| Double DQN | 2 | 2.6 – 2.7% | 2.7 – 3.1 |
| **PPO** | 4 | **2.6 – 3.2%** | 2.4 – 3.2 |
| *Published references (pixels, 1M steps)* | | PPO 4.6% · DreamerV3 14.8% · human 50.5% | |

The best PPO run collects wood (59% of episodes) and places a table (31%), but almost never
crafts tools (5%) and does not reach stone.

## Installation (Windows, conda)

The code runs natively on Windows: no Linux or WSL needed.

```powershell
conda activate crafterEnv          # existing env; otherwise: conda create -n crafterEnv python=3.12
cd Crafter-game

# 1) PyTorch with CUDA (CUDA 13.0 build: `nvidia-smi` must report CUDA >= 13.0).
#    Check the exact command at https://pytorch.org/get-started/locally/ (Windows / Pip / latest CUDA).
pip install torch --index-url https://download.pytorch.org/whl/cu130

# 2) Everything else
pip install -r requirements.txt

# 3) Checks
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import crafter; e = crafter.Env(); print(e.reset().shape)"
```

The first check must print `True` and the GPU name. If it prints `False`, the CPU build of
PyTorch is installed: uninstall it (`pip uninstall torch`) and redo step 1.

`numba` (in `requirements.txt`) is used automatically by `opensimplex`: Crafter's world
generation becomes about 16× faster (0.82 s → 0.05 s per world) and produces identical worlds.

## Running the code

Quick tests (one to two minutes each) to check that everything works:

```powershell
python train_ppo.py --steps 4096 --eval-interval 4096 --eval-episodes 2 --logdir logdir/smoke/ppo/0
python train_dqn.py --agent ddqn --steps 3000 --eval-interval 3000 --eval-episodes 2 --logdir logdir/smoke/ddqn/0
```

Full training runs (1M steps, as in the benchmark):

```powershell
python train_ppo.py --steps 1000000 --eval-interval 50000 --logdir logdir/ppo_agent/1
python train_dqn.py --agent dqn  --steps 1000000 --eval-interval 50000 --logdir logdir/dqn_agent/1
python train_dqn.py --agent ddqn --steps 1000000 --eval-interval 50000 --logdir logdir/ddqn_agent/1

# several runs in parallel (one window per run)
scripts\run_seeds.bat -Runs 3 -Steps 1000000 -EvalInterval 50000     # from cmd / Anaconda Prompt
.\scripts\run_seeds.ps1 -Runs 3 -Steps 1000000 -EvalInterval 50000    # from PowerShell
```

### Several environments in parallel (PPO)

```powershell
python train_ppo.py --num-envs 16 --num-workers 8 --steps 1000000 --eval-interval 50000 --logdir logdir/ppo_vec16/1

# 3 runs of it at once, workers shared out over the CPU automatically -> logdir/ppo_vec16/1..3
scripts\run_seeds.bat -NumEnvs 16 -Runs 3 -Steps 1000000 -EvalInterval 50000
```

`--num-envs N` runs N Crafter environments in `--num-workers` processes (`src/vec_env.py`) and picks
their N actions in one batched network pass. The update is unchanged (2,048 transitions,
10 epochs, mini-batches of 64), but the 2,048 transitions now come from N environments × 2048/N
steps. This changes the data the agent learns from, so it is compared against the reference like
any other step. `--num-envs 1` (default) is the 2022 single-environment loop.

Play the game yourself (useful to understand it): `pip install pygame-ce` once, then
`python -m crafter.run_gui`.

Results and plots:

```powershell
python analysis/crafter_score.py --logdir logdir --budget 1000000 --rates
python analysis/learning_curve.py --logdir logdir/ppo_agent               # score / reward / length per 50k-step window
python analysis/plot_eval_performance.py --logdir logdir/ppo_agent      # -> analysis/ppo_agent/*.png
```

During training, a progress line is printed every 5,000 steps (`--log-every`):

```
[  120000/1000000  12.0%]   341 steps/s | 5m52s elapsed, ~43m left | 652 episodes | reward (last 20) 2.45 | act 21% env 48% eval 0% learn 31% | cuda:0
```

The percentages show where the time went since the previous line:
- `env`: Crafter simulation and rendering, on CPU;
- `act`: action selection by the network;
- `learn`: learning updates;
- `eval`: evaluation;
- `replay` (DQN): writing to the replay memory.

A summary is printed at the end of the run. The original "Train mean_reward" line (every 20
episodes) is kept; `--log-episodes 0` hides it.

### Measuring the GPU speed-up

Run the same short training twice, on GPU then on CPU (`CUDA_VISIBLE_DEVICES` hides the GPU from
PyTorch), and compare steps/s and the time split:

```powershell
python train_ppo.py --steps 50000 --eval-interval 1000000 --logdir logdir/bench/ppo_gpu/0
$env:CUDA_VISIBLE_DEVICES = "-1"
python train_ppo.py --steps 50000 --eval-interval 1000000 --logdir logdir/bench/ppo_cpu/0
Remove-Item Env:CUDA_VISIBLE_DEVICES
```

The last column of the progress line confirms the device used (`cuda:0` or `cpu`).

### Output files

Each run writes to its own folder `logdir/<agent>/<n>/`:
- `stats.jsonl`: one line per training episode (length, reward, achievements);
- `eval_stats.pkl`: mean reward of each evaluation;
- the loss logs;
- `checkpoints/` (PPO): network weights at the end of the run.

**Memory**: DQN and DDQN allocate a replay buffer of 100,000 transitions × 2 × (4×64×64)
float32, about **13 GB of RAM**. Lower `MAX_MEMORY` in `train_dqn.py` if needed.

## Repository layout

```
train_ppo.py            PPO training loop                    (formerly train.py)
train_dqn.py            DQN / Double DQN training loop (--agent) (formerly train_classic.py)
agents/                 dqn_agent, ddqn_agent, ppo_agent, random_agent
networks/               convolutional networks (Q-network, actor, critic)
memories/ppo_memory.py  PPO rollout buffer
src/crafter_wrapper.py  Crafter wrapper: grayscale, 4-frame stack, logging
src/progress.py         progress line and time split
analysis/               crafter_score.py, learning_curve.py, plot_eval_performance.py
scripts/                run_seeds.ps1 / run_seeds.bat: several runs in parallel
results/2022/           runs, PPO checkpoints and plots of the original assignment
docs/                   assignment, rules, reference papers, detailed project description
```

How the code works in detail (game, pipeline, networks, each agent, results, compute time):
[docs/state_0.md](docs/state_0.md).

## Known limitations of the 2022 code (next steps)

Found while reviewing the code and **deliberately not fixed yet**: the current code remains the
measurable starting point, and each fix will be a step compared against it.

- **Double DQN, inverted target** (`agents/ddqn_agent.py`): `q_target = r + γ·max Q'(s') · done`
  instead of `· (1 − done)`. The future value is only counted at the end of an episode, so the
  agent essentially learns the immediate reward only.
- **No real target network**: in DDQN, Q2 is copied from Q1 at every step; DQN has none.
  Learning is therefore unstable.
- **DQN exploration**: `EPSILON_DECAY` is never passed to the agent. ε goes from 1 to 0.1 in
  about 1,800 steps instead of about 900,000.
- **PPO**:
  - no entropy bonus, so the policy freezes early (now available with `--ent-coef`,
    default 0 = 2022 behaviour; the policy entropy is logged to `entropy_log.txt` in every run);
  - advantages not normalised;
  - GAE recomputed in O(n²) at every epoch, about 20 s per rollout: **fixed**, it is now
    computed once in O(n), with bit-identical results;
  - GAE does not stop the discounted sum at episode boundaries (not fixed);
  - no value bootstrap at the end of a rollout;
  - a single environment.
- **Grayscale observations**: colour distinguishes key elements (iron, coal, diamond, water…),
  and grayscale loses it.
- **Under-used GPU**: the environment runs on a single CPU core, so the GPU waits most of the
  time. Several environments in parallel are now available with `--num-envs` (default 1).

The 2022 M-DQN agent was removed: it depended on a `ReplayBuffer` class missing from the
repository. It is still available in the git history (first commit).
