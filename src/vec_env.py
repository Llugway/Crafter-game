"""Several Crafter environments stepped in parallel worker processes.

Each worker process hosts a few environments. The main process sends one action per
environment and gets back, for each one, the raw RGB image, the reward, the done flag and,
when an episode ends, its statistics in the same format as crafter.Recorder's stats.jsonl.
Finished environments are reset automatically by the worker.

Preprocessing (grayscale + stack of 4 frames, values in [0, 1]) is done in the main process
and reproduces src/crafter_wrapper.Env exactly, so the agent sees the same observations as
with a single environment.

Works on Windows (spawn) and Linux.
"""
import multiprocessing as mp

import numpy as np


def _episode_stats(length, reward, achievements):
    # same fields and rounding as crafter.recorder.StatsRecorder
    stats = {"length": length, "reward": round(reward, 1)}
    for key, value in achievements.items():
        stats[f"achievement_{key}"] = value
    return stats


def _worker(conn, n_envs, seed):
    import crafter  # imported in the worker process

    envs = [crafter.Env(seed=None if seed is None else seed + i) for i in range(n_envs)]
    lengths = [0] * n_envs
    rewards = [0.0] * n_envs
    conn.send([env.reset() for env in envs])
    while True:
        cmd, data = conn.recv()
        if cmd == "close":
            conn.close()
            return
        out = []
        for i, (env, action) in enumerate(zip(envs, data)):
            obs, reward, done, info = env.step(int(action))
            lengths[i] += 1
            rewards[i] += info["reward"]
            stats = None
            if done:
                stats = _episode_stats(lengths[i], rewards[i], info["achievements"])
                lengths[i], rewards[i] = 0, 0.0
                obs = env.reset()
            out.append((obs, reward, done, stats))
        conn.send(out)


class FrameStack:
    """Observation preprocessing for N environments, shared by training and evaluation.

    rgb=False (2022): grayscale, values in [0, 1], `history` frames -> [N, history, 64, 64].
    rgb=True: colour channels kept, values in [0, 1], `history` frames -> [N, 3*history, 64, 64].
    """

    def __init__(self, num_envs, history, rgb=False):
        self.history, self.rgb = history, rgb
        self.c = 3 if rgb else 1
        self.stack = np.zeros((num_envs, self.c * history, 64, 64), dtype=np.float32)

    @staticmethod
    def _gray(rgb):
        # identical to crafter_wrapper: GrayScale (mean over channels, float64), then
        # torch.tensor(..., dtype=float32).div_(255)
        return np.float32(rgb.mean(-1)) / np.float32(255)

    def push(self, i, rgb):
        c = self.c
        self.stack[i, :-c] = self.stack[i, c:]
        if self.rgb:
            self.stack[i, -c:] = rgb.transpose(2, 0, 1).astype(np.float32) / np.float32(255)
        else:
            self.stack[i, -1] = self._gray(rgb)

    def new_episode(self, i, rgb):
        # same as crafter_wrapper.Env.reset: zeros, then the first frame
        self.stack[i] = 0.0
        self.push(i, rgb)

    def observations(self):
        return self.stack.copy()


def evaluate(act_fn, episodes, history, rgb, seed=None):
    """Play `episodes` evaluation episodes, one after the other, in this process.
    act_fn(observations [1, C, 64, 64]) -> actions array [1]. Returns the episode returns."""
    import crafter
    env = crafter.Env(seed=seed)
    frames = FrameStack(1, history, rgb)
    returns = []
    for _ in range(episodes):
        frames.new_episode(0, env.reset())
        done, total = False, 0.0
        while not done:
            action = act_fn(frames.observations())[0]
            rgb_obs, reward, done, _ = env.step(int(action))
            total += reward
            frames.push(0, rgb_obs)
        returns.append(total)
    return returns


class VecCrafter:
    """`num_envs` Crafter environments in `num_workers` processes, with frame stacking."""

    def __init__(self, num_envs, num_workers=None, history_length=4, seed=None, rgb=False):
        num_workers = max(1, min(num_workers or num_envs, num_envs))
        self.num_envs = num_envs
        self.history = history_length
        ctx = mp.get_context("spawn")
        # split the environments as evenly as possible between workers
        sizes = [num_envs // num_workers + (1 if w < num_envs % num_workers else 0) for w in range(num_workers)]
        self.sizes = [s for s in sizes if s > 0]
        self.conns, self.procs = [], []
        offset = 0
        for size in self.sizes:
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=_worker, args=(child, size, None if seed is None else seed + offset),
                               daemon=True)
            proc.start()
            child.close()
            self.conns.append(parent)
            self.procs.append(proc)
            offset += size
        first = sum((c.recv() for c in self.conns), [])
        self.frames = FrameStack(num_envs, history_length, rgb)
        self.obs_channels = self.frames.stack.shape[1]
        for i, rgb_obs in enumerate(first):
            self.frames.push(i, rgb_obs)

    @property
    def stack(self):
        return self.frames.stack

    def observations(self):
        """Current observations, float32 array [num_envs, channels, 64, 64]."""
        return self.frames.observations()

    def step(self, actions):
        """Returns rewards [N], dones [N] and the list of finished-episode stats."""
        start = 0
        for conn, size in zip(self.conns, self.sizes):
            conn.send(("step", list(actions[start:start + size])))
            start += size
        results = sum((c.recv() for c in self.conns), [])
        rewards = np.zeros(self.num_envs, dtype=np.float64)
        dones = np.zeros(self.num_envs, dtype=bool)
        finished = []
        for i, (rgb, reward, done, stats) in enumerate(results):
            rewards[i], dones[i] = reward, done
            if done:
                finished.append(stats)
                self.frames.new_episode(i, rgb)
            else:
                self.frames.push(i, rgb)
        return rewards, dones, finished

    def close(self):
        for conn in self.conns:
            try:
                conn.send(("close", None))
            except (BrokenPipeError, EOFError):
                pass
        for proc in self.procs:
            proc.join(timeout=5)
