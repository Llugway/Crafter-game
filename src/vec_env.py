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


class VecCrafter:
    """`num_envs` Crafter environments in `num_workers` processes, with frame stacking."""

    def __init__(self, num_envs, num_workers=None, history_length=4, seed=None):
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
        self.stack = np.zeros((num_envs, history_length, 64, 64), dtype=np.float32)
        for i, rgb in enumerate(first):
            self._push(i, rgb)

    @staticmethod
    def _gray(rgb):
        # identical to crafter_wrapper: GrayScale (mean over channels, float64), then
        # torch.tensor(..., dtype=float32).div_(255)
        return np.float32(rgb.mean(-1)) / np.float32(255)

    def _push(self, i, rgb):
        self.stack[i, :-1] = self.stack[i, 1:]
        self.stack[i, -1] = self._gray(rgb)

    def observations(self):
        """Current stacked observations, float32 array [num_envs, history, 64, 64]."""
        return self.stack.copy()

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
                # new episode: same as crafter_wrapper.Env.reset (zeros, then the first frame)
                self.stack[i] = 0.0
                finished.append(stats)
            self._push(i, rgb)
        return rewards, dones, finished

    def close(self):
        for conn in self.conns:
            try:
                conn.send(("close", None))
            except (BrokenPipeError, EOFError):
                pass
        for proc in self.procs:
            proc.join(timeout=5)
