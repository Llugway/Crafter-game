"""Progress line printed during training, plus a breakdown of where the time goes.

Purely observational: it does not change what the agent does.

    progress = Progress(total_steps=opt.steps, every=opt.log_every, device=agent_device)
    with progress.time("env"):   obs, r, done, info = env.step(a)
    with progress.time("act"):   a = agent.act(obs)
    with progress.time("learn"): agent.learn()
    progress.step(reward, done)   # once per environment step

Every `every` steps it prints, for example:

    [ 120000/1000000  12.0%] 341 pas/s | 5m52s écoulé, reste ~43m | 652 épisodes | reward (20 dern.) 2.45 | env 48% act 21% learn 31% | cuda
"""
import time
from collections import deque, defaultdict
from contextlib import contextmanager


def _fmt(seconds):
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


class Progress:
    def __init__(self, total_steps, every=10_000, device="cpu", window=20):
        self.total = total_steps
        self.every = every
        self.device = str(device)
        self.steps = 0
        self.episodes = 0
        self.ep_return = 0.0
        self.returns = deque(maxlen=window)
        self.window = window
        self.t0 = time.perf_counter()
        self.t_last = self.t0
        self.steps_last = 0
        self.timers = defaultdict(float)
        self.timers_total = defaultdict(float)

    @contextmanager
    def time(self, name):
        t = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t
            self.timers[name] += dt
            self.timers_total[name] += dt

    def step(self, reward, done):
        self.steps += 1
        self.ep_return += reward
        if done:
            self.episodes += 1
            self.returns.append(self.ep_return)
            self.ep_return = 0.0
        if self.every and self.steps % self.every == 0:
            self.print_line()

    def print_line(self):
        now = time.perf_counter()
        elapsed = now - self.t0
        sps = (self.steps - self.steps_last) / max(now - self.t_last, 1e-9)
        remaining = (self.total - self.steps) / max(self.steps / elapsed, 1e-9)
        pct = 100 * self.steps / self.total if self.total else 0
        mean_ret = sum(self.returns) / len(self.returns) if self.returns else float("nan")
        tracked = sum(self.timers.values())
        split = " ".join(f"{k} {100 * v / tracked:.0f}%" for k, v in sorted(self.timers.items())) if tracked else ""
        print(f"[{self.steps:>8d}/{self.total} {pct:5.1f}%] {sps:5.0f} pas/s | "
              f"{_fmt(elapsed)} écoulé, reste ~{_fmt(remaining)} | {self.episodes} épisodes | "
              f"reward ({self.window} dern.) {mean_ret:.2f} | {split} | {self.device}", flush=True)
        self.t_last, self.steps_last = now, self.steps
        self.timers.clear()

    def summary(self):
        elapsed = time.perf_counter() - self.t0
        tracked = sum(self.timers_total.values()) or 1e-9
        parts = ", ".join(f"{k} {v:.0f}s ({100 * v / tracked:.0f}%)" for k, v in sorted(self.timers_total.items()))
        print(f"Terminé : {self.steps} pas en {_fmt(elapsed)} ({self.steps / max(elapsed, 1e-9):.0f} pas/s). "
              f"Temps mesuré : {parts}.", flush=True)
