"""Learning curves from training episodes (stats.jsonl), one or several runs.

For every window of `--window` steps it computes, over the episodes that ended in it:
the Crafter score, the mean reward, the mean episode length and the success rate of
each achievement. Prints a table and saves a figure.

    python analysis/learning_curve.py --logdir logdir/ppo_ref
    python analysis/learning_curve.py --logdir logdir/ppo_ref results/2022/runs/ppo_agent --labels "ref 2026" "2022"

Each --logdir is a group: all stats.jsonl below it are averaged (mean ± min/max over runs).
"""
import argparse
import json
import pathlib

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_run(path):
    eps = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    keys = sorted(k for k in eps[0] if k.startswith("achievement_"))
    ends = np.cumsum([e["length"] for e in eps])
    ach = np.array([[e[k] >= 1 for k in keys] for e in eps], dtype=float)
    return {"ends": ends, "reward": np.array([e["reward"] for e in eps]),
            "length": np.array([e["length"] for e in eps]), "ach": ach,
            "names": [k[len("achievement_"):] for k in keys]}


def windows(run, window, budget):
    rows = []
    for hi in range(window, budget + 1, window):
        m = (run["ends"] > hi - window) & (run["ends"] <= hi)
        if m.sum() < 3:
            continue
        rates = 100 * run["ach"][m].mean(0)
        rows.append({"steps": hi, "episodes": int(m.sum()),
                     "score": float(np.exp(np.mean(np.log1p(rates))) - 1),
                     "reward": float(run["reward"][m].mean()), "length": float(run["length"][m].mean()),
                     "rates": rates})
    return rows


if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--logdir", nargs="+", required=True)
    pa.add_argument("--labels", nargs="+", default=None)
    pa.add_argument("--window", type=int, default=50_000)
    pa.add_argument("--budget", type=int, default=1_000_000)
    pa.add_argument("--out", default="analysis/learning_curve.png")
    pa.add_argument("--rollout", type=int, default=2048, help="Transitions per PPO update (x axis of the entropy plot).")
    cfg = pa.parse_args()
    labels = cfg.labels or [pathlib.Path(d).name for d in cfg.logdir]

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    for d, label in zip(cfg.logdir, labels):
        files = sorted(pathlib.Path(d).glob("**/stats.jsonl"))
        print(f"\n{label}: {len(files)} stats.jsonl found under {pathlib.Path(d).resolve()}")
        if not files:
            print("  -> nothing to plot for this group: check the path (run from the repository root?)")
            continue
        runs = [windows(load_run(f), cfg.window, cfg.budget) for f in files]
        runs = [r for r in runs if r]
        if not runs:
            print(f"  -> runs too short for a {cfg.window:,}-step window")
            continue
        n = min(len(r) for r in runs)
        steps = np.array([runs[0][i]["steps"] for i in range(n)])
        names = load_run(files[0])["names"]
        print(f"\n=== {label}: {len(runs)} run(s), mean per window of {cfg.window:,} steps ===")
        print(f"{'steps':>9s} {'score %':>8s} {'reward':>7s} {'length':>6s}  top achievements (success rate %)")
        for i in range(n):
            score = np.mean([r[i]["score"] for r in runs])
            reward = np.mean([r[i]["reward"] for r in runs])
            length = np.mean([r[i]["length"] for r in runs])
            rates = np.mean([r[i]["rates"] for r in runs], 0)
            top = ", ".join(f"{names[j]} {rates[j]:.0f}" for j in np.argsort(-rates)[:6] if rates[j] > 0)
            print(f"{steps[i]:9d} {score:8.2f} {reward:7.2f} {length:6.0f}  {top}")
        for ax, key, title in zip(axes, ("score", "reward", "length"),
                                  ("Crafter score (%)", "Mean episode reward", "Mean episode length (steps)")):
            vals = np.array([[r[i][key] for i in range(n)] for r in runs])
            line, = ax.plot(steps, vals.mean(0), label=f"{label} ({len(runs)} runs)")
            if len(runs) > 1:
                ax.fill_between(steps, vals.min(0), vals.max(0), alpha=0.2, color=line.get_color())
            ax.set_title(title)
            ax.set_xlabel("training steps")
            ax.grid(alpha=0.3)
        # policy entropy per PPO update (entropy_log.txt), if the runs logged it
        ent = [np.loadtxt(f.parent / "entropy_log.txt", ndmin=1) for f in files if (f.parent / "entropy_log.txt").exists()]
        ent = [e for e in ent if len(e)]
        # transitions per PPO update: from the run's config.json if it has one (newer runs),
        # otherwise --rollout
        conf = files[0].parent / "config.json"
        rollout = json.load(open(conf)).get("rollout", cfg.rollout) if conf.exists() else cfg.rollout
        if ent:
            m = min(min(len(e) for e in ent), cfg.budget // rollout)
            e = np.array([x[:m] for x in ent])
            xs = np.arange(1, m + 1) * rollout
            line, = axes[3].plot(xs, e.mean(0), label=label)
            if len(ent) > 1:
                axes[3].fill_between(xs, e.min(0), e.max(0), alpha=0.2, color=line.get_color())
            print(f"policy entropy (max ln17 = 2.83): start {e.mean(0)[0]:.2f}, "
                  + ", ".join(f"{int(k)//1000}k {e.mean(0)[min(m - 1, int(k // cfg.rollout))]:.2f}"
                              for k in (100_000, 300_000, 500_000, 1_000_000) if k // cfg.rollout < m + 1))
    axes[3].axhline(np.log(17), color="grey", ls="--", lw=1)
    axes[3].set_title("Policy entropy (dashed: uniform = ln 17)")
    axes[3].set_xlabel("training steps")
    axes[3].grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    pathlib.Path(cfg.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg.out, dpi=120)
    plt.close(fig)
    print(f"\nFigure: {cfg.out}")
