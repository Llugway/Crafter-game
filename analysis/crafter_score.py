"""Crafter score of training runs, computed like the official benchmark.

Score = geometric mean of the 22 achievement success rates (in %), offset by 1:
    exp(mean(ln(1 + s_i))) - 1
computed over all *training* episodes logged in stats.jsonl (by crafter.Recorder),
optionally limited to the first N environment steps (the benchmark uses 1M).

    python analysis/crafter_score.py --logdir results/2022/runs
    python analysis/crafter_score.py --logdir logdir/ppo_agent --budget 1000000
"""
import argparse
import json
import pathlib

import numpy as np


def run_score(path, budget=None):
    eps = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    if not eps:
        return None
    keys = sorted(k for k in eps[0] if k.startswith("achievement_"))
    steps = np.cumsum([e["length"] for e in eps])
    if budget:
        eps = [e for e, s in zip(eps, steps) if s <= budget]
    rates = np.array([100 * np.mean([e[k] >= 1 for e in eps]) for k in keys])
    return {
        "episodes": len(eps),
        "steps": int(steps[min(len(eps), len(steps)) - 1]),
        "score": float(np.exp(np.mean(np.log1p(rates))) - 1),
        "reward_last100": float(np.mean([e["reward"] for e in eps[-100:]])),
        "rates": dict(zip([k[len("achievement_"):] for k in keys], rates.round(1).tolist())),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default="logdir", help="Folder searched recursively for stats.jsonl files.")
    parser.add_argument("--budget", type=int, default=None, help="Only count episodes within the first N steps.")
    parser.add_argument("--rates", action="store_true", help="Also print each achievement's success rate.")
    cfg = parser.parse_args()

    files = sorted(pathlib.Path(cfg.logdir).glob("**/stats.jsonl"))
    if not files:
        print(f"No stats.jsonl under {cfg.logdir}")
    print(f"{'run':40s} {'episodes':>8s} {'steps':>9s} {'score %':>8s} {'reward(last100)':>16s}")
    for f in files:
        r = run_score(f, cfg.budget)
        if r is None:
            continue
        name = str(f.parent.relative_to(cfg.logdir))
        print(f"{name:40s} {r['episodes']:8d} {r['steps']:9d} {r['score']:8.2f} {r['reward_last100']:16.2f}")
        if cfg.rates:
            print("   " + ", ".join(f"{k}={v}" for k, v in sorted(r["rates"].items(), key=lambda kv: -kv[1])))
