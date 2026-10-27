"""Watch a trained PPO agent play: videos, causes of death, what it does before dying.

    python analysis/watch_agent.py --run logdir/ppo_tuned_resnet_1M/1
    python analysis/watch_agent.py --run logdir/ppo_tuned_resnet_1M/1 --episodes 200 --videos 4

Loads the run's settings (config.json) and its final network weights (checkpoints/), plays
--episodes episodes on new worlds (sampled policy, as in training evaluation; --greedy for the
most likely action) and writes to <run>/watch/:
  - report.json   per-episode data and the summary below;
  - video_*.mp4   the recorded episodes;
plays --videos more episodes recorded as mp4 (with a caption band), and prints a summary: score, length, causes of death, state of the agent when it dies,
day/night, when each achievement is unlocked.

Cause of death = what removed the last health point: lava, zombie, skeleton arrow, or a need
at 0 (food / drink / energy) with no attacker next to the agent.
"""
import argparse
import collections
import json
import os
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ------------------------------------------------------------------------------- agent ---
def load_policy(run_dir, greedy=False):
    """Returns act(observations [1, C, 64, 64] float32) -> action int, and the run config."""
    import torch
    cfg = json.load(open(run_dir / "config.json"))
    channels = (3 if cfg.get("rgb") else 1) * cfg.get("history_length", 4)
    ckpt_dir = str(run_dir / "checkpoints")
    if cfg.get("arch") == "impala":
        from networks.impala import ImpalaActorCritic
        net = ImpalaActorCritic(17, [channels, 64, 64], 3e-4, chkpt_dir=ckpt_dir)
        net.load_checkpoint()
        net.eval()
        policy = lambda s: net(s)[0]  # noqa: E731
        device = net.device
    else:
        from networks.actor_network import ActorNetwork
        net = ActorNetwork(17, [channels, 64, 64], 3e-4, chkpt_dir=ckpt_dir)
        net.load_checkpoint()
        net.eval()
        policy = net
        device = net.device

    def act(obs):
        with torch.no_grad():
            dist = policy(torch.as_tensor(obs).to(device))
            a = dist.probs.argmax(-1) if greedy else dist.sample()
        return int(a.item())
    return act, cfg


# ------------------------------------------------------------------------ one episode ---
def cause_of_death(env, prev_inventory):
    from crafter import objects
    player, world = env._player, env._world
    if world[player.pos][0] == "lava":
        return "lava"
    near = [o for o in world.objects if o is not player and player.distance(o) <= 1]
    if any(isinstance(o, objects.Zombie) for o in near):
        return "zombie (asleep)" if player.sleeping or prev_inventory.get("_sleeping") else "zombie"
    if any(isinstance(o, objects.Arrow) for o in near):
        return "skeleton arrow"
    starving = [k for k in ("food", "drink", "energy") if prev_inventory[k] == 0]
    if starving:
        return "no " + " + ".join(starving)
    if any(isinstance(o, objects.Skeleton) for o in world.objects if player.distance(o) <= 4):
        return "skeleton arrow"
    return "other"


def play_episode(env, frames, act, record=False, size=384):
    """Plays one episode. Returns its record and, if asked, the rendered frames."""
    from src.vec_env import FrameStack  # noqa: F401  (frames is a FrameStack)
    rgb = env.reset()
    frames.new_episode(0, rgb)
    done, length, ret = False, 0, 0.0
    unlocked_at, video, actions = {}, [], collections.Counter()
    while not done:
        if record:
            video.append((env.render((size, size)), length, list(unlocked_at)))
        player = env._player
        prev = dict(player.inventory)
        prev["_sleeping"] = player.sleeping
        a = act(frames.observations())
        actions[a] += 1
        rgb, reward, done, info = env.step(a)
        frames.push(0, rgb)
        length += 1
        ret += reward
        for name in info.get("achievements", {}):
            if info["achievements"][name] > 0 and name not in unlocked_at:
                unlocked_at[name] = length
    dead = env._player.health <= 0
    record_ = {
        "length": length, "return": round(ret, 2), "dead": bool(dead),
        "cause": cause_of_death(env, prev) if dead else "survived",
        "night": bool(env._world.daylight < 0.3),
        "inventory_at_end": {k: int(v) for k, v in env._player.inventory.items()},
        "unlocked_at": unlocked_at,
        "actions": {int(k): int(v) for k, v in actions.items()},
    }
    return record_, video


def annotate(img, step, unlocked):
    from PIL import Image, ImageDraw, ImageFont
    band = 64
    canvas = Image.new("RGB", (img.shape[1], img.shape[0] + band), (20, 20, 24))
    canvas.paste(Image.fromarray(img), (0, 0))
    d = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except OSError:
        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except OSError:
            font = ImageFont.load_default()
    d.text((10, img.shape[0] + 8), f"step {step:4d}   achievements {len(unlocked):2d}/22", fill=(235, 235, 235), font=font)
    if unlocked:
        d.text((10, img.shape[0] + 32), "last: " + unlocked[-1].replace("_", " "), fill=(250, 200, 90), font=font)
    return np.array(canvas)


# --------------------------------------------------------------------------- summary ---
def summarise(episodes):
    from crafter import constants
    n = len(episodes)
    names = list(constants.achievements)
    rates = {a: 100 * np.mean([a in e["unlocked_at"] for e in episodes]) for a in names}
    score = float(np.exp(np.mean(np.log1p(list(rates.values())))) - 1)
    causes = collections.Counter(e["cause"] for e in episodes)
    deaths = [e for e in episodes if e["dead"]]
    when = {a: float(np.median([e["unlocked_at"][a] for e in episodes if a in e["unlocked_at"]]))
            for a in names if any(a in e["unlocked_at"] for e in episodes)}
    at_death = {k: float(np.mean([e["inventory_at_end"][k] for e in deaths])) if deaths else None
                for k in ("food", "drink", "energy")}
    acts = collections.Counter()
    for e in episodes:
        acts.update({int(k): v for k, v in e["actions"].items()})
    total = sum(acts.values()) or 1
    return {
        "episodes": n, "score": score,
        "length_mean": float(np.mean([e["length"] for e in episodes])),
        "length_median": float(np.median([e["length"] for e in episodes])),
        "return_mean": float(np.mean([e["return"] for e in episodes])),
        "achievements_per_episode": float(np.mean([len(e["unlocked_at"]) for e in episodes])),
        "success_rates": rates,
        "median_unlock_step": dict(sorted(when.items(), key=lambda kv: kv[1])),
        "causes_of_death": {k: 100 * v / n for k, v in causes.most_common()},
        "deaths_at_night": 100 * np.mean([e["night"] for e in deaths]) if deaths else None,
        "needs_at_death_mean": at_death,
        "actions": {constants.actions[k]: round(100 * v / total, 1) for k, v in acts.most_common()},
    }


def print_summary(s):
    print(f"\n{s['episodes']} episodes | score {s['score']:.1f}% | length mean {s['length_mean']:.0f} "
          f"(median {s['length_median']:.0f}) | reward {s['return_mean']:.2f} | "
          f"{s['achievements_per_episode']:.1f} achievements per episode")
    print("\nCauses of death (% of episodes):")
    for k, v in s["causes_of_death"].items():
        print(f"  {k:24s} {v:5.1f}%")
    if s["deaths_at_night"] is not None:
        print(f"  deaths at night: {s['deaths_at_night']:.0f}% | needs at death (mean, max 9): "
              + ", ".join(f"{k} {v:.1f}" for k, v in s["needs_at_death_mean"].items()))
    print("\nAchievements: success rate, median step of unlock")
    for a, r in sorted(s["success_rates"].items(), key=lambda kv: -kv[1]):
        when = s["median_unlock_step"].get(a)
        print(f"  {a:20s} {r:5.1f}%" + (f"   step {when:.0f}" if when is not None else ""))
    print("\nActions (% of steps): " + ", ".join(f"{k} {v}" for k, v in s["actions"].items()))


# ------------------------------------------------------------------------------ main ---
if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--run", required=True, help="Run folder (contains config.json and checkpoints/).")
    pa.add_argument("--episodes", type=int, default=100)
    pa.add_argument("--videos", type=int, default=3, help="Extra episodes recorded as mp4 videos.")
    pa.add_argument("--greedy", action="store_true", help="Most likely action instead of sampling.")
    pa.add_argument("--seed", type=int, default=12345, help="World seed (new worlds, never seen in training).")
    cfg_args = pa.parse_args()

    import crafter
    from src.vec_env import FrameStack

    run_dir = pathlib.Path(cfg_args.run)
    act_fn, cfg = load_policy(run_dir, cfg_args.greedy)
    out = run_dir / "watch"
    out.mkdir(exist_ok=True)

    env = crafter.Env(seed=cfg_args.seed)
    frames = FrameStack(1, cfg.get("history_length", 4), bool(cfg.get("rgb")))
    episodes = []
    for i in range(cfg_args.episodes):
        rec, _ = play_episode(env, frames, act_fn)
        episodes.append(rec)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{cfg_args.episodes} episodes", flush=True)

    summary = summarise(episodes)
    print_summary(summary)
    json.dump({"run": str(run_dir), "greedy": cfg_args.greedy, "summary": summary, "episodes": episodes},
              open(out / "report.json", "w"), indent=1)

    if cfg_args.videos:
        # a few extra episodes on new worlds, recorded (rendering every frame is slow, so the
        # statistics above are computed without video)
        import imageio
        env_v = crafter.Env(seed=cfg_args.seed + 1)
        for k in range(cfg_args.videos):
            rec, video = play_episode(env_v, frames, act_fn, record=True)
            name = out / f"video_{k + 1}_{len(rec['unlocked_at'])}ach_{rec['length']}steps_{rec['cause'].replace(' ', '_')}.mp4"
            imageio.mimwrite(name, [annotate(*f) for f in video], fps=12, macro_block_size=16)
            print(f"video: {name}")
    print(f"\nReport: {out / 'report.json'}")
