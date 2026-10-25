import argparse
import json
import pathlib
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

SHOW = False


acronyms = {
    "random_agent": "RDN",
    "dqn_agent": "DQN",
    "ddqn_agent": "DDQN",
    "ppo_agent": "PPO",
    "own_agent": "DQN",
}

def read_pkl(path):
    events = []
    with (open(path, "rb")) as openfile:
        while True:
            try:
                events.append(pickle.load(openfile))
            except EOFError:
                break
    return events


def read_json(path):
    with (open(path, "rb")) as json_file:
        # json_list = list(json_file)
        data = [json.loads(line) for line in json_file]
    return data


def read_txt(path):
    items = []
    # open file and read the content in a list
    with open(path, 'r') as fp:
        for line in fp:
            x = line[:-1]
            items.append(x)
    return items


def read_crafter_logs(indir, outdir, clip=True):
    indir = pathlib.Path(indir)
    # read the pickles
    filenames = sorted(list(indir.glob("**/*/eval_stats.pkl")))
    # filenames = [filenames[0]]
    runs = []
    for idx, fn in enumerate(filenames):
        df = pd.DataFrame(columns=["step", "avg_return"], data=read_pkl(fn))
        df["run"] = idx
        runs.append(df)

    # some runs might not have finished and you might want to clip all of them
    # to the shortest one.
    if clip:
        min_len = min([len(run) for run in runs])
        runs = [run[:min_len] for run in runs]
        print(f"Clipped al runs to {min_len}.")

    # plot
    df = pd.concat(runs, ignore_index=True)
    sns.lineplot(x="step", y="avg_return", data=df)
    plt.savefig(outdir + "/avg_rewards.png")
    plt.show() if SHOW else plt.close("all")


def read_crafter_json(indir, outdir,  clip=True):
    indir = pathlib.Path(indir)

    filenames = sorted(list(indir.glob("**/*/stats.jsonl")))

    columns = ["length", "reward", "achievement_collect_coal",
               "achievement_collect_diamond", "achievement_collect_drink",
               "achievement_collect_iron", "achievement_collect_sapling",
               "achievement_collect_stone", "achievement_collect_wood",
               "achievement_defeat_skeleton", "achievement_defeat_zombie",
               "achievement_eat_cow", "achievement_eat_plant", "achievement_make_iron_pickaxe",
               "achievement_make_iron_sword", "achievement_make_stone_pickaxe", "achievement_make_stone_sword",
               "achievement_make_wood_pickaxe", "achievement_make_wood_sword", "achievement_place_furnace",
               "achievement_place_plant", "achievement_place_stone", "achievement_place_table", "achievement_wake_up"]

    runs = []
    for idx, fn in enumerate(filenames):
        ret = read_json(fn)
        df = pd.DataFrame(columns=columns, data=ret)
        df["run"] = idx
        runs.append(df)

    if clip:
        min_len = min([len(run) for run in runs])
        runs = [run[:min_len] for run in runs]
        print(f"Clipped al runs to {min_len}.")

    methods = {
        acronyms.get(pathlib.Path(outdir).name, pathlib.Path(outdir).name): {}
    }

    colors = ['g', 'r', 'c']
    length = df["length"]
    achievements = df.keys().drop(labels=["length", "reward", "run"])
    for key in achievements:
        percent = 100 * (np.array(df[key]) >= 1).mean()
        methods[acronyms.get(pathlib.Path(outdir).name, pathlib.Path(outdir).name)][key] = percent

    fig, ax = plt.subplots(figsize=(7, 3))
    centers = np.arange(len(methods[acronyms.get(pathlib.Path(outdir).name, pathlib.Path(outdir).name)]))
    width = 0.7
    for index, (key, sr) in enumerate(methods.items(), start=0):
        heights = np.array(list(sr.values()))
        # pos = centers + width * (0.5 / len(methods) + index / len(methods) - 0.5)
        pos = centers
        color = 'b'
        ax.bar(pos, heights, width / len(methods), label=key, color=color)

    names = [x[len('achievement_'):].replace('_', ' ').title() for x in methods[acronyms.get(pathlib.Path(outdir).name, pathlib.Path(outdir).name)].keys()]
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(
        axis='x', which='both', width=14, length=0.8, direction='inout')
    ax.set_xlim(centers[0] - 2 * (1 - width), centers[-1] + 2 * (1 - width))
    ax.set_xticks(centers + 0.0)
    ax.set_xticklabels(names, rotation=45, ha='right', rotation_mode='anchor')
    ax.set_ylabel('Success Rate (%)')
    ax.set_yscale('log')
    ax.set_ylim(0.01, 100)
    ax.set_yticks([0.01, 0.1, 1, 10, 100])
    ax.set_yticklabels('0.01 0.1 1 10 100'.split())
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.legend(
        loc='upper center', ncol=10, frameon=False, borderpad=1, borderaxespad=0)
    fig.savefig(outdir + "/spectrum_achievements.png")
    plt.show() if SHOW else plt.close("all")

    """
    score = np.array(list(methods[acronyms.get(pathlib.Path(outdir).name, pathlib.Path(outdir).name)].values()))
    mean = score.mean()
    std = score.std()
    print(mean)
    fig, ax = plt.subplots(figsize=(4, 3))
    centers = np.arange(len(methods))
    error_kw = dict(capsize=5, c='#000')
    ax.bar(centers, mean, yerr=std, color='b', error_kw=error_kw)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(
        axis='x', which='both', width=50, length=0.8, direction='inout')
    ax.set_xlim(centers[0] - 2 * (1 - width), centers[-1] + 2 * (1 - width))
    ax.set_xticks(centers + 0.0)
    ax.set_xticklabels(list(methods.keys()), rotation=45, ha='right', rotation_mode='anchor')
    ax.set_ylabel('Crafter Score (%)')
    ax.set_ylim(1, 20)

    # fig.tight_layout()
    plt.show() if SHOW else plt.close("all")
    
    """

def read_loss_log(indir, outdir):
    indir = pathlib.Path(indir)

    filenames = sorted(list(indir.glob("**/*/loss_log.txt")))

    lists = []

    for idx, fn in enumerate(filenames):
        ret = read_txt(fn)
        lists.append(np.asarray(ret, dtype=float))

    if not lists:
        print("No loss_log.txt found, skipping the loss plot.")
        return
    n = min(len(x) for x in lists)
    avg_data = np.average(np.asarray([x[:n] for x in lists]), axis=0)

    plt.style.use("ggplot")
    plt.figure()
    plt.plot(avg_data)
    plt.savefig(outdir + "/avg_loss.png")
    plt.show() if SHOW else plt.close("all")


def read_q_values_log(indir, outdir):
    indir = pathlib.Path(indir)

    filenames = sorted(list(indir.glob("**/*/q_values_log.txt")))

    lists = []

    for idx, fn in enumerate(filenames):
        ret = read_txt(fn)
        lists.append(np.asarray(ret, dtype=float))

    if not lists:
        return
    n = min(len(x) for x in lists)
    avg_data = np.average(np.asarray([x[:n] for x in lists]), axis=0)

    plt.style.use("ggplot")
    plt.figure()
    plt.plot(avg_data)
    plt.savefig(outdir + "/avg_q_values.png")
    plt.show() if SHOW else plt.close("all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logdir",
        default="logdir/random_agent",
        help="Path to the folder containing different runs.",
    )
    parser.add_argument("--out", default=None, help="Output folder (default: analysis/<agent folder name>).")
    parser.add_argument("--show", action="store_true", help="Also open the figures in a window.")
    cfg = parser.parse_args()
    SHOW = cfg.show

    name = pathlib.Path(cfg.logdir).name
    outdir = cfg.out or str(pathlib.Path(__file__).parent / name)
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
    print("Saving plots in " + outdir)

    read_loss_log(cfg.logdir, outdir)
    # read_q_values_log(cfg.logdir, outdir)
    read_crafter_json(cfg.logdir, outdir)
    read_crafter_logs(cfg.logdir, outdir)
