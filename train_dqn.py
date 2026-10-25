import argparse
import pickle
from pathlib import Path
import torch
from agents.dqn_agent import DQN_Agent
from agents.ddqn_agent import DDQN_Agent
from src.crafter_wrapper import Env
from src.progress import Progress

GAMMA = 0.99
BATCH_SIZE = 32
MAX_MEMORY = 100_000
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 1e-6
LR = 0.00025  # 0.001


def _save_stats(episodic_returns, crt_step, path):
    # save the evaluation stats
    episodic_returns = torch.tensor(episodic_returns)
    avg_return = episodic_returns.mean().item()
    print(
        "[{:06d}] eval results: R/ep={:03.2f}, std={:03.2f}.".format(
            crt_step, avg_return, episodic_returns.std().item()
        )
    )
    with open(path + "/eval_stats.pkl", "ab") as f:
        pickle.dump({"step": crt_step, "avg_return": avg_return}, f)


def eval(agent, env, crt_step, opt):
    """ Use the greedy, deterministic policy, not the epsilon-greedy policy you
    might use during training.
    """
    episodic_returns = []
    for _ in range(opt.eval_episodes):
        obs, done = env.reset(), False
        obs = obs.cpu().unsqueeze(0)
        episodic_returns.append(0)
        while not done:
            action = agent.act(obs, train=False)
            obs, reward, done, info = env.step(action)
            obs = obs.cpu().unsqueeze(0)
            episodic_returns[-1] += reward

    _save_stats(episodic_returns, crt_step, opt.logdir)


def _info(opt):
    try:
        int(opt.logdir.split("/")[-1])
    except:
        print(
            "Warning, logdir path should end in a number indicating a separate"
            + "training run, else the results might be overwritten."
        )
    if Path(opt.logdir).exists():
        print("Warning! Logdir path exists, results can be corrupted.")
    print(f"Saving results in {opt.logdir}.")
    print(
        f"Observations are of dims ({opt.history_length},64,64),"
        + "with values between 0 and 1."
    )


def save_training_logs(loss_log, q_values_log, output_dir):
    with open(output_dir + '/loss_log.txt', 'w') as fp:
        for item in loss_log:
            # write each item on a new line
            fp.write("%s\n" % item)

    with open(output_dir + '/q_values_log.txt', 'w') as fp:
        for item in q_values_log:
            # write each item on a new line
            fp.write("%s\n" % item)


def main(opt):
    # Verify the informations
    _info(opt)

    opt.device = torch.device("cpu")

    env = Env("train", opt)
    eval_env = Env("eval", opt)

    # agent = RandomAgent(env.action_space.n)
    agent_cls = {"dqn": DQN_Agent, "ddqn": DDQN_Agent}[opt.agent]
    agent = agent_cls(gamma=GAMMA, epsilon=EPSILON_START, batch_size=BATCH_SIZE, n_actions=env.action_space.n,
                      eps_end=EPSILON_END, input_dims=[4, 64, 64], lr=LR, max_mem_size=MAX_MEMORY)

    loss_log = []
    q_values_log = []
    log_interval = 10000

    # main loop
    ep_cnt, step_cnt, done = 0, 0, True

    batch_episode_reward = 0.0
    batch_episode_count = 0

    progress = Progress(opt.steps, every=opt.log_every, device=agent.device)

    while step_cnt < opt.steps or not done:
        if done:
            ep_cnt += 1
            batch_episode_count += 1
            if opt.log_episodes and batch_episode_count % 20 == 0:
                print("[" + str(step_cnt) + "] Train mean_reward: " + str(
                    '%.1f' % (batch_episode_reward / batch_episode_count)))
                batch_episode_reward = 0.0
                batch_episode_count = 0
            with progress.time("env"):
                obs, done = env.reset(), False
                obs = obs.cpu().unsqueeze(0)

        with progress.time("act"):
            action = agent.act(obs)
        with progress.time("env"):
            obs_, reward, done, info = env.step(action)
            obs_ = obs_.cpu().unsqueeze(0)
        batch_episode_reward += reward
        with progress.time("replay"):
            agent.store_transition(obs, action, reward, obs_, done)
        with progress.time("learn"):
            agent.learn()
        obs = obs_
        step_cnt += 1
        progress.step(reward, done)

        # save logs once in a while
        if step_cnt % log_interval == 0:
            loss_log = loss_log + agent.loss_log
            q_values_log = q_values_log + agent.q_values_log
            agent.remove_logs()

        # evaluate once in a while
        if step_cnt % opt.eval_interval == 0:
            with progress.time("eval"):
                eval(agent, eval_env, step_cnt, opt)

    progress.summary()
    print("Number of episodes : " + str(ep_cnt))
    save_training_logs(loss_log, q_values_log, opt.logdir)


def get_options():
    """ Configures a parser. Extend this with all the best performing hyperparameters of
        your agent as defaults.

        For devel purposes feel free to change the number of training steps and
        the evaluation interval.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["dqn", "ddqn"], default="ddqn")
    parser.add_argument("--logdir", default="logdir/ddqn_agent/0")
    parser.add_argument(
        "--steps",
        type=int,
        metavar="STEPS",
        default=1_000_000,
        help="Total number of training steps.",
    )
    parser.add_argument(
        "-hist-len",
        "--history-length",
        default=4,
        type=int,
        help="The number of frames to stack when creating an observation.",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=100_000,
        metavar="STEPS",
        help="Number of training steps between evaluations",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=20,
        metavar="N",
        help="Number of evaluation episodes to average over",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=5_000,
        metavar="STEPS",
        help="Print a progress line (speed, ETA, reward, time split) every N steps; 0 to disable.",
    )
    parser.add_argument(
        "--log-episodes",
        type=int,
        default=1,
        help="Also print the original 'Train mean_reward' line every 20 episodes (1) or not (0).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(get_options())
