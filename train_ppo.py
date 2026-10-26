import argparse
import pickle
from pathlib import Path

# With --num-envs > 1 on Windows, each worker process re-imports this file (under the name
# "__mp_main__"). Importing PyTorch in every worker loads the CUDA libraries again, which commits
# several GB of memory per process and exhausts the page file (WinError 1455). The heavy imports
# are therefore done only in the main process; workers only need Crafter (see src/vec_env.py).
if __name__ == "__main__":
    import torch
    # cuDNN benchmarks its convolution algorithms once per input shape and keeps the fastest.
    # Pure speed: the shapes here never change (batches of 1, N and 64), the maths is the same.
    torch.backends.cudnn.benchmark = True
    from agents.ppo_agent import PPO_Agent
    from src.crafter_wrapper import Env
    from src.progress import Progress


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
            action, prob, val = agent.act(obs)
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


def save_training_logs(actor_losses_log, critic_losses_log, total_losses_log, output_dir):
    with open(output_dir + '/actor_losses_log.txt', 'w') as fp:
        for item in actor_losses_log:
            # write each item on a new line
            fp.write("%s\n" % item)

    with open(output_dir + '/critic_losses_log.txt', 'w') as fp:
        for item in critic_losses_log:
            # write each item on a new line
            fp.write("%s\n" % item)

    with open(output_dir + '/total_losses_log.txt', 'w') as fp:
        for item in total_losses_log:
            # write each item on a new line
            fp.write("%s\n" % item)


def save_entropy_log(entropy_log, output_dir):
    with open(output_dir + '/entropy_log.txt', 'w') as fp:
        fp.writelines("%s\n" % item for item in entropy_log)


def make_agent(opt, n_rollouts, channels=None):
    fixes = bool(opt.fixes)
    channels = channels or opt.history_length
    batch_size = n_rollouts // opt.minibatches
    agent = PPO_Agent(n_actions=17, input_dims=[channels, 64, 64], n_rollouts=n_rollouts,
                      gamma=opt.gamma, gae_lambda=opt.gae_lambda, lr=opt.lr, n_epochs=opt.epochs,
                      batch_size=batch_size,
                      chkpt_dir=opt.logdir + "/checkpoints", ent_coef=opt.ent_coef,
                      norm_adv=fixes or bool(opt.norm_adv),
                      gae_cut_at_done=fixes or bool(opt.gae_cut_at_done),
                      bootstrap=fixes or bool(opt.bootstrap))
    print(f"PPO: input {channels}x64x64, gamma={opt.gamma}, lambda={opt.gae_lambda}, lr={opt.lr}, "
          f"rollout={n_rollouts}, epochs={opt.epochs} x {opt.minibatches} mini-batches of {batch_size}, "
          f"ent_coef={opt.ent_coef}, norm_adv={agent.norm_adv}, gae_cut_at_done={agent.gae_cut_at_done}, "
          f"bootstrap={agent.bootstrap}")
    return agent


def main(opt):

    _info(opt)

    opt.device = torch.device("cpu")

    env = Env("train", opt)
    eval_env = Env("eval", opt)

    # Rollout
    nb_rollouts = 2048

    # agent = RandomAgent(env.action_space.n)
    # checkpoints go inside the run folder, so parallel runs do not overwrite each other
    agent = make_agent(opt, n_rollouts=nb_rollouts)

    actor_losses_log = []
    critic_losses_log = []
    total_losses_log = []
    entropy_log = []

    # main loop
    ep_cnt, step_cnt, done = 0, 0, True

    display_interval = 20

    batch_episode_reward = 0.0
    batch_episode_count = 0

    progress = Progress(opt.steps, every=opt.log_every, device=agent.actor.device)

    while step_cnt < opt.steps or not done:
        if done:
            ep_cnt += 1
            batch_episode_count += 1
            if opt.log_episodes and batch_episode_count % display_interval == 0:
                print("[" + str(step_cnt) + "] Train mean_reward: " + str(
                    '%.1f' % (batch_episode_reward / batch_episode_count)))
                batch_episode_reward = 0.0
                batch_episode_count = 0
            with progress.time("env"):
                obs, done = env.reset(), False
                obs = obs.cpu().unsqueeze(0)

        with progress.time("act"):
            action, prob, val = agent.act(obs)
        with progress.time("env"):
            obs_, reward, done, info = env.step(action)
            obs_ = obs_.cpu().unsqueeze(0)
        batch_episode_reward += reward
        agent.remember(obs, action, prob, val, reward, done)
        step_cnt += 1
        progress.step(reward, done)

        if step_cnt % nb_rollouts == 0:
            with progress.time("learn"):
                agent.learn(last_obs=obs_)
            actor_losses_log.append(agent.avg_actor_loss)
            critic_losses_log.append(agent.avg_critic_loss)
            total_losses_log.append(agent.total_loss)
            entropy_log.append(agent.avg_entropy)
            # written after every update, so a stopped run keeps its logs
            save_training_logs(actor_losses_log, critic_losses_log, total_losses_log, opt.logdir)
            save_entropy_log(entropy_log, opt.logdir)

        obs = obs_

        # evaluate once in a while
        if step_cnt % opt.eval_interval == 0:
            with progress.time("eval"):
                eval(agent, eval_env, step_cnt, opt)

    progress.summary()
    print("Number of episodes : " + str(ep_cnt))
    agent.save_models()
    print("... saving logs...")
    save_training_logs(actor_losses_log, critic_losses_log, total_losses_log, opt.logdir)
    save_entropy_log(entropy_log, opt.logdir)


def main_vec(opt):
    """PPO with several environments in parallel (--num-envs N > 1).

    Same agent, same update (2048 transitions, 10 epochs, mini-batches of 64). What changes is
    how the 2048 transitions are collected: N environments x (2048 / N) steps instead of one
    environment x 2048 steps. Actions for the N environments are chosen in a single batched
    network pass, and the environments step in parallel worker processes.
    """
    import json
    import os
    import numpy as np
    from src.vec_env import VecCrafter, evaluate

    _info(opt)
    opt.device = torch.device("cpu")
    os.makedirs(opt.logdir, exist_ok=True)

    n_envs = opt.num_envs
    n_steps = opt.rollout // n_envs  # steps per environment per rollout
    assert n_steps * n_envs == opt.rollout, "--rollout must be a multiple of --num-envs"
    print(f"{n_envs} environments in {min(opt.num_workers or n_envs, n_envs)} worker processes, "
          f"{n_steps} steps each per rollout ({opt.rollout} transitions per update).")

    venv = VecCrafter(n_envs, num_workers=opt.num_workers, history_length=opt.history_length, rgb=bool(opt.rgb))
    channels = venv.obs_channels
    agent = make_agent(opt, n_rollouts=opt.rollout, channels=channels)

    stats_file = open(opt.logdir + "/stats.jsonl", "a")  # same format as crafter.Recorder
    actor_losses_log, critic_losses_log, total_losses_log, entropy_log = [], [], [], []
    progress = Progress(opt.steps, every=opt.log_every, device=agent.actor.device)
    obs = venv.observations()
    step_cnt, ep_cnt, next_eval = 0, 0, opt.eval_interval
    recent = []

    states = np.zeros((n_steps, n_envs, channels, 64, 64), dtype=np.float32)
    actions = np.zeros((n_steps, n_envs), dtype=np.int64)
    logps = np.zeros((n_steps, n_envs), dtype=np.float64)
    values = np.zeros((n_steps, n_envs), dtype=np.float64)
    rewards = np.zeros((n_steps, n_envs), dtype=np.float64)
    dones = np.zeros((n_steps, n_envs), dtype=bool)

    while step_cnt < opt.steps:
        for t in range(n_steps):
            with progress.time("act"):
                a, lp, v = agent.act_batch(obs)
            states[t], actions[t], logps[t], values[t] = obs, a, lp, v
            with progress.time("env"):
                r, d, finished = venv.step(a)
                obs = venv.observations()
            rewards[t], dones[t] = r, d
            for stats in finished:
                stats_file.write(json.dumps(stats) + "\n")
                ep_cnt += 1
                recent.append(stats["reward"])
                if opt.log_episodes and len(recent) == 20:
                    print(f"[{step_cnt}] Train mean_reward: {sum(recent) / len(recent):.1f}")
                    recent = []
            if finished:
                stats_file.flush()
            for i in range(n_envs):
                progress.step(r[i], d[i])
            step_cnt += n_envs

        with progress.time("learn"):
            # GAE per environment (each column is one environment's trajectory), same formula
            # as the single-environment code
            last_values = agent.value(obs) if agent.bootstrap else [None] * n_envs
            adv = np.stack([agent._advantages(rewards[:, i], values[:, i], dones[:, i], last_values[i])
                            for i in range(n_envs)], axis=1)
            agent.update(states.reshape(-1, channels, 64, 64), actions.reshape(-1),
                         logps.reshape(-1), values.reshape(-1), adv.reshape(-1))
        actor_losses_log.append(agent.avg_actor_loss)
        critic_losses_log.append(agent.avg_critic_loss)
        total_losses_log.append(agent.total_loss)
        entropy_log.append(agent.avg_entropy)
        save_training_logs(actor_losses_log, critic_losses_log, total_losses_log, opt.logdir)
        save_entropy_log(entropy_log, opt.logdir)

        if step_cnt >= next_eval:
            with progress.time("eval"):
                # same protocol as eval(): --eval-episodes episodes with the sampled policy,
                # but with this run's preprocessing (grayscale stack or colour)
                returns = evaluate(lambda o: agent.act_batch(o)[0], opt.eval_episodes,
                                   opt.history_length, bool(opt.rgb))
                _save_stats(returns, step_cnt, opt.logdir)
            next_eval += opt.eval_interval

    progress.summary()
    venv.close()
    stats_file.close()
    print("Number of episodes : " + str(ep_cnt))
    agent.save_models()
    print("... saving logs...")
    save_training_logs(actor_losses_log, critic_losses_log, total_losses_log, opt.logdir)
    save_entropy_log(entropy_log, opt.logdir)


def get_options():
    """ Configures a parser. Extend this with all the best performing hyperparameters of
        your agent as defaults.

        For devel purposes feel free to change the number of training steps and
        the evaluation interval.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default="logdir/ppo_agent/0")
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
        "--ent-coef",
        type=float,
        default=0.0,
        help="Entropy bonus weight in the PPO loss (0 = 2022 behaviour; 0.01 is the usual value).",
    )
    parser.add_argument("--fixes", type=int, default=0,
                        help="1 = enable the three PPO correctness fixes below (default 0 = 2022 behaviour).")
    parser.add_argument("--norm-adv", type=int, default=0, help="Normalise advantages per mini-batch.")
    parser.add_argument("--gae-cut-at-done", type=int, default=0, help="Stop the GAE sum at episode ends.")
    parser.add_argument("--bootstrap", type=int, default=0, help="Bootstrap V(s_T) at the end of a rollout.")
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of environments run in parallel (1 = 2022 behaviour: a single environment).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Worker processes hosting the environments (default: one per environment).",
    )
    parser.add_argument(
        "--rollout",
        type=int,
        default=2048,
        help="Transitions collected per PPO update, over all environments (2022: 2048).",
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
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor (2022: 0.99).")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda (2022: 0.95).")
    parser.add_argument("--lr", type=float, default=3e-4, help="Adam learning rate (2022: 3e-4).")
    parser.add_argument("--epochs", type=int, default=10, help="PPO epochs per rollout (2022: 10).")
    parser.add_argument("--minibatches", type=int, default=32,
                        help="Mini-batches per epoch; batch size = rollout / minibatches (2022: 32 -> 64).")
    parser.add_argument("--rgb", type=int, default=0,
                        help="1 = colour observations instead of grayscale (needs --num-envs > 1).")
    parser.add_argument("--preset", default=None, choices=["tuned"],
                        help="tuned = PPO settings of Moon et al. 2023 (see README); explicit flags are overridden.")
    opt = parser.parse_args()
    if opt.preset == "tuned":
        # Moon et al. 2023, Table 3: gamma 0.95, lambda 0.65, 4096 steps per rollout, 3 epochs x 8
        # mini-batches, entropy 0.01; plus the correctness fixes and colour input without stacking.
        opt.gamma, opt.gae_lambda, opt.rollout, opt.epochs, opt.minibatches = 0.95, 0.65, 4096, 3, 8
        opt.ent_coef, opt.fixes, opt.rgb, opt.history_length = 0.01, 1, 1, 1
        opt.num_envs = max(opt.num_envs, 16)
    if opt.rgb and opt.num_envs <= 1:
        parser.error("--rgb / --preset need --num-envs > 1")
    return opt


if __name__ == "__main__":
    options = get_options()
    if options.num_envs > 1:
        main_vec(options)
    else:
        main(options)
