import argparse
import pickle
from pathlib import Path
import torch
from agents.dqn_agent import DQN_Agent
from agents.ddqn_agent import DDQN_Agent
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


def main(opt):

    _info(opt)

    opt.device = torch.device("cpu")

    env = Env("train", opt)
    eval_env = Env("eval", opt)

    # Rollout
    nb_rollouts = 2048

    # agent = RandomAgent(env.action_space.n)
    # checkpoints go inside the run folder, so parallel runs do not overwrite each other
    agent = PPO_Agent(n_actions=env.action_space.n, input_dims=[4, 64, 64], n_rollouts=nb_rollouts,
                      chkpt_dir=opt.logdir + "/checkpoints", ent_coef=opt.ent_coef)

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
                agent.learn()
            actor_losses_log.append(agent.avg_actor_loss)
            critic_losses_log.append(agent.avg_critic_loss)
            total_losses_log.append(agent.total_loss)
            entropy_log.append(agent.avg_entropy)

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
    with open(opt.logdir + '/entropy_log.txt', 'w') as fp:
        fp.writelines("%s\n" % item for item in entropy_log)


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
